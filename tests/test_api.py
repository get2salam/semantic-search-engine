"""
API Integration Tests
=====================
Tests for the FastAPI REST endpoints using httpx TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture(scope="module")
def client():
    """Create a test client with application lifespan."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def seeded_client(client):
    """Client with pre-loaded documents."""
    docs = [
        "Machine learning is a branch of artificial intelligence",
        "Python is a versatile programming language",
        "Deep neural networks learn hierarchical representations",
        "JavaScript powers interactive web applications",
        "Natural language processing enables text understanding",
    ]
    resp = client.post("/documents", json={"documents": docs})
    assert resp.status_code == 201
    return client


class TestHealthEndpoints:
    """Health and info endpoints."""

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_stats_returns_model_info(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "embedding_dim" in data
        assert isinstance(data["total_documents"], int)


class TestDocumentManagement:
    """Document CRUD operations."""

    def test_add_documents(self, client):
        resp = client.post(
            "/documents",
            json={"documents": ["Test document one", "Test document two"]},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["new_documents"] == 2

    def test_add_empty_documents_fails(self, client):
        resp = client.post("/documents", json={"documents": []})
        assert resp.status_code == 422  # Validation error

    def test_document_count(self, client):
        resp = client.get("/documents/count")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 0

    def test_clear_documents(self, client):
        # Add, then clear
        client.post("/documents", json={"documents": ["throwaway"]})
        resp = client.delete("/documents")
        assert resp.status_code == 200
        assert resp.json()["total_documents"] == 0


class TestSearchEndpoints:
    """Search functionality via API."""

    def test_post_search(self, seeded_client):
        resp = seeded_client.post(
            "/search",
            json={"query": "artificial intelligence", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "artificial intelligence"
        assert len(data["results"]) <= 3
        assert data["results"][0]["rank"] == 1
        assert data["elapsed_ms"] >= 0

    def test_get_search(self, seeded_client):
        resp = seeded_client.get("/search", params={"q": "programming", "top_k": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) <= 2

    def test_search_with_threshold(self, seeded_client):
        resp = seeded_client.post(
            "/search",
            json={"query": "AI", "top_k": 10, "threshold": 0.5},
        )
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert result["score"] >= 0.5

    def test_search_results_sorted_by_score(self, seeded_client):
        resp = seeded_client.post("/search", json={"query": "machine learning", "top_k": 5})
        scores = [r["score"] for r in resp.json()["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_batch_search(self, seeded_client):
        resp = seeded_client.post(
            "/search/batch",
            json={"queries": ["AI", "Python", "web"], "top_k": 2},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert data["elapsed_ms"] >= 0

    def test_search_empty_index_returns_400(self, client):
        # Clear first
        client.delete("/documents")
        resp = client.post("/search", json={"query": "test"})
        assert resp.status_code == 400


class TestRequestValidation:
    """Input validation and edge cases."""

    def test_missing_query_returns_422(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422

    def test_negative_top_k_returns_422(self, client):
        resp = client.post("/search", json={"query": "test", "top_k": -1})
        assert resp.status_code == 422

    def test_threshold_out_of_range_returns_422(self, client):
        resp = client.post("/search", json={"query": "test", "threshold": 2.0})
        assert resp.status_code == 422

    def test_batch_too_large_returns_400(self, seeded_client):
        queries = [f"query_{i}" for i in range(200)]
        resp = seeded_client.post("/search/batch", json={"queries": queries, "top_k": 1})
        assert resp.status_code == 400


class TestMiddleware:
    """Middleware behaviour."""

    def test_server_timing_header(self, client):
        resp = client.get("/health")
        assert "server-timing" in resp.headers
        assert "x-request-time-ms" in resp.headers

    def test_cors_headers_present(self, client):
        resp = client.options(
            "/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert "access-control-allow-origin" in resp.headers

    def test_security_headers_present(self, client):
        resp = client.get("/health")
        # Security headers are enabled by default via settings
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "referrer-policy" in resp.headers
        assert "permissions-policy" in resp.headers

    def test_request_id_header_minted_when_absent(self, client):
        resp = client.get("/health")
        assert resp.headers.get("x-request-id")
        # Minted IDs are 32 hex chars (uuid4 .hex)
        req_id = resp.headers["x-request-id"]
        assert len(req_id) == 32
        int(req_id, 16)  # must be valid hex

    def test_request_id_header_propagated_when_provided(self, client):
        resp = client.get("/health", headers={"X-Request-ID": "my-trace-42"})
        assert resp.headers["x-request-id"] == "my-trace-42"


class TestRateLimiting:
    """Token-bucket rate limiter wired into middleware."""

    def test_rate_limit_rejects_after_burst(self, monkeypatch):
        # Enable rate limiting with a tiny capacity via a fresh app import cycle
        import importlib

        import api as api_module

        monkeypatch.setattr(api_module.settings, "rate_limit_enabled", True, raising=False)
        # Reset any pre-existing limiter instance
        api_module._rate_limiter = None
        monkeypatch.setattr(api_module.settings, "rate_limit_per_minute", 2, raising=False)

        with TestClient(api_module.app) as c:
            r1 = c.get("/stats")
            r2 = c.get("/stats")
            r3 = c.get("/stats")
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert r3.json()["detail"] == "Rate limit exceeded"
            assert "retry-after" in {k.lower() for k in r3.headers}

        # Cleanup so subsequent tests are not rate-limited
        api_module._rate_limiter = None
        importlib.invalidate_caches()
