"""
REST API for Semantic Search Engine
====================================
Production-grade FastAPI application with health checks, structured logging,
request validation, CORS, and OpenAPI documentation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000
    # or
    python api.py
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from config import get_settings
from logging_config import (
    configure_logging,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from metrics import MetricsRegistry
from rate_limit import TokenBucketRateLimiter
from semantic_search import SemanticSearchEngine

# ---------------------------------------------------------------------------
# Metrics registry
# ---------------------------------------------------------------------------
registry = MetricsRegistry()
REQ_TOTAL = registry.counter(
    "sse_requests_total",
    "Total API requests handled, labelled by method, path, and status",
    labels=("method", "path", "status"),
)
REQ_LATENCY = registry.histogram(
    "sse_request_latency_seconds",
    "API request latency in seconds",
    labels=("method", "path"),
)
SEARCHES_TOTAL = registry.counter(
    "sse_searches_total",
    "Total /search requests (including GET alias and batch children)",
)
DOCS_INDEXED = registry.gauge(
    "sse_documents_indexed",
    "Current number of documents in the search index",
)
RATE_LIMITED_TOTAL = registry.counter(
    "sse_rate_limited_total",
    "Total requests rejected by the rate limiter",
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("sse.api")


def _configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    configure_logging(level=level, json_format=json_format)


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class DocumentsRequest(BaseModel):
    """Request body for adding documents."""

    documents: list[str] = Field(
        ..., min_length=1, max_length=10_000, description="List of text documents to index"
    )
    batch_size: int = Field(default=64, ge=1, le=1024, description="Encoding batch size")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": [
                        "Machine learning is fascinating",
                        "Python is a great language",
                    ],
                    "batch_size": 64,
                }
            ]
        }
    }


class SearchRequest(BaseModel):
    """Request body for a single search query."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query text")
    top_k: int = Field(default=5, ge=1, description="Number of results to return")
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity score"
    )
    mmr_lambda: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "If set, apply Maximal Marginal Relevance diversification. "
            "1.0 = pure relevance (same as omitting it), 0.0 = maximum "
            "diversity, 0.5 = balanced."
        ),
    )
    mmr_candidate_k: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Size of the candidate pool to pull before MMR re-ranking. "
            "Defaults to max(4*top_k, 25). Ignored when mmr_lambda is unset."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "artificial intelligence", "top_k": 3, "threshold": 0.3},
                {
                    "query": "climate change",
                    "top_k": 5,
                    "mmr_lambda": 0.5,
                    "mmr_candidate_k": 50,
                },
            ]
        }
    }


class BatchSearchRequest(BaseModel):
    """Request body for batch search."""

    queries: list[str] = Field(..., min_length=1, description="List of search queries")
    top_k: int = Field(default=5, ge=1, description="Results per query")

    model_config = {
        "json_schema_extra": {
            "examples": [{"queries": ["AI models", "web development"], "top_k": 3}]
        }
    }


class SearchResult(BaseModel):
    """A single search result."""

    document: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    """Response for a search query."""

    query: str
    results: list[SearchResult]
    total_documents: int
    elapsed_ms: float


class BatchSearchResponse(BaseModel):
    """Response for batch search."""

    results: list[SearchResponse]
    elapsed_ms: float


class IndexStats(BaseModel):
    """Statistics about the current index."""

    total_documents: int
    model_name: str
    embedding_dim: int
    faiss_enabled: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    model_loaded: bool
    documents_indexed: int


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

# Module-level engine reference (populated during lifespan)
engine: SemanticSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle handler."""
    global engine
    settings = get_settings()
    _configure_logging(settings.log_level, json_format=settings.log_json)

    logger.info("Initializing search engine (model=%s) ...", settings.model_name)
    engine = SemanticSearchEngine(
        model_name=settings.model_name,
        use_faiss=settings.use_faiss,
        normalize_embeddings=settings.normalize_embeddings,
    )

    # Optionally load a pre-built index
    if settings.index_path:
        try:
            engine = SemanticSearchEngine.load(settings.index_path)
            logger.info("Loaded index from %s (%d docs)", settings.index_path, len(engine))
        except Exception as exc:
            logger.warning("Could not load index from %s: %s", settings.index_path, exc)

    logger.info("Search engine ready")
    yield

    # Shutdown: auto-save if configured
    if settings.auto_save_path and engine and len(engine) > 0:
        logger.info("Auto-saving index to %s ...", settings.auto_save_path)
        engine.save(settings.auto_save_path)

    logger.info("Shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

settings = get_settings()

app = FastAPI(
    title="Semantic Search Engine API",
    description=(
        "A production-ready REST API for semantic similarity search, "
        "powered by sentence-transformers and FAISS."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# --- Middleware ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Rate limiter (lazy-init so tests can toggle settings)
_rate_limiter: TokenBucketRateLimiter | None = None


def _get_rate_limiter() -> TokenBucketRateLimiter | None:
    global _rate_limiter
    if not settings.rate_limit_enabled:
        return None
    if _rate_limiter is None:
        _rate_limiter = TokenBucketRateLimiter.per_minute(settings.rate_limit_per_minute)
    return _rate_limiter


# Paths that are always exempt from rate limiting (liveness, metrics, docs)
_RATE_LIMIT_EXEMPT_PATHS = frozenset(
    {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
)


def _client_key(request: Request) -> str:
    """Resolve a stable client identity for rate limiting."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Common security headers applied to every response when enabled
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """
    End-to-end request handling:
    - assigns / propagates an X-Request-ID for log correlation
    - enforces rate limiting (opt-in)
    - emits Server-Timing + security headers
    """
    # Preserve an upstream request ID if the caller supplied one, else mint a new one
    incoming_id = request.headers.get("x-request-id")
    request_id = incoming_id if incoming_id else new_request_id()
    token = set_request_id(request_id)

    try:
        # Rate limit check (before work)
        limiter = _get_rate_limiter()
        if limiter is not None and request.url.path not in _RATE_LIMIT_EXEMPT_PATHS:
            key = _client_key(request)
            if not limiter.allow(key):
                retry_after = max(1, int(limiter.retry_after_seconds(key)))
                RATE_LIMITED_TOTAL.inc()
                logger.warning(
                    "rate_limited", extra={"client": key, "path": request.url.path}
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded", "retry_after_seconds": retry_after},
                    headers={
                        "Retry-After": str(retry_after),
                        "X-Request-ID": request_id,
                    },
                )

        start = time.perf_counter()
        response = await call_next(request)
        elapsed_s = time.perf_counter() - start
        elapsed_ms = elapsed_s * 1000

        response.headers["Server-Timing"] = f"total;dur={elapsed_ms:.1f}"
        response.headers["X-Request-Time-Ms"] = f"{elapsed_ms:.1f}"
        response.headers["X-Request-ID"] = request_id

        if settings.security_headers_enabled:
            for header, value in _SECURITY_HEADERS.items():
                response.headers.setdefault(header, value)

        # Record metrics (skip /metrics itself to avoid unbounded cardinality)
        path = request.url.path
        if path != "/metrics":
            REQ_TOTAL.inc(method=request.method, path=path, status=str(response.status_code))
            REQ_LATENCY.observe(elapsed_s, method=request.method, path=path)

        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )
        return response
    finally:
        reset_request_id(token)


# --- Exception handlers ---


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _get_engine() -> SemanticSearchEngine:
    """Return the engine or raise 503 if not ready."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Search engine is not initialized")
    return engine


# --- Health & Info ---


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for load balancers and orchestrators.

    Returns the service status, model state, and document count.
    """
    eng = _get_engine()
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=eng.model is not None,
        documents_indexed=len(eng),
    )


@app.get("/stats", response_model=IndexStats, tags=["Health"])
async def index_stats():
    """Return statistics about the current search index."""
    eng = _get_engine()
    return IndexStats(
        total_documents=len(eng),
        model_name=eng.model_name,
        embedding_dim=eng.embedding_dim,
        faiss_enabled=eng.use_faiss,
    )


@app.get("/metrics", tags=["Health"], response_class=PlainTextResponse)
async def metrics_endpoint():
    """
    Expose Prometheus-style metrics.

    Scrapers (Prometheus, Grafana Agent, Datadog OpenMetrics) should
    point at this endpoint. The response content-type follows the
    Prometheus text exposition format.
    """
    if engine is not None:
        DOCS_INDEXED.set(len(engine))
    return PlainTextResponse(
        registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# --- Document Management ---


@app.post("/documents", tags=["Documents"], status_code=201)
async def add_documents(body: DocumentsRequest):
    """
    Add documents to the search index.

    Documents are encoded into embeddings and added to the vector store.
    Duplicate detection is **not** performed -- the caller is responsible
    for de-duplication.
    """
    eng = _get_engine()
    count_before = len(eng)
    eng.add_documents(body.documents, batch_size=body.batch_size, show_progress=False)

    # Auto-save
    if settings.auto_save_path:
        eng.save(settings.auto_save_path)

    return {
        "message": f"Added {len(body.documents)} documents",
        "total_documents": len(eng),
        "new_documents": len(eng) - count_before,
    }


@app.get("/documents/count", tags=["Documents"])
async def document_count():
    """Return the number of indexed documents."""
    eng = _get_engine()
    return {"count": len(eng)}


@app.delete("/documents", tags=["Documents"])
async def clear_documents():
    """Remove all documents and embeddings from the index."""
    eng = _get_engine()
    eng.clear()
    return {"message": "Index cleared", "total_documents": 0}


# --- Search ---


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(body: SearchRequest):
    """
    Perform a semantic search over indexed documents.

    Returns the top-k most similar documents ranked by cosine similarity.
    """
    eng = _get_engine()

    if len(eng) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Add documents first.")

    top_k = min(body.top_k, settings.max_top_k)

    start = time.perf_counter()
    raw_results = eng.search(
        body.query,
        top_k=top_k,
        threshold=body.threshold,
        mmr_lambda=body.mmr_lambda,
        mmr_candidate_k=body.mmr_candidate_k,
    )
    elapsed = (time.perf_counter() - start) * 1000

    SEARCHES_TOTAL.inc()

    results = [
        SearchResult(document=doc, score=round(score, 4), rank=i + 1)
        for i, (doc, score) in enumerate(raw_results)
    ]

    return SearchResponse(
        query=body.query,
        results=results,
        total_documents=len(eng),
        elapsed_ms=round(elapsed, 2),
    )


@app.post("/search/batch", response_model=BatchSearchResponse, tags=["Search"])
async def search_batch(body: BatchSearchRequest):
    """
    Batch search: run multiple queries in a single request.

    Useful for comparing multiple queries or building recommendation matrices.
    """
    eng = _get_engine()

    if len(eng) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Add documents first.")

    if len(body.queries) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(f"Batch size {len(body.queries)} exceeds maximum of {settings.max_batch_size}"),
        )

    top_k = min(body.top_k, settings.max_top_k)

    overall_start = time.perf_counter()
    responses: list[SearchResponse] = []

    for query in body.queries:
        start = time.perf_counter()
        raw_results = eng.search(query, top_k=top_k)
        elapsed = (time.perf_counter() - start) * 1000
        SEARCHES_TOTAL.inc()

        results = [
            SearchResult(document=doc, score=round(score, 4), rank=i + 1)
            for i, (doc, score) in enumerate(raw_results)
        ]
        responses.append(
            SearchResponse(
                query=query,
                results=results,
                total_documents=len(eng),
                elapsed_ms=round(elapsed, 2),
            )
        )

    total_elapsed = (time.perf_counter() - overall_start) * 1000
    return BatchSearchResponse(results=responses, elapsed_ms=round(total_elapsed, 2))


# --- Convenience GET search ---


@app.get("/search", response_model=SearchResponse, tags=["Search"])
async def search_get(
    q: str = Query(..., min_length=1, max_length=10_000, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of results"),
    threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    mmr_lambda: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
        description="Enable MMR (1=relevance, 0=diversity, 0.5=balanced)",
    ),
    mmr_candidate_k: int | None = Query(default=None, ge=1),
):
    """
    GET-based search endpoint for simple integrations and browser testing.

    Example: ``/search?q=machine+learning&top_k=3&mmr_lambda=0.5``
    """
    return await search(
        SearchRequest(
            query=q,
            top_k=top_k,
            threshold=threshold,
            mmr_lambda=mmr_lambda,
            mmr_candidate_k=mmr_candidate_k,
        )
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
    )
