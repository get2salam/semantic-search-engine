"""
Tests for the clustering module (clustering.py).

All tests use synthetic numpy arrays so no network calls or GPU are needed.
The SemanticSearchEngine is mocked with a lightweight stub.

Coverage targets:
  - KMeansClusterer: fit, predict, init strategies, edge cases
  - ClusterResult: summary, cluster_sizes auto-fill
  - ClusterAnalyzer: top_terms, centroid_docs
  - ClusterIndex: fit, search variants, persistence, introspection
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from clustering import ClusterAnalyzer, ClusterIndex, ClusterResult, KMeansClusterer

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_blobs(
    n_clusters: int = 3,
    n_per_cluster: int = 20,
    dim: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate well-separated random clusters and return (embeddings, true_labels)."""
    rng = np.random.default_rng(seed)
    emb_parts, y_parts = [], []
    for k in range(n_clusters):
        centre = rng.normal(loc=k * 5, scale=0.5, size=(1, dim))
        pts = centre + rng.normal(scale=0.2, size=(n_per_cluster, dim))
        emb_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, k, dtype=int))
    emb = np.vstack(emb_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    # L2-normalise for cosine metric
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms, y


def _make_engine_stub(docs: list[str], embeddings: np.ndarray) -> Any:
    """Return a minimal mock that looks like SemanticSearchEngine."""
    engine = MagicMock()
    engine.documents = docs
    engine.embeddings = embeddings
    engine.normalize_embeddings = True

    def _encode(query, normalize_embeddings=True, convert_to_numpy=True):  # noqa: ANN001
        rng = np.random.default_rng(abs(hash(query)) % (2**31))
        v = rng.normal(size=embeddings.shape[1]).astype(np.float32)
        if normalize_embeddings:
            v /= np.linalg.norm(v) or 1.0
        return v

    engine.model.encode.side_effect = _encode
    engine.search.side_effect = lambda q, top_k=5: [
        (docs[i], 0.9 - i * 0.05) for i in range(min(top_k, len(docs)))
    ]
    return engine


# ---------------------------------------------------------------------------
# ClusterResult tests
# ---------------------------------------------------------------------------


class TestClusterResult:
    def test_cluster_sizes_auto_filled(self) -> None:
        labels = np.array([0, 0, 1, 1, 1, 2])
        centroids = np.zeros((3, 4))
        cr = ClusterResult(
            labels=labels, centroids=centroids, inertia=1.0, n_iter=5, elapsed_sec=0.1
        )
        assert cr.cluster_sizes == {0: 2, 1: 3, 2: 1}

    def test_cluster_sizes_explicit(self) -> None:
        labels = np.array([0, 1])
        centroids = np.zeros((2, 4))
        cr = ClusterResult(
            labels=labels,
            centroids=centroids,
            inertia=0.5,
            n_iter=3,
            elapsed_sec=0.05,
            cluster_sizes={0: 10, 1: 20},
        )
        assert cr.cluster_sizes == {0: 10, 1: 20}

    def test_summary_returns_string(self) -> None:
        labels = np.array([0, 0, 1])
        centroids = np.zeros((2, 4))
        cr = ClusterResult(
            labels=labels, centroids=centroids, inertia=2.5, n_iter=10, elapsed_sec=0.2
        )
        s = cr.summary()
        assert "Clusters" in s
        assert "Inertia" in s
        assert "2.5" in s


# ---------------------------------------------------------------------------
# KMeansClusterer tests
# ---------------------------------------------------------------------------


class TestKMeansClusterer:
    def test_fit_returns_cluster_result(self) -> None:
        emb, _ = _make_blobs(n_clusters=3, n_per_cluster=15, dim=8)
        km = KMeansClusterer(n_clusters=3, n_init=2, random_state=0)
        result = km.fit(emb)
        assert isinstance(result, ClusterResult)
        assert len(result.labels) == len(emb)
        assert result.centroids.shape == (3, emb.shape[1])

    def test_fit_correct_number_of_labels(self) -> None:
        emb, _ = _make_blobs(n_clusters=4, n_per_cluster=10, dim=8)
        km = KMeansClusterer(n_clusters=4, n_init=1, random_state=1)
        result = km.fit(emb)
        assert set(result.labels.tolist()) == {0, 1, 2, 3}

    def test_fit_inertia_positive(self) -> None:
        emb, _ = _make_blobs(n_clusters=2, n_per_cluster=10, dim=8)
        km = KMeansClusterer(n_clusters=2, n_init=3, random_state=2)
        result = km.fit(emb)
        assert result.inertia >= 0.0

    def test_fit_elapsed_sec_positive(self) -> None:
        emb, _ = _make_blobs(n_clusters=2, n_per_cluster=10, dim=8)
        km = KMeansClusterer(n_clusters=2, n_init=1, random_state=3)
        result = km.fit(emb)
        assert result.elapsed_sec >= 0.0

    def test_fit_n_iter_tracked(self) -> None:
        emb, _ = _make_blobs(n_clusters=2, n_per_cluster=10, dim=8)
        km = KMeansClusterer(n_clusters=2, max_iter=10, n_init=1, random_state=4)
        result = km.fit(emb)
        assert 1 <= result.n_iter <= 10

    def test_fit_euclidean_metric(self) -> None:
        rng = np.random.default_rng(5)
        emb = rng.normal(size=(30, 8)).astype(np.float32)
        km = KMeansClusterer(n_clusters=3, metric="euclidean", n_init=2, random_state=5)
        result = km.fit(emb)
        assert len(result.labels) == 30

    def test_fit_single_cluster(self) -> None:
        emb, _ = _make_blobs(n_clusters=1, n_per_cluster=10, dim=4)
        km = KMeansClusterer(n_clusters=1, n_init=1, random_state=6)
        result = km.fit(emb)
        assert (result.labels == 0).all()

    def test_raises_too_many_clusters(self) -> None:
        emb = np.random.default_rng(7).normal(size=(3, 4)).astype(np.float32)
        km = KMeansClusterer(n_clusters=5, n_init=1)
        with pytest.raises(ValueError, match="Number of documents"):
            km.fit(emb)

    def test_invalid_n_clusters(self) -> None:
        with pytest.raises(ValueError, match="n_clusters must be"):
            KMeansClusterer(n_clusters=0)

    def test_invalid_metric(self) -> None:
        with pytest.raises(ValueError, match="metric must be"):
            KMeansClusterer(metric="manhattan")

    def test_predict_assigns_to_nearest_centroid(self) -> None:
        emb, _ = _make_blobs(n_clusters=3, n_per_cluster=20, dim=8)
        km = KMeansClusterer(n_clusters=3, n_init=3, random_state=8)
        result = km.fit(emb)
        pred = km.predict(emb[:5], result.centroids)
        assert pred.shape == (5,)
        assert set(pred.tolist()).issubset({0, 1, 2})

    def test_multiple_restarts_lower_inertia(self) -> None:
        emb, _ = _make_blobs(n_clusters=3, n_per_cluster=20, dim=8, seed=9)
        km1 = KMeansClusterer(n_clusters=3, n_init=1, random_state=9)
        km5 = KMeansClusterer(n_clusters=3, n_init=5, random_state=9)
        r1 = km1.fit(emb)
        r5 = km5.fit(emb)
        # With more restarts inertia should be <= single run (not always strictly less)
        assert r5.inertia <= r1.inertia + 1e-6

    def test_well_separated_blobs_recovered(self) -> None:
        """With orthogonally-separated blobs, euclidean K-means should recover true labels."""
        rng = np.random.default_rng(42)
        emb_parts, y_parts = [], []
        for k in range(3):
            # Each cluster centred along a distinct axis → no overlap after normalisation
            centre = np.zeros(16, dtype=np.float32)
            centre[k * 5] = 20.0
            pts = centre + rng.normal(scale=0.1, size=(30, 16)).astype(np.float32)
            emb_parts.append(pts)
            y_parts.append(np.full(30, k, dtype=int))
        emb = np.vstack(emb_parts)
        true_labels = np.concatenate(y_parts)

        km = KMeansClusterer(n_clusters=3, metric="euclidean", n_init=5, random_state=42)
        result = km.fit(emb)

        # Each predicted cluster should map almost entirely to one true cluster (purity >= 0.9)
        assert len(set(result.labels.tolist())) == 3
        for pred_cid in range(3):
            idx = np.where(result.labels == pred_cid)[0]
            if len(idx) == 0:
                continue
            _, counts = np.unique(true_labels[idx], return_counts=True)
            purity = counts.max() / len(idx)
            assert purity >= 0.9, f"Cluster {pred_cid} purity {purity:.2f} below threshold"


# ---------------------------------------------------------------------------
# ClusterAnalyzer tests
# ---------------------------------------------------------------------------


class TestClusterAnalyzer:
    def test_top_terms_returns_correct_keys(self) -> None:
        docs = [
            "machine learning neural network deep learning",
            "machine learning gradient descent optimisation",
            "database query sql index performance",
            "database schema normalisation table join",
        ]
        labels = np.array([0, 0, 1, 1])
        analyzer = ClusterAnalyzer()
        terms = analyzer.top_terms(docs, labels, top_n=3)
        assert set(terms.keys()) == {0, 1}
        assert len(terms[0]) <= 3
        assert len(terms[1]) <= 3

    def test_top_terms_no_stop_words(self) -> None:
        docs = ["the and or but is are", "machine learning deep neural"]
        labels = np.array([0, 1])
        analyzer = ClusterAnalyzer()
        terms = analyzer.top_terms(docs, labels, top_n=5)
        stop = {"the", "and", "or", "but", "is", "are"}
        for term_list in terms.values():
            for t in term_list:
                assert t not in stop

    def test_centroid_docs_returns_indices(self) -> None:
        emb, _ = _make_blobs(n_clusters=2, n_per_cluster=10, dim=8, seed=0)
        km = KMeansClusterer(n_clusters=2, n_init=3, random_state=0)
        result = km.fit(emb)
        analyzer = ClusterAnalyzer()
        rep = analyzer.centroid_docs(emb, result.labels, result.centroids, top_n=2)
        assert set(rep.keys()) == {0, 1}
        for idx_list in rep.values():
            assert len(idx_list) <= 2
            assert all(0 <= i < len(emb) for i in idx_list)

    def test_centroid_docs_within_cluster(self) -> None:
        emb, _ = _make_blobs(n_clusters=3, n_per_cluster=10, dim=8, seed=1)
        km = KMeansClusterer(n_clusters=3, n_init=3, random_state=1)
        result = km.fit(emb)
        analyzer = ClusterAnalyzer()
        rep = analyzer.centroid_docs(emb, result.labels, result.centroids, top_n=3)
        for cid, idx_list in rep.items():
            cluster_members = set(np.where(result.labels == cid)[0].tolist())
            for i in idx_list:
                assert i in cluster_members


# ---------------------------------------------------------------------------
# ClusterIndex tests
# ---------------------------------------------------------------------------


class TestClusterIndex:
    def _make_index(
        self, n_clusters: int = 3, n_per: int = 15, dim: int = 16
    ) -> tuple[ClusterIndex, Any]:
        emb, _ = _make_blobs(n_clusters=n_clusters, n_per_cluster=n_per, dim=dim)
        docs = [f"Document {i} about topic {i % n_clusters}" for i in range(len(emb))]
        engine = _make_engine_stub(docs, emb)
        km = KMeansClusterer(n_clusters=n_clusters, n_init=3, random_state=0)
        ci = ClusterIndex(engine, km)
        ci.fit()
        return ci, engine

    def test_fit_populates_result(self) -> None:
        ci, _ = self._make_index()
        assert ci.result is not None

    def test_cluster_ids_correct(self) -> None:
        ci, _ = self._make_index(n_clusters=3)
        assert ci.cluster_ids == [0, 1, 2]

    def test_n_clusters_correct(self) -> None:
        ci, _ = self._make_index(n_clusters=4)
        assert ci.n_clusters == 4

    def test_search_delegates_to_engine(self) -> None:
        ci, engine = self._make_index(n_clusters=3, n_per=10)
        results = ci.search("some query", top_k=3)
        engine.search.assert_called_once_with("some query", top_k=3)
        assert len(results) == 3

    def test_search_in_cluster_returns_results(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=15)
        results = ci.search_in_cluster("machine learning", cluster_id=0, top_k=3)
        assert isinstance(results, list)
        for doc, score in results:
            assert isinstance(doc, str)
            assert isinstance(score, float)

    def test_search_in_cluster_invalid_id(self) -> None:
        ci, _ = self._make_index(n_clusters=3)
        with pytest.raises(KeyError):
            ci.search_in_cluster("query", cluster_id=99)

    def test_search_nearest_cluster_returns_results(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=15)
        results = ci.search_nearest_cluster("deep learning", top_k=5)
        assert len(results) <= 5
        if results:
            scores = [s for _, s in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_nearest_cluster_expand_2(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=15)
        results = ci.search_nearest_cluster("query", top_k=5, expand_to_n_clusters=2)
        assert len(results) <= 5

    def test_cluster_documents_returns_list(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=10)
        for cid in ci.cluster_ids:
            docs = ci.cluster_documents(cid)
            assert isinstance(docs, list)
            assert len(docs) > 0

    def test_document_cluster_valid_label(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=10)
        for i in range(len(ci.engine.documents)):
            cid = ci.document_cluster(i)
            assert cid in ci.cluster_ids

    def test_requires_fit_before_search(self) -> None:
        emb, _ = _make_blobs(n_clusters=2, n_per_cluster=10, dim=8)
        docs = [f"doc {i}" for i in range(len(emb))]
        engine = _make_engine_stub(docs, emb)
        ci = ClusterIndex(engine)
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ci.search_in_cluster("q", cluster_id=0)

    def test_fit_raises_with_no_documents(self) -> None:
        engine = MagicMock()
        engine.documents = []
        ci = ClusterIndex(engine)
        with pytest.raises(RuntimeError, match="no documents"):
            ci.fit()

    def test_repr_unfitted(self) -> None:
        engine = MagicMock()
        engine.documents = []
        ci = ClusterIndex(engine)
        assert "unfitted" in repr(ci)

    def test_repr_fitted(self) -> None:
        ci, _ = self._make_index()
        assert "fitted" in repr(ci)

    def test_save_and_load(self) -> None:
        ci, engine = self._make_index(n_clusters=3, n_per=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            ci.save(tmpdir)
            assert (Path(tmpdir) / "cluster_result.npz").exists()
            assert (Path(tmpdir) / "cluster_meta.json").exists()

            ci2 = ClusterIndex(engine)
            ci2.load(tmpdir)
            assert ci2.result is not None
            np.testing.assert_array_equal(ci2.result.labels, ci.result.labels)
            assert ci2.n_clusters == ci.n_clusters

    def test_save_meta_json_content(self) -> None:
        ci, _ = self._make_index(n_clusters=3, n_per=10)
        with tempfile.TemporaryDirectory() as tmpdir:
            ci.save(tmpdir)
            with open(Path(tmpdir) / "cluster_meta.json") as f:
                meta = json.load(f)
            assert "inertia" in meta
            assert "n_clusters" in meta
            assert "metric" in meta
