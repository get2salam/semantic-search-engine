"""
Document Clustering Module
===========================
Cluster documents in a semantic index using K-means implemented in pure NumPy —
no scikit-learn required.

Enables corpus exploration, topic discovery, and cluster-aware retrieval where
a query is first matched to the nearest cluster and then searched within it.

Classes:
    KMeansClusterer       -- partition documents into K clusters (Lloyd's algorithm)
    ClusterResult         -- data-class holding assignment metadata and statistics
    ClusterIndex          -- integrates clustering with SemanticSearchEngine
    ClusterAnalyzer       -- extract representative terms per cluster

Usage::

    from semantic_search import SemanticSearchEngine
    from clustering import KMeansClusterer, ClusterIndex

    engine = SemanticSearchEngine()
    engine.add_documents(["Machine learning ...", "Deep learning ...", ...])

    clusterer = KMeansClusterer(n_clusters=5, random_state=42)
    cluster_index = ClusterIndex(engine, clusterer)
    cluster_index.fit()

    results = cluster_index.search("neural network optimisation", top_k=5)

Author: get2salam
License: MIT
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from semantic_search import SemanticSearchEngine

__all__ = [
    "KMeansClusterer",
    "ClusterResult",
    "ClusterIndex",
    "ClusterAnalyzer",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """
    Holds the output of a clustering run.

    Attributes:
        labels: Array of cluster ids, one per document (shape ``[N]``).
        centroids: Centroid matrix (shape ``[K, D]``).
        inertia: Sum of squared distances from each point to its centroid.
        n_iter: Number of iterations until convergence.
        elapsed_sec: Wall-clock time for the clustering run.
        cluster_sizes: Dict mapping cluster id to document count.
    """

    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iter: int
    elapsed_sec: float
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cluster_sizes:
            unique, counts = np.unique(self.labels, return_counts=True)
            self.cluster_sizes = dict(zip(unique.tolist(), counts.tolist(), strict=False))

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Clusters  : {len(self.cluster_sizes)}",
            f"Documents : {len(self.labels)}",
            f"Inertia   : {self.inertia:.4f}",
            f"Iterations: {self.n_iter}",
            f"Time (s)  : {self.elapsed_sec:.3f}",
            "Sizes     : " + ", ".join(f"C{k}={v}" for k, v in sorted(self.cluster_sizes.items())),
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# K-Means Clusterer
# ---------------------------------------------------------------------------


class KMeansClusterer:
    """
    Lloyd's K-means algorithm implemented in NumPy.

    The implementation supports:
    * **K-means++ initialisation** for better centroid seeding.
    * **Cosine distance** (for L2-normalised embeddings, cosine similarity
      maximisation is equivalent to Euclidean minimisation).
    * **Euclidean distance** as an alternative metric.
    * Configurable convergence tolerance and iteration cap.
    * **Multiple restarts** (``n_init``) — keeps the run with lowest inertia.

    Parameters:
        n_clusters: Number of clusters ``K``. Must be >= 1.
        max_iter: Maximum iterations per run. Default 300.
        tol: Centroid shift threshold for convergence (L2 norm). Default 1e-4.
        n_init: Number of independent restarts. Default 5.
        metric: ``"cosine"`` (default) or ``"euclidean"``.
        random_state: Seed for reproducibility. Default ``None``.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 5,
        metric: str = "cosine",
        random_state: int | None = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.metric = metric
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> ClusterResult:
        """
        Cluster *embeddings* and return a :class:`ClusterResult`.

        Parameters:
            embeddings: Float array of shape ``[N, D]``.

        Returns:
            Best :class:`ClusterResult` across ``n_init`` restarts.

        Raises:
            ValueError: If there are fewer documents than clusters.
        """
        n_docs = len(embeddings)
        if n_docs < self.n_clusters:
            raise ValueError(
                f"Number of documents ({n_docs}) must be >= n_clusters ({self.n_clusters})"
            )

        emb = embeddings.astype(np.float64)
        if self.metric == "cosine":
            emb = self._l2_normalise(emb)

        t0 = time.perf_counter()
        best: ClusterResult | None = None

        for _ in range(self.n_init):
            result = self._single_run(emb)
            if best is None or result.inertia < best.inertia:
                best = result

        elapsed = time.perf_counter() - t0
        assert best is not None  # n_init >= 1 guaranteed
        best.elapsed_sec = elapsed
        return best

    def predict(self, embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign *embeddings* to the nearest centroid in *centroids*.

        Parameters:
            embeddings: Float array ``[M, D]``.
            centroids: Float array ``[K, D]`` from a previous :meth:`fit` call.

        Returns:
            Integer label array ``[M]``.
        """
        emb = embeddings.astype(np.float64)
        cen = centroids.astype(np.float64)
        if self.metric == "cosine":
            emb = self._l2_normalise(emb)
            cen = self._l2_normalise(cen)
        return self._assign(emb, cen)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_run(self, emb: np.ndarray) -> ClusterResult:
        """One full K-means run from a fresh initialisation."""
        centroids = self._init_kmeanspp(emb)
        labels = self._assign(emb, centroids)

        n_iter = self.max_iter
        for _iter in range(1, self.max_iter + 1):
            new_centroids = self._recompute_centroids(emb, labels)
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            labels = self._assign(emb, centroids)
            if shift < self.tol:
                n_iter = _iter
                break

        inertia = self._compute_inertia(emb, labels, centroids)
        return ClusterResult(
            labels=labels,
            centroids=centroids,
            inertia=inertia,
            n_iter=n_iter,
            elapsed_sec=0.0,
        )

    def _init_kmeanspp(self, emb: np.ndarray) -> np.ndarray:
        """K-means++ centroid seeding."""
        n = len(emb)
        idx = self._rng.integers(0, n)
        centroids = [emb[idx]]

        for _ in range(1, self.n_clusters):
            cen = np.array(centroids)
            # Squared distances to nearest centroid
            dists = self._min_sq_distances(emb, cen)
            probs = dists / dists.sum()
            idx = self._rng.choice(n, p=probs)
            centroids.append(emb[idx])

        return np.array(centroids)

    def _min_sq_distances(self, emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Squared distance from each point to its nearest centroid."""
        # Shape: [N, K]
        diffs = emb[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # [N, K, D]
        sq_dists = (diffs**2).sum(axis=2)  # [N, K]
        return sq_dists.min(axis=1)  # [N]

    def _assign(self, emb: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each point to its nearest centroid."""
        if self.metric == "cosine":
            # cosine similarity = dot product on normalised vectors
            sims = emb @ centroids.T  # [N, K]
            return np.argmax(sims, axis=1)
        else:
            sq_dists = self._pairwise_sq_euclidean(emb, centroids)
            return np.argmin(sq_dists, axis=1)

    @staticmethod
    def _pairwise_sq_euclidean(emb: np.ndarray, cen: np.ndarray) -> np.ndarray:
        """Vectorised pairwise squared Euclidean distance [N, K]."""
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x*c
        x_sq = (emb**2).sum(axis=1, keepdims=True)  # [N, 1]
        c_sq = (cen**2).sum(axis=1, keepdims=True).T  # [1, K]
        cross = emb @ cen.T  # [N, K]
        sq_dists = x_sq + c_sq - 2 * cross
        return np.maximum(sq_dists, 0.0)  # numerical safety

    def _recompute_centroids(self, emb: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recompute centroids as mean of assigned points."""
        n_dim = emb.shape[1]
        centroids = np.zeros((self.n_clusters, n_dim), dtype=np.float64)
        for c in range(self.n_clusters):
            mask = labels == c
            if mask.any():
                centroids[c] = emb[mask].mean(axis=0)
            else:
                # Empty cluster: re-seed with a random point
                centroids[c] = emb[self._rng.integers(0, len(emb))]
        if self.metric == "cosine":
            centroids = self._l2_normalise(centroids)
        return centroids

    @staticmethod
    def _compute_inertia(emb: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        """Sum of squared distances to assigned centroid."""
        diffs = emb - centroids[labels]
        return float((diffs**2).sum())

    @staticmethod
    def _l2_normalise(emb: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation (safe division)."""
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return emb / norms


# ---------------------------------------------------------------------------
# Cluster Analyzer
# ---------------------------------------------------------------------------


class ClusterAnalyzer:
    """
    Extract representative information from each cluster.

    Works on the raw documents and the cluster assignment labels returned
    by :class:`KMeansClusterer`.

    Methods:
        top_terms     -- most frequent non-stop words per cluster (TF-based)
        centroid_docs -- indices of documents closest to each centroid
    """

    _STOP = frozenset(
        {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "it",
            "that",
            "this",
            "as",
            "from",
            "not",
            "can",
            "will",
            "its",
            "into",
            "also",
            "more",
        }
    )

    def top_terms(
        self,
        documents: list[str],
        labels: np.ndarray,
        top_n: int = 5,
    ) -> dict[int, list[str]]:
        """
        Return the ``top_n`` most frequent content words for each cluster.

        Parameters:
            documents: List of raw document strings.
            labels: Cluster label per document (shape ``[N]``).
            top_n: Number of terms to return per cluster. Default 5.

        Returns:
            Mapping ``{cluster_id: [term, ...]}``.
        """
        import re

        cluster_ids = sorted(set(labels.tolist()))
        result: dict[int, list[str]] = {}

        for cid in cluster_ids:
            indices = np.where(labels == cid)[0]
            freq: dict[str, int] = {}
            for idx in indices:
                tokens = re.findall(r"[a-z]{3,}", documents[idx].lower())
                for tok in tokens:
                    if tok not in self._STOP:
                        freq[tok] = freq.get(tok, 0) + 1
            top = sorted(freq, key=lambda t: -freq[t])[:top_n]
            result[cid] = top

        return result

    def centroid_docs(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        top_n: int = 3,
    ) -> dict[int, list[int]]:
        """
        Return indices of the documents closest to each cluster centroid.

        Parameters:
            embeddings: Document embeddings ``[N, D]``.
            labels: Cluster label per document ``[N]``.
            centroids: Centroid matrix ``[K, D]``.
            top_n: Number of representative documents per cluster. Default 3.

        Returns:
            Mapping ``{cluster_id: [doc_index, ...]}``.
        """
        result: dict[int, list[int]] = {}
        cluster_ids = sorted(set(labels.tolist()))

        for cid in cluster_ids:
            idxs = np.where(labels == cid)[0]
            if len(idxs) == 0:
                result[cid] = []
                continue
            cluster_embs = embeddings[idxs]
            centroid = centroids[cid]
            # Cosine similarity to centroid
            norms = np.linalg.norm(cluster_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normed = cluster_embs / norms
            c_norm_val = np.linalg.norm(centroid)
            c_unit = centroid / (c_norm_val if c_norm_val > 0 else 1.0)
            sims = normed @ c_unit
            order = np.argsort(-sims)[: min(top_n, len(idxs))]
            result[cid] = idxs[order].tolist()

        return result


# ---------------------------------------------------------------------------
# Cluster Index — integrates clustering with SemanticSearchEngine
# ---------------------------------------------------------------------------


class ClusterIndex:
    """
    Wraps a :class:`~semantic_search.SemanticSearchEngine` and adds
    cluster-aware retrieval.

    After calling :meth:`fit`, two search strategies are available:

    * **Standard search** (:meth:`search`) — searches the full index (same as
      calling ``engine.search`` directly but accepts a ``ClusterIndex``
      interface).
    * **Cluster-scoped search** (:meth:`search_in_cluster`) — restricts
      candidates to a specific cluster (faster and more focused for large
      corpora).
    * **Auto-cluster search** (:meth:`search_nearest_cluster`) — finds the
      nearest cluster centroid for the query and searches only within it.

    Parameters:
        engine: A fitted :class:`~semantic_search.SemanticSearchEngine`
                with documents already indexed.
        clusterer: A :class:`KMeansClusterer` instance.
    """

    def __init__(
        self,
        engine: SemanticSearchEngine,
        clusterer: KMeansClusterer | None = None,
    ) -> None:
        self.engine = engine
        self.clusterer = clusterer or KMeansClusterer()
        self.result: ClusterResult | None = None
        self._cluster_doc_indices: dict[int, list[int]] = {}

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self) -> ClusterResult:
        """
        Run clustering on the current index embeddings.

        Must be called (or re-called after adding documents) before using
        cluster-aware search methods.

        Returns:
            :class:`ClusterResult` from the clustering run.

        Raises:
            RuntimeError: If the engine has no indexed documents.
        """
        if not self.engine.documents:
            raise RuntimeError("Engine has no documents — add documents before clustering.")
        if self.engine.embeddings is None:
            raise RuntimeError("Engine embeddings are not available.")

        self.result = self.clusterer.fit(self.engine.embeddings)

        # Build per-cluster document index lists
        self._cluster_doc_indices = {}
        for doc_idx, cid in enumerate(self.result.labels.tolist()):
            self._cluster_doc_indices.setdefault(cid, []).append(doc_idx)

        return self.result

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Standard full-index search — thin wrapper around engine.search.

        Parameters:
            query: Search query string.
            top_k: Number of results.

        Returns:
            List of ``(document, score)`` tuples.
        """
        return self.engine.search(query, top_k=top_k)

    def search_in_cluster(
        self, query: str, cluster_id: int, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """
        Search only within documents assigned to *cluster_id*.

        Parameters:
            query: Search query string.
            cluster_id: Target cluster identifier.
            top_k: Maximum number of results.

        Returns:
            List of ``(document, score)`` tuples from the specified cluster.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            KeyError: If *cluster_id* is not a valid cluster.
        """
        self._require_fit()
        if cluster_id not in self._cluster_doc_indices:
            raise KeyError(f"Cluster {cluster_id} not found. Valid ids: {self.cluster_ids}")

        doc_indices = self._cluster_doc_indices[cluster_id]
        cluster_docs = [self.engine.documents[i] for i in doc_indices]

        if not cluster_docs:
            return []

        # Encode query
        query_emb = self.engine.model.encode(
            query,
            normalize_embeddings=self.engine.normalize_embeddings,
            convert_to_numpy=True,
        ).reshape(1, -1)

        # Retrieve embeddings for cluster members
        cluster_embs = self.engine.embeddings[doc_indices]

        # Cosine similarity (embeddings are already normalised)
        scores = (cluster_embs @ query_emb.T).flatten()
        order = np.argsort(-scores)[: min(top_k, len(scores))]

        return [(cluster_docs[i], float(scores[i])) for i in order]

    def search_nearest_cluster(
        self,
        query: str,
        top_k: int = 5,
        expand_to_n_clusters: int = 1,
    ) -> list[tuple[str, float]]:
        """
        Route the query to the nearest cluster(s) and search within them.

        This is useful for large corpora where full-index search is expensive.
        The query is embedded, matched to the nearest ``expand_to_n_clusters``
        centroids, and results from those clusters are merged and re-ranked.

        Parameters:
            query: Search query string.
            top_k: Number of results to return.
            expand_to_n_clusters: How many nearest clusters to search.
                                  ``1`` is the fastest; higher values increase
                                  recall at the cost of latency. Default ``1``.

        Returns:
            Merged list of ``(document, score)`` tuples, sorted by score.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._require_fit()

        query_emb = self.engine.model.encode(
            query,
            normalize_embeddings=self.engine.normalize_embeddings,
            convert_to_numpy=True,
        ).reshape(1, -1)

        centroids = self.result.centroids  # type: ignore[union-attr]
        # Cosine similarity between query and each centroid
        c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        c_norms = np.where(c_norms == 0, 1.0, c_norms)
        normed_centroids = centroids / c_norms
        sims = (normed_centroids @ query_emb.T).flatten()
        nearest = np.argsort(-sims)[:expand_to_n_clusters].tolist()

        # Gather results from selected clusters
        seen: set[str] = set()
        merged: list[tuple[str, float]] = []

        for cid in nearest:
            cluster_results = self.search_in_cluster(query, cid, top_k=top_k)
            for doc, score in cluster_results:
                if doc not in seen:
                    seen.add(doc)
                    merged.append((doc, score))

        merged.sort(key=lambda x: -x[1])
        return merged[:top_k]

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def cluster_ids(self) -> list[int]:
        """Sorted list of cluster identifiers."""
        self._require_fit()
        return sorted(self._cluster_doc_indices.keys())

    @property
    def n_clusters(self) -> int:
        """Number of clusters fitted."""
        self._require_fit()
        return len(self._cluster_doc_indices)

    def cluster_documents(self, cluster_id: int) -> list[str]:
        """Return all documents in *cluster_id*."""
        self._require_fit()
        return [self.engine.documents[i] for i in self._cluster_doc_indices.get(cluster_id, [])]

    def document_cluster(self, doc_index: int) -> int:
        """Return the cluster id for document at *doc_index*."""
        self._require_fit()
        return int(self.result.labels[doc_index])  # type: ignore[index]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save clustering state to *path* directory.

        Saves ``cluster_result.npz`` (labels, centroids) and
        ``cluster_meta.json`` (cluster sizes, n_iter, inertia).
        """
        self._require_fit()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.savez(
            path / "cluster_result.npz",
            labels=self.result.labels,  # type: ignore[union-attr]
            centroids=self.result.centroids,  # type: ignore[union-attr]
        )

        meta = {
            "inertia": self.result.inertia,  # type: ignore[union-attr]
            "n_iter": self.result.n_iter,  # type: ignore[union-attr]
            "elapsed_sec": self.result.elapsed_sec,  # type: ignore[union-attr]
            "cluster_sizes": {
                str(k): v
                for k, v in self.result.cluster_sizes.items()  # type: ignore[union-attr]
            },
            "n_clusters": self.clusterer.n_clusters,
            "metric": self.clusterer.metric,
        }
        with open(path / "cluster_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str | Path) -> None:
        """
        Load clustering state from *path* directory (saved via :meth:`save`).

        The engine must already have its documents and embeddings loaded.
        """
        path = Path(path)
        data = np.load(path / "cluster_result.npz")

        with open(path / "cluster_meta.json") as f:
            meta = json.load(f)

        self.result = ClusterResult(
            labels=data["labels"],
            centroids=data["centroids"],
            inertia=meta["inertia"],
            n_iter=meta["n_iter"],
            elapsed_sec=meta["elapsed_sec"],
            cluster_sizes={int(k): v for k, v in meta["cluster_sizes"].items()},
        )

        self._cluster_doc_indices = {}
        for doc_idx, cid in enumerate(self.result.labels.tolist()):
            self._cluster_doc_indices.setdefault(cid, []).append(doc_idx)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_fit(self) -> None:
        if self.result is None:
            raise RuntimeError("ClusterIndex.fit() must be called before using this method.")

    def __repr__(self) -> str:
        state = "fitted" if self.result is not None else "unfitted"
        return (
            f"ClusterIndex("
            f"engine={self.engine!r}, "
            f"n_clusters={self.clusterer.n_clusters}, "
            f"state={state!r})"
        )
