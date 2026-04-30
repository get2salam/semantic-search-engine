"""
Embedding-Space Statistics
==========================
Aggregate health checks for an indexed embedding matrix.

These metrics are computed over the *whole* corpus once after indexing — they
characterise the geometry of the embedding space and surface dataset-level
issues that don't show up in per-query diagnostics:

* **Centroid norm** — distance of the corpus centroid from the origin. A
  drifting centroid across snapshots is a fast leading indicator of
  distribution shift.
* **Mean / median pairwise similarity** — how clumped the corpus is. Very
  high values ("isotropy collapse") mean the encoder isn't separating
  documents well; very low values mean the corpus is over-dispersed
  (rare for sentence-transformers but possible for fine-tuned models).
* **Hubness** — the skewness of the k-NN in-degree distribution. High
  hubness means a few documents appear in many neighbour lists, which
  typically degrades retrieval quality (Radovanović et al., JMLR 2010).
* **Effective rank** — the entropy of normalised singular values, a
  continuous proxy for how many directions the corpus actually uses.

Pairwise computations use sampling for large corpora so the report is fast
on a million-row index without blowing up memory.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "EmbeddingStats",
    "embedding_stats",
]


# Pairwise similarity is O(n^2) — sample above this row count by default.
_PAIRWISE_SAMPLE_THRESHOLD = 2000
_PAIRWISE_SAMPLE_SIZE = 1000


@dataclass(frozen=True)
class EmbeddingStats:
    n: int
    dim: int
    centroid_norm: float
    mean_pairwise_similarity: float
    median_pairwise_similarity: float
    hubness_skewness: float
    effective_rank: float
    sampled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pairwise_off_diag(sim: np.ndarray) -> np.ndarray:
    """Return the upper-triangular (excl. diagonal) flat view of a square matrix."""
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return sim[iu]


def _hubness_skewness(embeddings: np.ndarray, k: int) -> float:
    """Skewness of the k-NN in-degree distribution."""
    n = embeddings.shape[0]
    if n <= k + 1:
        return 0.0

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)  # exclude self
    # Top-k neighbours per row → in-degree counts how many times each row
    # appears as someone else's neighbour.
    nn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    in_degree = np.bincount(nn_idx.flatten(), minlength=n).astype(np.float64)

    mean = in_degree.mean()
    std = in_degree.std(ddof=0)
    if std < 1e-12:
        return 0.0
    m3 = float(((in_degree - mean) ** 3).mean())
    return float(m3 / (std**3))


def _effective_rank(embeddings: np.ndarray) -> float:
    """Entropy-based effective rank: exp(H(σ)) where σ are normalised SVs."""
    n_rows, n_cols = embeddings.shape
    if min(n_rows, n_cols) <= 1:
        return float(min(n_rows, n_cols))

    # SVD on a subsample if the matrix is large to keep memory bounded.
    if n_rows > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_rows, size=5000, replace=False)
        mat = embeddings[idx]
    else:
        mat = embeddings

    # `compute_uv=False` returns singular values only.
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    h = float(-np.sum(p * np.log(p)))
    return float(np.exp(h))


def embedding_stats(
    embeddings: np.ndarray,
    *,
    sample_size: int | None = None,
    seed: int = 0,
    knn_k: int = 10,
) -> EmbeddingStats:
    """Compute health-check statistics over an embedding matrix.

    Args:
        embeddings: ``(n, d)`` matrix. L2-normalised for the cleanest
            similarity numbers; non-normalised input is accepted but
            interpreted as raw dot products.
        sample_size: Number of rows to subsample for the pairwise similarity
            computation. ``None`` triggers the default policy (sample when
            ``n > 2000``, otherwise full).
        seed: Sampling seed.
        knn_k: ``k`` for the hubness in-degree calculation.

    Returns:
        :class:`EmbeddingStats` with corpus-level health metrics.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    n, d = embeddings.shape
    arr = embeddings.astype(np.float64, copy=False)

    if n == 0:
        return EmbeddingStats(0, d, 0.0, 0.0, 0.0, 0.0, 0.0, False)

    centroid = arr.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))

    # Pairwise similarity (sampled for large corpora).
    do_sample = sample_size is not None or n > _PAIRWISE_SAMPLE_THRESHOLD
    if do_sample:
        rng = np.random.default_rng(seed)
        m = sample_size if sample_size is not None else _PAIRWISE_SAMPLE_SIZE
        m = min(m, n)
        idx = rng.choice(n, size=m, replace=False)
        sub = arr[idx]
        sampled = True
    else:
        sub = arr
        sampled = False

    sim = sub @ sub.T
    pairs = _pairwise_off_diag(sim)
    mean_sim = float(pairs.mean()) if pairs.size else 0.0
    median_sim = float(np.median(pairs)) if pairs.size else 0.0

    hubness = _hubness_skewness(sub, k=knn_k)
    eff_rank = _effective_rank(arr)

    return EmbeddingStats(
        n=n,
        dim=d,
        centroid_norm=centroid_norm,
        mean_pairwise_similarity=mean_sim,
        median_pairwise_similarity=median_sim,
        hubness_skewness=hubness,
        effective_rank=eff_rank,
        sampled=sampled,
    )
