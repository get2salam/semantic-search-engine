"""
Retrieval Diagnostics
=====================
Per-query and per-result-list signals that surface *why* a search felt good
or bad without needing labelled data.

Three families of signals are implemented here:

1. **Query difficulty** — post-retrieval estimators that score how risky a
   query is given its top-k results. These don't predict ground truth but they
   correlate with it well enough to be useful as triage signals (highlight
   queries the system is uncertain about).

2. **Result diversity** — entropy and intra-list similarity for the returned
   set. Useful for tuning MMR or detecting "echo chamber" results from a
   single dominant cluster.

3. **Score-distribution health** — gap between top-1 and top-k, score
   variance, fraction below a relevance threshold. Cheap to compute, very
   informative for live monitoring.

All functions take pre-computed embeddings and scores; nothing here calls the
encoder. That keeps diagnostics composable with any retrieval backend in the
codebase (dense, hybrid, reranked).

Author: get2salam
License: MIT
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "QueryDifficulty",
    "DiversityReport",
    "ScoreDistribution",
    "query_difficulty",
    "result_diversity",
    "score_distribution",
]


@dataclass(frozen=True)
class QueryDifficulty:
    """Post-retrieval difficulty signals for a single query.

    Attributes:
        clarity: A normalised score-spread proxy in ``[0, 1]``. Higher means
            the top result stands clearly above the rest (= "easier" query).
            Computed as ``(top1 - mean(top_k)) / (top1 + 1e-9)`` clipped to
            ``[0, 1]``.
        score_drop: Absolute gap between the top-1 and the top-k boundary.
            Large drops indicate a clear winner; flat distributions suggest
            ambiguity.
        score_std: Standard deviation of the top-k scores. Low std signals
            uniform mediocrity (often a hard / out-of-domain query).
        below_threshold_ratio: Fraction of top-k scores below ``threshold``.
            ``None`` when no threshold was supplied.
    """

    clarity: float
    score_drop: float
    score_std: float
    below_threshold_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiversityReport:
    """Diversity statistics for a returned result set."""

    intra_list_similarity: float  # mean off-diagonal cosine
    coverage: float  # 1 - intra_list_similarity, clipped to [0, 1]
    entropy: float  # Shannon entropy of softmax(scores)
    n_results: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScoreDistribution:
    """Coarse score-distribution shape for live monitoring."""

    top1: float
    topk: float
    mean: float
    std: float
    skewness: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_array(scores: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(scores), dtype=np.float64)
    return arr


def query_difficulty(
    scores: list[float] | np.ndarray,
    threshold: float | None = None,
) -> QueryDifficulty:
    """Estimate query difficulty from a top-k score vector.

    Args:
        scores: Top-k similarity scores, ordered best-first (the engine's
            native output is already in this order).
        threshold: Optional cutoff. When supplied, the report includes the
            fraction of top-k below the threshold — a useful "no good
            results" signal.

    Returns:
        :class:`QueryDifficulty` with clarity, score_drop, score_std and
        optional below_threshold_ratio.
    """
    arr = _safe_array(scores)
    if arr.size == 0:
        return QueryDifficulty(
            clarity=0.0,
            score_drop=0.0,
            score_std=0.0,
            below_threshold_ratio=None,
        )

    top1 = float(arr[0])
    topk = float(arr[-1])
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))

    # Clarity: how much top-1 dominates the average. Normalise by top-1 so the
    # result is roughly on the same scale across queries with very different
    # absolute similarity magnitudes.
    denom = abs(top1) + 1e-9
    clarity = max(0.0, min(1.0, (top1 - mean) / denom))

    below = None
    if threshold is not None:
        below = float(np.mean(arr < threshold))

    return QueryDifficulty(
        clarity=clarity,
        score_drop=top1 - topk,
        score_std=std,
        below_threshold_ratio=below,
    )


def result_diversity(
    embeddings: np.ndarray,
    scores: list[float] | np.ndarray | None = None,
) -> DiversityReport:
    """Diversity statistics over a result-set embedding matrix.

    Args:
        embeddings: ``(n, d)`` matrix of result embeddings, ideally L2-
            normalised (the engine normalises by default).
        scores: Optional similarity scores aligned with ``embeddings``. When
            present, the entropy is computed over a softmax of these scores
            — a measure of how concentrated the relevance mass is.

    Returns:
        :class:`DiversityReport` with intra-list similarity, coverage (its
        complement, clipped), entropy, and the number of results.
    """
    if embeddings.size == 0 or embeddings.shape[0] == 0:
        return DiversityReport(0.0, 1.0, 0.0, 0)

    arr = embeddings.astype(np.float64, copy=False)
    n = arr.shape[0]

    # Intra-list similarity: mean of the off-diagonal entries of the cosine
    # similarity matrix. Assumes inputs are L2-normalised; renormalise just
    # in case to keep this robust to non-normalised callers.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = arr / safe_norms
    sim = normed @ normed.T

    if n > 1:
        # Off-diagonal mean: subtract the diagonal trace then divide by n*(n-1).
        off_diag_sum = float(sim.sum() - np.trace(sim))
        ils = off_diag_sum / (n * (n - 1))
    else:
        ils = 1.0  # one result is maximally redundant with itself

    coverage = max(0.0, min(1.0, 1.0 - ils))

    # Entropy of the relevance distribution (or uniform when scores are absent).
    if scores is not None and len(scores) == n:
        s = _safe_array(scores)
        # Softmax for stability and to make the entropy comparable across queries.
        s = s - s.max()
        exp = np.exp(s)
        p = exp / max(exp.sum(), 1e-12)
        entropy = float(-np.sum(p * np.log(np.clip(p, 1e-12, None))))
    else:
        entropy = math.log(n) if n > 0 else 0.0

    return DiversityReport(
        intra_list_similarity=float(ils),
        coverage=float(coverage),
        entropy=entropy,
        n_results=n,
    )


def score_distribution(scores: list[float] | np.ndarray) -> ScoreDistribution:
    """Coarse shape of a top-k score vector for live monitoring."""
    arr = _safe_array(scores)
    if arr.size == 0:
        return ScoreDistribution(0.0, 0.0, 0.0, 0.0, 0.0)

    mean = float(arr.mean())
    std = float(arr.std(ddof=0))

    # Skewness: third standardised moment. Positive → long tail of low scores
    # (a few good hits then a cliff); negative → uniform-ish (often a hard query).
    if std > 1e-12:
        m3 = float(((arr - mean) ** 3).mean())
        skew = m3 / (std**3)
    else:
        skew = 0.0

    return ScoreDistribution(
        top1=float(arr[0]),
        topk=float(arr[-1]),
        mean=mean,
        std=std,
        skewness=skew,
    )
