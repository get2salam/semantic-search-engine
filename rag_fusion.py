"""Rank-fusion utilities for hybrid RAG retrieval."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

SearchHit = str | tuple[str, float]


def _hit_id(hit: SearchHit) -> str:
    return hit if isinstance(hit, str) else hit[0]


def _hit_score(hit: SearchHit, default: float) -> float:
    return default if isinstance(hit, str) else float(hit[1])


def _validate_weights(weights: Sequence[float] | None, n_runs: int) -> None:
    if weights is None:
        return
    if len(weights) != n_runs:
        raise ValueError("weights length must match runs length")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")


def _validate_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative")


def reciprocal_rank_fusion(
    runs: Sequence[Iterable[SearchHit]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked retrieval runs using Reciprocal Rank Fusion.

    RRF is robust when scores come from incomparable systems, e.g. BM25 and
    dense vectors. Only rank positions matter; optional weights can bias toward
    trusted retrievers.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    _validate_weights(weights, len(runs))
    _validate_top_k(top_k)

    scores: dict[str, float] = defaultdict(float)
    for run_index, run in enumerate(runs):
        weight = 1.0 if weights is None else weights[run_index]
        if weight == 0.0:
            continue
        seen: set[str] = set()
        for rank, hit in enumerate(run, start=1):
            doc_id = _hit_id(hit)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += weight / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k] if top_k is not None else ranked


def weighted_score_fusion(
    runs: Sequence[Iterable[SearchHit]],
    *,
    weights: Sequence[float] | None = None,
    top_k: int | None = None,
    normalize: bool = True,
    default_score: float = 0.0,
) -> list[tuple[str, float]]:
    """Fuse ranked runs by summing weighted per-run scores (CombSUM).

    Complements :func:`reciprocal_rank_fusion`: where RRF discards score
    magnitudes and only uses rank positions, this fusion preserves the
    relative strength of each retriever's confidence. Per-run min-max
    normalization (enabled by default) maps each run's scores to ``[0, 1]``
    so heterogeneous retrievers — e.g. BM25 (unbounded) and cosine
    similarity (``[-1, 1]``) — can be combined without one drowning out the
    other.

    String-only hits (no attached score) contribute ``default_score`` for
    that run, which makes the function safe to mix with ID-only ranked
    lists.

    Args:
        runs: Ranked hits per retriever. Each hit may be a doc ID string or
            an ``(id, score)`` tuple.
        weights: Optional non-negative per-run weights. Defaults to uniform.
        top_k: If set, return only the top ``top_k`` fused results.
        normalize: Min-max normalize each run's scores onto ``[0, 1]``
            before weighting. A run whose scores are all equal collapses
            to ``1.0`` for each present hit (treated as a pure inclusion
            signal). Disable when scores are already comparable across
            runs.
        default_score: Score assigned to string-only hits within a run.

    Returns:
        ``(doc_id, fused_score)`` pairs sorted by descending fused score,
        ties broken by doc_id ascending.

    Raises:
        ValueError: If ``weights`` length disagrees with ``runs`` length,
            any weight is negative, or ``top_k`` is negative.
    """

    _validate_weights(weights, len(runs))
    _validate_top_k(top_k)

    fused: dict[str, float] = defaultdict(float)
    for run_index, run in enumerate(runs):
        weight = 1.0 if weights is None else weights[run_index]
        if weight == 0.0:
            continue

        run_scores: dict[str, float] = {}
        for hit in run:
            doc_id = _hit_id(hit)
            if doc_id in run_scores:
                continue
            run_scores[doc_id] = _hit_score(hit, default_score)

        if not run_scores:
            continue

        if normalize:
            values = run_scores.values()
            lo, hi = min(values), max(values)
            span = hi - lo
            if span > 0:
                run_scores = {doc: (score - lo) / span for doc, score in run_scores.items()}
            else:
                run_scores = dict.fromkeys(run_scores, 1.0)

        for doc_id, score in run_scores.items():
            fused[doc_id] += weight * score

    ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k] if top_k is not None else ranked
