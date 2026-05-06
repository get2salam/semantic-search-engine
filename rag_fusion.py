"""Rank-fusion utilities for hybrid RAG retrieval."""

from collections import defaultdict
from collections.abc import Iterable, Sequence

SearchHit = str | tuple[str, float]


def _hit_id(hit: SearchHit) -> str:
    return hit if isinstance(hit, str) else hit[0]


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
    if weights is not None and len(weights) != len(runs):
        raise ValueError("weights length must match runs length")

    scores: dict[str, float] = defaultdict(float)
    for run_index, run in enumerate(runs):
        weight = 1.0 if weights is None else weights[run_index]
        seen: set[str] = set()
        for rank, hit in enumerate(run, start=1):
            doc_id = _hit_id(hit)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] += weight / (k + rank)

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top_k] if top_k is not None else ranked
