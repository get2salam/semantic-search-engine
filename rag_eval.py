"""Small retrieval evaluation metrics for RAG experiments."""

from collections.abc import Iterable, Sequence
from math import log2


def _relevant_set(relevant: Iterable[str]) -> set[str]:
    return {item for item in relevant if item}


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return precision@k for a single query."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    top = list(retrieved[:k])
    if not top:
        return 0.0
    relevant_ids = _relevant_set(relevant)
    return sum(1 for doc_id in top if doc_id in relevant_ids) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return recall@k for a single query."""

    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_ids = _relevant_set(relevant)
    if not relevant_ids:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant_ids) / len(relevant_ids)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return reciprocal rank for the first relevant result."""

    relevant_ids = _relevant_set(relevant)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_ids:
            return 1 / index
    return 0.0


def mean_reciprocal_rank(runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]]) -> float:
    """Return MRR across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(reciprocal_rank(run, rel) for run, rel in zip(runs, qrels, strict=True)) / len(runs)


def average_precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return average precision@k for a single query.

    Averages the precision observed at each rank that contains a relevant document,
    normalised by the number of relevant items reachable within ``k``. Returns
    ``0.0`` when the relevance set is empty so callers can safely average across
    queries without dividing by zero.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_ids = _relevant_set(relevant)
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            score += hits / rank
    return score / min(k, len(relevant_ids))


def mean_average_precision(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return MAP@k across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(
        average_precision_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)
    ) / len(runs)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return nDCG@k using binary relevance gains.

    The discount uses ``log2(rank + 1)`` so the first hit contributes ``1.0`` and
    later hits decay smoothly. Returns ``0.0`` when the relevance set is empty so
    callers can safely average across queries without dividing by zero.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_ids = _relevant_set(relevant)
    if not relevant_ids:
        return 0.0
    dcg = sum(
        1 / log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant_ids
    )
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
