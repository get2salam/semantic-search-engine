"""Small retrieval evaluation metrics for RAG experiments."""

from collections.abc import Iterable, Sequence


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
