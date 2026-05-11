"""Small retrieval evaluation metrics for RAG experiments."""

from collections.abc import Iterable, Mapping, Sequence
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


def mean_precision_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return mean precision@k across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(
        precision_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)
    ) / len(runs)


def mean_recall_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return mean recall@k across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(
        recall_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)
    ) / len(runs)


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


def hit_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return ``1.0`` when any relevant document appears in the top ``k`` results.

    Useful as a coarse RAG success signal when a single grounded passage is
    enough to answer the query. Returns ``0.0`` for empty relevance sets so
    callers can safely average across queries without dividing by zero.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_ids = _relevant_set(relevant)
    if not relevant_ids:
        return 0.0
    return 1.0 if any(doc_id in relevant_ids for doc_id in retrieved[:k]) else 0.0


def hit_rate_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return the mean hit@k across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(hit_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)) / len(runs)


def f1_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Return the F1 score that balances precision@k and recall@k for one query.

    Materialises ``relevant`` once so both component metrics see the same set,
    and returns ``0.0`` when either component is zero so callers can safely
    average across queries without dividing by zero.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    relevant_ids = _relevant_set(relevant)
    precision = precision_at_k(retrieved, relevant_ids, k)
    recall = recall_at_k(retrieved, relevant_ids, k)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_f1_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return the mean F1@k across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(f1_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)) / len(runs)


def r_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """Return R-precision: precision at rank ``R``, where ``R = |relevant|``.

    R-precision adapts the cutoff to the number of relevant documents for the
    query, which makes it comparable across queries with very different
    relevance set sizes without picking a fixed ``k``. Returns ``0.0`` when the
    relevance set is empty so callers can safely average across queries without
    dividing by zero.
    """

    relevant_ids = _relevant_set(relevant)
    if not relevant_ids:
        return 0.0
    cutoff = len(relevant_ids)
    return sum(1 for doc_id in retrieved[:cutoff] if doc_id in relevant_ids) / cutoff


def mean_r_precision(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]]
) -> float:
    """Return mean R-precision across aligned retrieval runs and relevance sets."""

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(r_precision(run, rel) for run, rel in zip(runs, qrels, strict=True)) / len(runs)


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


def mean_ndcg_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Iterable[str]], k: int
) -> float:
    """Return the mean nDCG@k across aligned retrieval runs and relevance sets.

    Provides the rank-aware aggregate that complements :func:`mean_average_precision`
    and :func:`mean_r_precision`. Each query contributes ``0.0`` when its relevance
    set is empty so callers can safely average heterogeneous evaluation slices
    without dividing by zero.
    """

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(ndcg_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)) / len(runs)


def graded_ndcg_at_k(
    retrieved: Sequence[str], relevance: Mapping[str, float], k: int
) -> float:
    """Return nDCG@k using graded relevance gains supplied per document.

    Generalises :func:`ndcg_at_k` for evaluations that distinguish "highly
    relevant" from "marginally relevant" passages. Documents with non-positive
    gain are treated as irrelevant, and the ideal ranking sorts the supplied
    gains in descending order. Returns ``0.0`` when no positive gain is
    available so callers can safely average across queries without dividing
    by zero.
    """

    if k <= 0:
        raise ValueError("k must be greater than zero")
    grades = {doc_id: float(gain) for doc_id, gain in relevance.items() if gain > 0}
    if not grades:
        return 0.0
    dcg = sum(
        grades.get(doc_id, 0.0) / log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def mean_graded_ndcg_at_k(
    runs: Sequence[Sequence[str]], qrels: Sequence[Mapping[str, float]], k: int
) -> float:
    """Return the mean graded nDCG@k across aligned runs and graded relevance maps.

    Complements :func:`mean_ndcg_at_k` for evaluations where relevance is graded
    rather than binary. Each query contributes ``0.0`` when its grades are all
    non-positive so callers can safely average heterogeneous evaluation slices
    without dividing by zero.
    """

    if len(runs) != len(qrels):
        raise ValueError("runs and qrels must have the same length")
    if not runs:
        return 0.0
    return sum(
        graded_ndcg_at_k(run, rel, k) for run, rel in zip(runs, qrels, strict=True)
    ) / len(runs)
