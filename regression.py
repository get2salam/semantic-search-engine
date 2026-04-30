"""
Golden Query Regression Diagnostics
===================================

Helpers for comparing two ranked-result snapshots for the same query set. This
is deliberately independent of the embedding stack so teams can store tiny
JSON fixtures and review ranking movement in pull requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["QueryRegression", "RegressionReport", "compare_ranked_snapshots"]


@dataclass(frozen=True)
class QueryRegression:
    """Ranking movement for one query."""

    query_id: str
    baseline_top: str | None
    current_top: str | None
    overlap_at_k: int
    jaccard_at_k: float
    dropped: list[str]
    added: list[str]

    @property
    def changed_top(self) -> bool:
        return self.baseline_top != self.current_top

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_top"] = self.changed_top
        return payload


@dataclass(frozen=True)
class RegressionReport:
    """Aggregate movement across a golden query suite."""

    k: int
    queries: list[QueryRegression]

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    @property
    def changed_top_count(self) -> int:
        return sum(1 for q in self.queries if q.changed_top)

    @property
    def mean_jaccard_at_k(self) -> float:
        if not self.queries:
            return 0.0
        return sum(q.jaccard_at_k for q in self.queries) / len(self.queries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "n_queries": self.n_queries,
            "changed_top_count": self.changed_top_count,
            "mean_jaccard_at_k": self.mean_jaccard_at_k,
            "queries": [q.to_dict() for q in self.queries],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Regression report ({self.n_queries} queries, top-{self.k})",
            f"- changed top result: {self.changed_top_count}",
            f"- mean jaccard@{self.k}: {self.mean_jaccard_at_k:.4f}",
        ]
        for query in self.queries:
            if query.changed_top or query.dropped or query.added:
                lines.append(
                    f"  {query.query_id}: top {query.baseline_top!r} -> {query.current_top!r}, "
                    f"jaccard={query.jaccard_at_k:.4f}"
                )
        return lines


def compare_ranked_snapshots(
    baseline: dict[str, Sequence[str]],
    current: dict[str, Sequence[str]],
    *,
    k: int = 10,
) -> RegressionReport:
    """Compare ranked doc-id snapshots keyed by query id."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if baseline.keys() != current.keys():
        missing = sorted(set(baseline) ^ set(current))
        raise ValueError(f"query ids differ: {missing}")

    queries: list[QueryRegression] = []
    for query_id in sorted(baseline):
        base_ranked = [str(doc_id) for doc_id in baseline[query_id]][:k]
        curr_ranked = [str(doc_id) for doc_id in current[query_id]][:k]
        base_set = set(base_ranked)
        curr_set = set(curr_ranked)
        union = base_set | curr_set
        jaccard = len(base_set & curr_set) / len(union) if union else 1.0
        queries.append(
            QueryRegression(
                query_id=str(query_id),
                baseline_top=base_ranked[0] if base_ranked else None,
                current_top=curr_ranked[0] if curr_ranked else None,
                overlap_at_k=len(base_set & curr_set),
                jaccard_at_k=float(jaccard),
                dropped=[doc_id for doc_id in base_ranked if doc_id not in curr_set],
                added=[doc_id for doc_id in curr_ranked if doc_id not in base_set],
            )
        )

    return RegressionReport(k=k, queries=queries)
