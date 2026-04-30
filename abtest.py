"""
A/B Comparison for Retrieval Systems
====================================
Drives a paired comparison between two retrieval configurations on the same
query set, then summarises the result with bootstrap CIs and a paired
significance test.

The shape of the API matches what retrieval engineers actually need when
tuning the engine:

    > "Does adding the cross-encoder reranker beat dense-only?"
    > "Does MMR with λ=0.5 hurt NDCG enough to be a problem?"

Both systems are passed in as ``Callable[[str, int], list[(doc_id, score)]]``
— anything that maps (query, top_k) → ranked results. The comparison is
metric-agnostic: callers supply per-query metric values for each system, and
this module runs the statistics. Computing the metrics themselves is left
to :mod:`evaluation`.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from eval_stats import (
    BootstrapCI,
    PairedTestResult,
    bootstrap_ci,
    paired_bootstrap_test,
    sign_test,
)

__all__ = [
    "MetricComparison",
    "ABReport",
    "compare_systems",
]


@dataclass(frozen=True)
class MetricComparison:
    """Side-by-side comparison for a single metric (e.g. NDCG@10)."""

    metric: str
    a_ci: BootstrapCI
    b_ci: BootstrapCI
    delta: float  # mean(B) - mean(A)
    relative_delta: float  # delta / mean(A) — None-safe (returns 0 when A is 0)
    paired_test: PairedTestResult
    sign_test: PairedTestResult
    a_wins: int
    b_wins: int
    ties: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": self.a_ci.to_dict(),
            "b": self.b_ci.to_dict(),
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "paired_test": self.paired_test.to_dict(),
            "sign_test": self.sign_test.to_dict(),
            "a_wins": self.a_wins,
            "b_wins": self.b_wins,
            "ties": self.ties,
        }


@dataclass(frozen=True)
class ABReport:
    """Full A/B report across one or more metrics."""

    name_a: str
    name_b: str
    n_queries: int
    metrics: list[MetricComparison] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "n_queries": self.n_queries,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def winner(self, metric: str, alpha: float = 0.05) -> str:
        """Name the winner on ``metric`` if the paired bootstrap p-value is
        below ``alpha``, else ``"tie"``.

        This is a convenience for terse summaries — full reports should
        always show the CI and p-value, not just the winner label.
        """
        for comp in self.metrics:
            if comp.metric == metric:
                if comp.paired_test.p_value < alpha and comp.delta > 0:
                    return self.name_b
                if comp.paired_test.p_value < alpha and comp.delta < 0:
                    return self.name_a
                return "tie"
        raise KeyError(f"No comparison for metric {metric!r}")


def _count_outcomes(a: Sequence[float], b: Sequence[float]) -> tuple[int, int, int]:
    a_wins = b_wins = ties = 0
    for x, y in zip(a, b, strict=True):
        if y > x:
            b_wins += 1
        elif x > y:
            a_wins += 1
        else:
            ties += 1
    return a_wins, b_wins, ties


def compare_systems(
    metrics_a: dict[str, Sequence[float]],
    metrics_b: dict[str, Sequence[float]],
    *,
    name_a: str = "A",
    name_b: str = "B",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> ABReport:
    """Build an A/B comparison report from per-query metric vectors.

    Args:
        metrics_a: Mapping ``metric_name -> per_query_values`` for system A.
        metrics_b: Same shape for system B. Must contain the same metric
            keys; per-query vectors must align row-wise (same query order).
        name_a: Display name for system A.
        name_b: Display name for system B.
        confidence: CI coverage probability.
        n_resamples: Bootstrap resamples (passed through to both CI and
            paired-bootstrap test).
        seed: Reproducibility seed shared across both bootstraps.

    Returns:
        :class:`ABReport`.
    """
    if metrics_a.keys() != metrics_b.keys():
        missing = (set(metrics_a) ^ set(metrics_b))
        raise ValueError(f"metric keys differ: {sorted(missing)}")

    n_queries: int | None = None
    comparisons: list[MetricComparison] = []
    for metric in metrics_a:
        a = list(metrics_a[metric])
        b = list(metrics_b[metric])
        if len(a) != len(b):
            raise ValueError(
                f"metric {metric!r}: length mismatch {len(a)} vs {len(b)}"
            )
        if n_queries is None:
            n_queries = len(a)
        elif len(a) != n_queries:
            raise ValueError(f"metric {metric!r} has {len(a)} queries, expected {n_queries}")

        ci_a = bootstrap_ci(a, confidence=confidence, n_resamples=n_resamples, seed=seed)
        ci_b = bootstrap_ci(b, confidence=confidence, n_resamples=n_resamples, seed=seed + 1)
        delta = ci_b.mean - ci_a.mean
        rel = delta / ci_a.mean if ci_a.mean else 0.0

        paired = paired_bootstrap_test(a, b, n_resamples=n_resamples, seed=seed + 2)
        signed = sign_test(a, b)
        a_wins, b_wins, ties = _count_outcomes(a, b)

        comparisons.append(
            MetricComparison(
                metric=metric,
                a_ci=ci_a,
                b_ci=ci_b,
                delta=delta,
                relative_delta=rel,
                paired_test=paired,
                sign_test=signed,
                a_wins=a_wins,
                b_wins=b_wins,
                ties=ties,
            )
        )

    return ABReport(
        name_a=name_a,
        name_b=name_b,
        n_queries=n_queries or 0,
        metrics=comparisons,
    )
