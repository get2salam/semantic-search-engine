"""
Embedding Drift Diagnostics
===========================

Small, dependency-light diagnostics for comparing baseline and current score
or embedding-feature distributions. The main statistic is population stability
index (PSI), a common production monitoring signal that is easy to explain in
model review notes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "DriftBin",
    "DriftReport",
    "population_stability_index",
]


@dataclass(frozen=True)
class DriftBin:
    """Contribution of one interval to a PSI report."""

    lower: float
    upper: float
    baseline_count: int
    current_count: int
    baseline_pct: float
    current_pct: float
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DriftReport:
    """Population stability report between two numeric distributions."""

    psi: float
    n_baseline: int
    n_current: int
    bins: list[DriftBin]

    @property
    def severity(self) -> str:
        if self.psi < 0.1:
            return "low"
        if self.psi < 0.25:
            return "moderate"
        return "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi": self.psi,
            "severity": self.severity,
            "n_baseline": self.n_baseline,
            "n_current": self.n_current,
            "bins": [b.to_dict() for b in self.bins],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Drift report: PSI={self.psi:.4f} ({self.severity})",
            f"- baseline examples: {self.n_baseline}",
            f"- current examples: {self.n_current}",
        ]
        for bucket in self.bins:
            lines.append(
                f"  [{bucket.lower:.4f}, {bucket.upper:.4f}]: "
                f"baseline={bucket.baseline_pct:.3f}, current={bucket.current_pct:.3f}, "
                f"contribution={bucket.contribution:.4f}"
            )
        return lines


def population_stability_index(
    baseline: Sequence[float],
    current: Sequence[float],
    *,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> DriftReport:
    """Compare two numeric distributions using population stability index."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")

    base = np.asarray(list(baseline), dtype=np.float64)
    curr = np.asarray(list(current), dtype=np.float64)
    if base.size == 0 or curr.size == 0:
        raise ValueError("baseline and current must both be non-empty")

    min_value = float(min(base.min(), curr.min()))
    max_value = float(max(base.max(), curr.max()))
    if min_value == max_value:
        max_value = min_value + 1.0

    edges = np.linspace(min_value, max_value, n_bins + 1)
    bins: list[DriftBin] = []
    psi = 0.0

    for idx in range(n_bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == n_bins - 1:
            base_mask = (base >= lower) & (base <= upper)
            curr_mask = (curr >= lower) & (curr <= upper)
        else:
            base_mask = (base >= lower) & (base < upper)
            curr_mask = (curr >= lower) & (curr < upper)

        base_count = int(base_mask.sum())
        curr_count = int(curr_mask.sum())
        base_pct = max(base_count / int(base.size), epsilon)
        curr_pct = max(curr_count / int(curr.size), epsilon)
        contribution = float((curr_pct - base_pct) * np.log(curr_pct / base_pct))
        psi += contribution
        bins.append(
            DriftBin(
                lower=lower,
                upper=upper,
                baseline_count=base_count,
                current_count=curr_count,
                baseline_pct=float(base_pct),
                current_pct=float(curr_pct),
                contribution=contribution,
            )
        )

    return DriftReport(float(psi), int(base.size), int(curr.size), bins)
