"""
Score Calibration Diagnostics
=============================

Utilities for turning raw retrieval scores into reliability-style summaries.
The module is deliberately lightweight: pass aligned ``scores`` and binary
``labels`` and it returns bins, expected calibration error, and Brier score.

This is useful when a semantic search system exposes score thresholds to users:
well-calibrated scores make threshold tuning safer and easier to explain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "calibration_report",
]


@dataclass(frozen=True)
class CalibrationBin:
    """Observed relevance rate for a score interval."""

    lower: float
    upper: float
    count: int
    avg_score: float
    positive_rate: float

    @property
    def gap(self) -> float:
        return self.avg_score - self.positive_rate

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gap"] = self.gap
        return payload


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability summary for retrieval scores."""

    n: int
    positive_rate: float
    brier_score: float
    expected_calibration_error: float
    max_calibration_error: float
    bins: list[CalibrationBin]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "positive_rate": self.positive_rate,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "bins": [b.to_dict() for b in self.bins],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Calibration report ({self.n} examples)",
            f"- positive_rate: {self.positive_rate:.4f}",
            f"- brier_score: {self.brier_score:.4f}",
            f"- expected_calibration_error: {self.expected_calibration_error:.4f}",
            f"- max_calibration_error: {self.max_calibration_error:.4f}",
        ]
        for bucket in self.bins:
            lines.append(
                f"  [{bucket.lower:.2f}, {bucket.upper:.2f}]: "
                f"n={bucket.count}, avg_score={bucket.avg_score:.4f}, "
                f"positive_rate={bucket.positive_rate:.4f}"
            )
        return lines


def calibration_report(
    scores: Sequence[float],
    labels: Sequence[int | bool],
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    """Compute calibration metrics for scores in the inclusive ``[0, 1]`` range."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    score_arr = np.asarray(list(scores), dtype=np.float64)
    label_arr = np.asarray(list(labels), dtype=np.float64)
    if score_arr.shape != label_arr.shape:
        raise ValueError(f"scores/labels shape mismatch: {score_arr.shape} vs {label_arr.shape}")
    if score_arr.size == 0:
        return CalibrationReport(0, 0.0, 0.0, 0.0, 0.0, [])
    if np.any((score_arr < 0.0) | (score_arr > 1.0)):
        raise ValueError("scores must be in [0, 1]")
    if np.any((label_arr != 0.0) & (label_arr != 1.0)):
        raise ValueError("labels must be binary (0/1 or bool)")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = int(score_arr.size)
    bins: list[CalibrationBin] = []
    ece = 0.0
    max_error = 0.0

    for idx in range(n_bins):
        lower = float(edges[idx])
        upper = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (score_arr >= lower) & (score_arr <= upper)
        else:
            mask = (score_arr >= lower) & (score_arr < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_score = float(score_arr[mask].mean())
        positive_rate = float(label_arr[mask].mean())
        gap = abs(avg_score - positive_rate)
        ece += (count / total) * gap
        max_error = max(max_error, gap)
        bins.append(CalibrationBin(lower, upper, count, avg_score, positive_rate))

    brier = float(np.mean((score_arr - label_arr) ** 2))
    return CalibrationReport(
        n=total,
        positive_rate=float(label_arr.mean()),
        brier_score=brier,
        expected_calibration_error=float(ece),
        max_calibration_error=float(max_error),
        bins=bins,
    )
