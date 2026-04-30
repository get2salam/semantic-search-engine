"""
Statistical Tools for Retrieval Evaluation
==========================================
Bootstrap confidence intervals and paired significance tests for retrieval
metrics. Designed to slot into the existing :mod:`evaluation` pipeline:
``evaluation.py`` already produces per-query metrics; this module turns
those vectors into intervals and p-values.

Why bootstrap rather than a parametric CI? Retrieval metrics aren't normally
distributed (NDCG is bounded in [0,1] and often spikes at 1.0; recall has
heavy mass at 0 and 1). Non-parametric bootstrap makes no distributional
assumptions and matches the way evaluation papers report uncertainty.

Why paired tests rather than independent two-sample? When comparing two
retrieval systems we usually run both on the *same* query set, so the
correct comparison is paired (Smucker et al., CIKM 2007). Independent tests
throw away the within-query variance reduction.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "BootstrapCI",
    "PairedTestResult",
    "bootstrap_ci",
    "paired_bootstrap_test",
    "sign_test",
]


@dataclass(frozen=True)
class BootstrapCI:
    """A bootstrap confidence interval for one metric."""

    mean: float
    lower: float
    upper: float
    confidence: float  # e.g. 0.95
    n_resamples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired comparison between two systems on the same queries."""

    delta: float  # mean(B - A) — positive means B beats A
    p_value: float
    n_queries: int
    test: str  # name of the test ("paired_bootstrap" or "sign_test")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bootstrap_ci(
    samples: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI for the mean of ``samples``.

    Args:
        samples: Per-query metric values (e.g. NDCG@10 for each query).
        confidence: Two-sided coverage probability. Default 0.95 → 2.5/97.5
            percentile interval.
        n_resamples: Number of bootstrap resamples. 2000 is a good speed/
            stability trade-off; bump to 10000 for publication-grade plots.
        seed: RNG seed.

    Returns:
        :class:`BootstrapCI` with mean and lower/upper bounds.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")

    arr = np.asarray(list(samples), dtype=np.float64)
    n = arr.size
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, confidence, n_resamples)

    rng = np.random.default_rng(seed)
    # Vectorised resampling: pick n indices per resample, take row-mean.
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(means, alpha))
    upper = float(np.quantile(means, 1.0 - alpha))

    return BootstrapCI(
        mean=float(arr.mean()),
        lower=lower,
        upper=upper,
        confidence=confidence,
        n_resamples=n_resamples,
    )


def paired_bootstrap_test(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    n_resamples: int = 2000,
    seed: int = 0,
) -> PairedTestResult:
    """Paired bootstrap test for ``mean(B) > mean(A)``.

    For each resample we draw ``n`` paired indices (with replacement) and
    take the mean of the per-query differences. The p-value is the fraction
    of resamples where the difference is non-positive — a standard one-sided
    paired bootstrap.

    Args:
        samples_a: Per-query metric for system A.
        samples_b: Per-query metric for system B (same order as A).

    Returns:
        :class:`PairedTestResult` with the mean delta (B − A) and the
        one-sided p-value.

    Raises:
        ValueError: If the two sample lists have different lengths.
    """
    a = np.asarray(list(samples_a), dtype=np.float64)
    b = np.asarray(list(samples_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"sample length mismatch: {a.shape} vs {b.shape}")

    n = a.size
    if n == 0:
        return PairedTestResult(0.0, 1.0, 0, "paired_bootstrap")

    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = diff[idx].mean(axis=1)
    # One-sided: how often the resampled mean is <= 0.
    p_value = float(np.mean(resampled_means <= 0.0))

    return PairedTestResult(
        delta=float(diff.mean()),
        p_value=p_value,
        n_queries=int(n),
        test="paired_bootstrap",
    )


def sign_test(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    tie_handling: str = "split",
) -> PairedTestResult:
    """Two-sided paired sign test: how often does B beat A per query?

    Distribution-free and trivially interpretable: under the null hypothesis
    the per-query difference has a 50/50 chance of being positive, so the
    number of wins follows a Binomial(n, 0.5). Ties are conventionally
    "split" (counted as half a win for each side) which matches the
    Smucker et al. recommendation for retrieval comparisons.

    Args:
        samples_a: Per-query metric for system A.
        samples_b: Per-query metric for system B (same order).
        tie_handling: ``"split"`` (default) or ``"drop"`` (exclude ties from
            the count, the classical sign test).

    Returns:
        :class:`PairedTestResult` with the mean delta and the two-sided
        p-value.
    """
    a = np.asarray(list(samples_a), dtype=np.float64)
    b = np.asarray(list(samples_b), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"sample length mismatch: {a.shape} vs {b.shape}")

    n = a.size
    if n == 0:
        return PairedTestResult(0.0, 1.0, 0, "sign_test")

    diff = b - a
    wins = float(np.sum(diff > 0))
    losses = float(np.sum(diff < 0))
    ties = float(np.sum(diff == 0))

    if tie_handling == "split":
        wins += ties / 2.0
        n_eff = n
    elif tie_handling == "drop":
        n_eff = int(wins + losses)
    else:
        raise ValueError(f"tie_handling must be 'split' or 'drop', got {tie_handling!r}")

    if n_eff == 0:
        return PairedTestResult(float(diff.mean()), 1.0, n, "sign_test")

    # Two-sided binomial p-value with p=0.5 — uses the symmetry of the
    # symmetric binomial to avoid pulling in scipy. We compute the prob of
    # outcomes at least as extreme as the observed in either tail.
    k = int(round(min(wins, n_eff - wins)))
    p_value = 2.0 * _binom_cdf(k, n_eff, 0.5)
    p_value = min(1.0, p_value)

    return PairedTestResult(
        delta=float(diff.mean()),
        p_value=p_value,
        n_queries=int(n),
        test="sign_test",
    )


def _binom_cdf(k: int, n: int, p: float) -> float:
    """Lower-tail binomial CDF P(X <= k) via direct summation (n small)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    # Use log-space to avoid overflow at moderate n.
    log_p = np.log(p) if p > 0 else -np.inf
    log_q = np.log(1.0 - p) if p < 1.0 else -np.inf
    log_factorial = np.cumsum(np.log(np.arange(1, n + 1)))

    def log_binom(j: int) -> float:
        # log C(n, j)
        if j == 0 or j == n:
            return 0.0
        return float(log_factorial[n - 1] - log_factorial[j - 1] - log_factorial[n - j - 1])

    log_terms = np.array([log_binom(j) + j * log_p + (n - j) * log_q for j in range(k + 1)])
    m = float(log_terms.max())
    return float(np.exp(m) * np.exp(log_terms - m).sum())
