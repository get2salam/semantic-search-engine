from __future__ import annotations

import pytest

from eval_stats import bootstrap_ci, paired_bootstrap_test, sign_test


def test_bootstrap_ci_is_reproducible_and_contains_mean():
    ci = bootstrap_ci([0.2, 0.4, 0.6, 0.8], n_resamples=200, seed=7)

    assert ci.mean == pytest.approx(0.5)
    assert ci.lower <= ci.mean <= ci.upper
    assert ci.confidence == 0.95
    assert ci.n_resamples == 200


def test_bootstrap_ci_validates_arguments():
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_ci([1.0], confidence=1.2)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_ci([1.0], n_resamples=0)


def test_paired_bootstrap_detects_positive_delta():
    result = paired_bootstrap_test([0.1, 0.2, 0.3], [0.3, 0.4, 0.5], n_resamples=200, seed=3)

    assert result.delta == pytest.approx(0.2)
    assert result.p_value == pytest.approx(0.0)
    assert result.n_queries == 3
    assert result.test == "paired_bootstrap"


def test_paired_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="sample length mismatch"):
        paired_bootstrap_test([0.1], [0.1, 0.2])


def test_sign_test_counts_direction_without_scipy():
    result = sign_test([0.1, 0.4, 0.5, 0.5], [0.2, 0.3, 0.5, 0.8])

    assert result.delta == pytest.approx(0.075)
    assert 0.0 <= result.p_value <= 1.0
    assert result.n_queries == 4
    assert result.test == "sign_test"


def test_sign_test_can_drop_ties():
    split = sign_test([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], tie_handling="split")
    dropped = sign_test([1.0, 1.0, 0.0], [1.0, 0.0, 1.0], tie_handling="drop")

    assert split.n_queries == 3
    assert dropped.n_queries == 3
    assert split.p_value >= dropped.p_value


def test_sign_test_validates_tie_handling():
    with pytest.raises(ValueError, match="tie_handling"):
        sign_test([0.1], [0.2], tie_handling="keep")
