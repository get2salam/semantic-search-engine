"""
Tests for retrieval diagnostics: query difficulty, result diversity,
and score-distribution shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from diagnostics import (
    DiversityReport,
    QueryDifficulty,
    ScoreDistribution,
    query_difficulty,
    result_diversity,
    score_distribution,
)


class TestQueryDifficulty:
    def test_clear_winner_high_clarity(self):
        # Sharp drop from top-1 to the rest → high clarity.
        out = query_difficulty([0.95, 0.20, 0.18, 0.15])
        assert isinstance(out, QueryDifficulty)
        assert out.clarity > 0.5
        assert out.score_drop == pytest.approx(0.95 - 0.15, abs=1e-9)

    def test_uniform_scores_low_clarity(self):
        # Flat top-k → no clear winner.
        out = query_difficulty([0.3, 0.3, 0.3, 0.3])
        assert out.clarity == pytest.approx(0.0, abs=1e-9)
        assert out.score_std == pytest.approx(0.0, abs=1e-9)

    def test_threshold_below_ratio(self):
        out = query_difficulty([0.9, 0.6, 0.4, 0.1], threshold=0.5)
        # Two of four scores are below 0.5.
        assert out.below_threshold_ratio == pytest.approx(0.5, abs=1e-9)

    def test_threshold_omitted_returns_none(self):
        out = query_difficulty([0.9, 0.5])
        assert out.below_threshold_ratio is None

    def test_empty_returns_zeros(self):
        out = query_difficulty([])
        assert out.clarity == 0.0
        assert out.score_drop == 0.0
        assert out.score_std == 0.0


class TestResultDiversity:
    def test_identical_results_zero_coverage(self):
        # Stack the same vector — intra-list similarity should be ~1, coverage ~0.
        v = np.array([[1.0, 0.0, 0.0]] * 4)
        out = result_diversity(v)
        assert isinstance(out, DiversityReport)
        assert out.intra_list_similarity == pytest.approx(1.0, abs=1e-6)
        assert out.coverage == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_results_full_coverage(self):
        v = np.eye(3)
        out = result_diversity(v)
        # Orthogonal unit vectors have zero pairwise similarity.
        assert out.intra_list_similarity == pytest.approx(0.0, abs=1e-6)
        assert out.coverage == pytest.approx(1.0, abs=1e-6)

    def test_entropy_matches_uniform_when_no_scores(self):
        v = np.eye(4)
        out = result_diversity(v)
        # No scores → uniform entropy = log(n)
        assert out.entropy == pytest.approx(math.log(4), abs=1e-9)

    def test_entropy_with_concentrated_scores(self):
        v = np.eye(3)
        # All mass on the first result → entropy near 0
        out = result_diversity(v, scores=[10.0, 0.0, 0.0])
        assert out.entropy < 0.5

    def test_empty_input(self):
        out = result_diversity(np.zeros((0, 4)))
        assert out.n_results == 0
        assert out.entropy == 0.0

    def test_single_result(self):
        out = result_diversity(np.array([[1.0, 0.0]]))
        assert out.n_results == 1
        # Single result is trivially "redundant with itself"
        assert out.intra_list_similarity == 1.0
        assert out.coverage == 0.0


class TestScoreDistribution:
    def test_basic_shape(self):
        out = score_distribution([0.9, 0.7, 0.4, 0.1])
        assert isinstance(out, ScoreDistribution)
        assert out.top1 == 0.9
        assert out.topk == 0.1
        assert out.mean == pytest.approx((0.9 + 0.7 + 0.4 + 0.1) / 4)

    def test_zero_variance_zero_skew(self):
        # All-equal scores: skewness should be exactly 0 (we guard against
        # divide-by-zero rather than returning a NaN).
        out = score_distribution([0.5, 0.5, 0.5])
        assert out.skewness == 0.0
        assert out.std == 0.0

    def test_empty(self):
        out = score_distribution([])
        assert out.top1 == 0.0
        assert out.mean == 0.0
        assert out.std == 0.0
