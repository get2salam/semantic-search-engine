"""
Tests for corpus-level embedding statistics.
"""

from __future__ import annotations

import numpy as np
import pytest

from embedding_stats import EmbeddingStats, embedding_stats


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


class TestEmbeddingStats:
    def test_returns_dataclass(self):
        rng = np.random.default_rng(0)
        emb = _l2(rng.standard_normal((40, 16)).astype(np.float32))
        stats = embedding_stats(emb)
        assert isinstance(stats, EmbeddingStats)
        assert stats.n == 40
        assert stats.dim == 16

    def test_centroid_norm_zero_for_balanced_corpus(self):
        # Two opposite unit vectors → centroid at the origin.
        emb = np.array([[1.0, 0.0], [-1.0, 0.0]])
        stats = embedding_stats(emb)
        assert stats.centroid_norm == pytest.approx(0.0, abs=1e-9)

    def test_centroid_norm_nonzero_for_skewed_corpus(self):
        emb = np.array([[1.0, 0.0]] * 5)
        stats = embedding_stats(emb)
        assert stats.centroid_norm == pytest.approx(1.0, abs=1e-6)

    def test_mean_pairwise_similarity_high_when_clumped(self):
        # All-equal *unit* vectors → pairwise cosine similarity = 1.
        emb = _l2(np.ones((6, 4)))
        stats = embedding_stats(emb)
        assert stats.mean_pairwise_similarity == pytest.approx(1.0, abs=1e-6)
        assert stats.median_pairwise_similarity == pytest.approx(1.0, abs=1e-6)

    def test_mean_pairwise_similarity_zero_for_orthogonal(self):
        # 4 orthogonal vectors → all pairs have cosine 0.
        emb = np.eye(4)
        stats = embedding_stats(emb)
        assert stats.mean_pairwise_similarity == pytest.approx(0.0, abs=1e-9)

    def test_effective_rank_full_for_orthogonal(self):
        # Orthogonal corpus → full effective rank.
        emb = np.eye(4)
        stats = embedding_stats(emb)
        # Effective rank for I_4 is exactly 4.
        assert stats.effective_rank == pytest.approx(4.0, abs=1e-6)

    def test_effective_rank_one_for_rank_one(self):
        # Rank-one corpus (all rows the same direction) → effective rank 1.
        emb = np.ones((10, 4))
        stats = embedding_stats(emb)
        assert stats.effective_rank == pytest.approx(1.0, abs=1e-6)

    def test_sampled_flag_when_large(self):
        # 2500 rows is above the default sampling threshold (2000).
        rng = np.random.default_rng(0)
        emb = _l2(rng.standard_normal((2500, 8)).astype(np.float32))
        stats = embedding_stats(emb, sample_size=200)
        assert stats.sampled is True
        assert stats.n == 2500  # n is the full corpus size, not the sample size

    def test_empty_input(self):
        stats = embedding_stats(np.zeros((0, 8)))
        assert stats.n == 0
        assert stats.dim == 8
        assert stats.centroid_norm == 0.0

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            embedding_stats(np.zeros(8))
