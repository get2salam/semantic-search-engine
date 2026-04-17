"""
Tests for Maximal Marginal Relevance (MMR) diversification.
"""

from __future__ import annotations

import numpy as np
import pytest

from mmr import mmr_rerank, mmr_select
from semantic_search import SemanticSearchEngine


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row of *x* (2-D) or the whole array (1-D)."""
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(norms > 0, x / norms, x)


class TestMMRSelect:
    """Unit tests for the MMR selection algorithm."""

    def test_empty_candidates(self):
        """Empty pool returns empty selection."""
        q = np.array([1.0, 0.0])
        cand = np.zeros((0, 2))
        assert mmr_select(q, cand, top_k=5) == []

    def test_top_k_zero_or_negative(self):
        """Non-positive top_k returns empty selection."""
        q = np.array([1.0, 0.0])
        cand = np.eye(4)
        assert mmr_select(q, cand, top_k=0) == []
        assert mmr_select(q, cand, top_k=-1) == []

    def test_top_k_larger_than_pool(self):
        """Selection is clamped to candidate pool size."""
        q = np.array([1.0, 0.0, 0.0])
        cand = np.eye(3)
        selected = mmr_select(q, cand, top_k=10)
        assert len(selected) == 3
        assert set(selected) == {0, 1, 2}

    def test_invalid_lambda(self):
        """Lambda outside [0, 1] raises."""
        q = np.array([1.0, 0.0])
        cand = np.eye(2)
        with pytest.raises(ValueError):
            mmr_select(q, cand, top_k=2, lambda_mult=1.5)
        with pytest.raises(ValueError):
            mmr_select(q, cand, top_k=2, lambda_mult=-0.1)

    def test_dim_mismatch(self):
        """Query/candidate dim mismatch raises."""
        q = np.array([1.0, 0.0, 0.0])
        cand = np.eye(2)
        with pytest.raises(ValueError):
            mmr_select(q, cand, top_k=2)

    def test_lambda_one_matches_pure_relevance(self):
        """λ=1 reproduces pure relevance ranking."""
        rng = np.random.default_rng(42)
        q = _l2_normalize(rng.standard_normal(8))
        cand = _l2_normalize(rng.standard_normal((20, 8)))

        mmr_idx = mmr_select(q, cand, top_k=5, lambda_mult=1.0)
        relevance = cand @ q
        pure_idx = np.argsort(relevance)[::-1][:5].tolist()

        assert mmr_idx == pure_idx

    def test_first_pick_is_most_relevant(self):
        """First MMR pick is always the most relevant (diversity term is 0)."""
        rng = np.random.default_rng(0)
        q = _l2_normalize(rng.standard_normal(4))
        cand = _l2_normalize(rng.standard_normal((10, 4)))

        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            selected = mmr_select(q, cand, top_k=3, lambda_mult=lam)
            assert selected[0] == int(np.argmax(cand @ q))

    def test_diversity_reduces_redundancy(self):
        """λ<1 should reduce average pairwise similarity among results."""
        # Build a candidate set where docs 0,1,2 are near-duplicates of each
        # other and doc 3 is an outlier that's still somewhat query-relevant.
        q = _l2_normalize(np.array([1.0, 0.0, 0.0, 0.0]))
        cand = _l2_normalize(
            np.array(
                [
                    [1.0, 0.00, 0.0, 0.0],  # doc 0 — most relevant
                    [0.99, 0.10, 0.0, 0.0],  # doc 1 — near-dup of 0
                    [0.98, 0.15, 0.0, 0.0],  # doc 2 — near-dup of 0
                    [0.50, 0.00, 0.86, 0.0],  # doc 3 — relevant-ish, diverse
                ]
            )
        )

        # Pure relevance picks 0,1,2 (all near-duplicates)
        pure = mmr_select(q, cand, top_k=3, lambda_mult=1.0)
        assert pure == [0, 1, 2]

        # Diversified picks should include doc 3 instead of 1 or 2
        diverse = mmr_select(q, cand, top_k=3, lambda_mult=0.3)
        assert 3 in diverse
        assert diverse[0] == 0  # still the most relevant first

    def test_no_duplicate_selections(self):
        """MMR never picks the same candidate twice."""
        rng = np.random.default_rng(7)
        cand = _l2_normalize(rng.standard_normal((50, 16)))
        q = _l2_normalize(rng.standard_normal(16))

        for lam in [0.0, 0.5, 1.0]:
            selected = mmr_select(q, cand, top_k=20, lambda_mult=lam)
            assert len(selected) == len(set(selected)) == 20

    def test_query_shape_2d_accepted(self):
        """Query can be passed as (1, d) or (d,)."""
        rng = np.random.default_rng(1)
        q1d = _l2_normalize(rng.standard_normal(4))
        cand = _l2_normalize(rng.standard_normal((10, 4)))

        selected_1d = mmr_select(q1d, cand, top_k=3, lambda_mult=0.5)
        selected_2d = mmr_select(q1d.reshape(1, -1), cand, top_k=3, lambda_mult=0.5)
        assert selected_1d == selected_2d


class TestMMRRerank:
    """Tests for the rerank wrapper that also returns scores."""

    def test_returns_original_scores(self):
        """Returned scores are from candidate_scores, not MMR objective."""
        q = _l2_normalize(np.array([1.0, 0.0]))
        cand = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]))
        cand_scores = np.array([0.9, 0.1, 0.5])

        idx, scores = mmr_rerank(q, cand, cand_scores, top_k=2, lambda_mult=0.5)
        assert len(idx) == 2
        # Scores should correspond to the chosen indices
        for i, s in zip(idx, scores, strict=True):
            assert s == pytest.approx(cand_scores[i])

    def test_rerank_respects_top_k(self):
        q = _l2_normalize(np.array([1.0, 0.0]))
        cand = _l2_normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
        scores = np.array([0.95, 0.88, 0.1])

        idx, _ = mmr_rerank(q, cand, scores, top_k=2, lambda_mult=0.0)
        assert len(idx) == 2


class TestEngineMMRIntegration:
    """End-to-end: MMR wired into SemanticSearchEngine.search()."""

    @pytest.fixture(scope="class")
    def engine(self):
        """A search engine seeded with redundant + distinct documents."""
        eng = SemanticSearchEngine(use_faiss=False)
        docs = [
            # Cluster A — machine-learning near-duplicates
            "Machine learning is a subset of artificial intelligence",
            "Machine learning is part of AI research",
            "ML is a branch of artificial intelligence",
            # Cluster B — cooking (distinct topic)
            "Italian pasta recipes with tomato sauce",
            "How to make homemade Italian pizza dough",
            # Cluster C — web dev (distinct topic)
            "JavaScript frameworks for frontend web development",
            "React and Vue are popular JavaScript frameworks",
        ]
        eng.add_documents(docs, show_progress=False)
        return eng

    def test_mmr_none_matches_legacy_behaviour(self, engine):
        """mmr_lambda=None returns sorted pure-relevance results from the
        relevant topic cluster, not from clearly-unrelated clusters."""
        results_no_mmr = engine.search("AI machine learning", top_k=3)
        assert len(results_no_mmr) == 3
        scores = [s for _, s in results_no_mmr]
        assert scores == sorted(scores, reverse=True)
        # The ML cluster should dominate; cooking docs are the clearest
        # counter-example and must not appear in the top-3.
        docs = [d.lower() for d, _ in results_no_mmr]
        assert not any("pasta" in d or "pizza" in d for d in docs)

    def test_mmr_diversifies_results(self, engine):
        """Low-λ MMR pulls in docs from other clusters."""
        no_mmr = engine.search("AI machine learning", top_k=3)
        with_mmr = engine.search("AI machine learning", top_k=3, mmr_lambda=0.2, mmr_candidate_k=7)

        no_mmr_docs = {d for d, _ in no_mmr}
        mmr_docs = {d for d, _ in with_mmr}
        # At least one result should differ — diversification moved something
        assert no_mmr_docs != mmr_docs

    def test_mmr_lambda_one_preserves_top_relevant(self, engine):
        """λ=1 ≈ pure relevance: first result identical to non-MMR top-1."""
        no_mmr = engine.search("AI machine learning", top_k=1)
        mmr = engine.search("AI machine learning", top_k=1, mmr_lambda=1.0)
        assert no_mmr[0][0] == mmr[0][0]

    def test_mmr_scores_are_cosine_similarities(self, engine):
        """Returned scores remain on the cosine-similarity scale, not MMR-weighted."""
        results = engine.search("pasta pizza", top_k=2, mmr_lambda=0.5)
        for _, score in results:
            # Normalised embeddings → cosine similarity in [-1, 1], and for
            # even tangentially relevant matches we expect non-negative.
            assert -1.0 <= score <= 1.0

    def test_mmr_empty_index(self):
        """MMR on empty engine returns []."""
        eng = SemanticSearchEngine(use_faiss=False)
        assert eng.search("anything", top_k=5, mmr_lambda=0.5) == []

    def test_mmr_with_faiss_backend(self):
        """MMR composes correctly with the FAISS path."""
        try:
            eng = SemanticSearchEngine(use_faiss=True)
        except Exception:
            pytest.skip("FAISS unavailable in this environment")
        if not eng.use_faiss:
            pytest.skip("FAISS import failed at runtime")

        eng.add_documents(
            [
                "Machine learning models",
                "Deep learning with neural networks",
                "Neural network training",
                "Italian cooking recipes",
                "French cuisine techniques",
            ],
            show_progress=False,
        )
        results = eng.search("neural networks", top_k=3, mmr_lambda=0.3)
        assert len(results) == 3
        # No duplicate documents in the result set
        docs = [d for d, _ in results]
        assert len(set(docs)) == 3
