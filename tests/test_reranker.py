"""
Tests for the Cross-Encoder Re-Ranker module.

Covers CrossEncoderReranker and TwoStageRetriever using a lightweight
mock model to keep tests fast and dependency-free.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from reranker import (
    CrossEncoderReranker,
    RerankResult,
    RetrievalStats,
    TwoStageRetriever,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _make_mock_cross_encoder(scores: list[float]):
    """Return a mock CrossEncoder whose predict() returns fixed scores."""
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array(scores)
    return mock_ce


def _make_reranker(scores: list[float]) -> CrossEncoderReranker:
    """Build a CrossEncoderReranker with a mocked underlying model."""
    with patch("reranker.CrossEncoderReranker.__init__", return_value=None):
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model_name = "mock-model"
        reranker.max_length = 512
        reranker.model = _make_mock_cross_encoder(scores)
    return reranker


def _make_engine(docs: list[str], scores: list[float]) -> MagicMock:
    """Return a mock SemanticSearchEngine whose search returns (doc, score) pairs."""
    engine = MagicMock()
    engine.__len__ = MagicMock(return_value=len(docs))
    engine.search.return_value = list(zip(docs, scores, strict=False))
    return engine


# ---------------------------------------------------------------------------
# RerankResult
# ---------------------------------------------------------------------------


class TestRerankResult:
    def test_rank_change_positive(self):
        r = RerankResult(document="doc", score=0.9, original_rank=5, reranked_rank=1)
        assert r.rank_change == 4

    def test_rank_change_negative(self):
        r = RerankResult(document="doc", score=0.3, original_rank=1, reranked_rank=3)
        assert r.rank_change == -2

    def test_rank_change_zero(self):
        r = RerankResult(document="doc", score=0.5, original_rank=2, reranked_rank=2)
        assert r.rank_change == 0


# ---------------------------------------------------------------------------
# RetrievalStats
# ---------------------------------------------------------------------------


class TestRetrievalStats:
    def _make_stats(self, recall_ms=10.0, rerank_ms=40.0) -> RetrievalStats:
        return RetrievalStats(
            query="test",
            recall_k=50,
            final_k=5,
            recall_ms=recall_ms,
            rerank_ms=rerank_ms,
            total_ms=recall_ms + rerank_ms,
            candidates_returned=50,
        )

    def test_rerank_overhead_pct(self):
        stats = self._make_stats(recall_ms=10.0, rerank_ms=40.0)
        assert abs(stats.rerank_overhead_pct - 80.0) < 0.1

    def test_rerank_overhead_pct_zero_total(self):
        stats = self._make_stats(recall_ms=0.0, rerank_ms=0.0)
        assert stats.rerank_overhead_pct == 0.0

    def test_str_representation(self):
        stats = self._make_stats()
        s = str(stats)
        assert "recall=" in s
        assert "rerank=" in s
        assert "total=" in s


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_rerank_empty_candidates(self):
        reranker = _make_reranker([])
        results = reranker.rerank("query", [], top_k=5)
        assert results == []

    def test_rerank_returns_sorted_by_score(self):
        docs = ["doc_a", "doc_b", "doc_c"]
        initial_scores = [0.9, 0.8, 0.7]
        # Cross-encoder reverses the ranking: doc_c is best
        ce_scores = [0.2, 0.5, 0.9]

        reranker = _make_reranker(ce_scores)
        candidates = list(zip(docs, initial_scores, strict=False))
        results = reranker.rerank("query", candidates)

        assert len(results) == 3
        assert results[0].document == "doc_c"
        assert results[1].document == "doc_b"
        assert results[2].document == "doc_a"

    def test_rerank_scores_are_cross_encoder_scores(self):
        ce_scores = [0.3, 0.7]

        reranker = _make_reranker(ce_scores)
        candidates = [("doc_x", 0.9), ("doc_y", 0.8)]
        results = reranker.rerank("q", candidates)

        # After sort: doc_y (0.7) first, doc_x (0.3) second
        assert abs(results[0].score - 0.7) < 1e-6
        assert abs(results[1].score - 0.3) < 1e-6

    def test_rerank_top_k_limits_results(self):
        docs = ["a", "b", "c", "d", "e"]
        ce_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        initial = [(d, 0.9) for d in docs]

        reranker = _make_reranker(ce_scores)
        results = reranker.rerank("query", initial, top_k=3)

        assert len(results) == 3

    def test_rerank_assigns_original_ranks(self):
        docs = ["a", "b", "c"]
        ce_scores = [0.9, 0.5, 0.1]  # same order as input
        candidates = [(d, 0.8) for d in docs]

        reranker = _make_reranker(ce_scores)
        results = reranker.rerank("q", candidates)

        # After sort: a=rank1, b=rank2, c=rank3
        # Original ranks: a=1, b=2, c=3 → no change
        assert results[0].original_rank == 1
        assert results[1].original_rank == 2
        assert results[2].original_rank == 3

    def test_rerank_assigns_reranked_ranks(self):
        # Cross-encoder reverses order
        docs = ["a", "b", "c"]
        ce_scores = [0.1, 0.5, 0.9]  # c is best
        candidates = [(d, 0.8) for d in docs]

        reranker = _make_reranker(ce_scores)
        results = reranker.rerank("q", candidates)

        assert results[0].reranked_rank == 1
        assert results[1].reranked_rank == 2
        assert results[2].reranked_rank == 3

    def test_rerank_rank_change_computation(self):
        # doc_c was original rank 3 but becomes rank 1 → change = +2
        docs = ["a", "b", "c"]
        ce_scores = [0.1, 0.5, 0.9]
        candidates = [(d, 0.8) for d in docs]

        reranker = _make_reranker(ce_scores)
        results = reranker.rerank("q", candidates)

        # results[0] is doc_c: original_rank=3, reranked_rank=1, change=+2
        doc_c_result = next(r for r in results if r.document == "c")
        assert doc_c_result.rank_change == 2

    def test_score_pair_calls_predict(self):
        reranker = _make_reranker([0.75])
        score = reranker.score_pair("my query", "some document")
        assert abs(score - 0.75) < 1e-6

    def test_rerank_none_top_k_returns_all(self):
        docs = ["a", "b", "c"]
        ce_scores = [0.3, 0.6, 0.9]
        candidates = [(d, 0.8) for d in docs]

        reranker = _make_reranker(ce_scores)
        results = reranker.rerank("q", candidates, top_k=None)
        assert len(results) == 3

    def test_repr(self):
        reranker = _make_reranker([])
        assert "mock-model" in repr(reranker)
        assert "512" in repr(reranker)


# ---------------------------------------------------------------------------
# TwoStageRetriever
# ---------------------------------------------------------------------------


class TestTwoStageRetriever:
    def _build_retriever(
        self,
        docs: list[str],
        bi_scores: list[float],
        ce_scores: list[float],
        recall_k: int = 50,
    ) -> TwoStageRetriever:
        engine = _make_engine(docs, bi_scores)
        reranker = _make_reranker(ce_scores)
        return TwoStageRetriever(engine, reranker=reranker, default_recall_k=recall_k)

    def test_search_returns_reranked_results(self):
        docs = ["ml doc", "python doc", "java doc"]
        bi_scores = [0.9, 0.8, 0.7]
        # Cross-encoder says python doc is best
        ce_scores = [0.3, 0.9, 0.5]

        retriever = self._build_retriever(docs, bi_scores, ce_scores)
        results = retriever.search("programming", top_k=2)

        assert len(results) == 2
        assert results[0].document == "python doc"

    def test_search_empty_engine(self):
        engine = MagicMock()
        engine.__len__ = MagicMock(return_value=0)
        reranker = _make_reranker([])
        retriever = TwoStageRetriever(engine, reranker=reranker)

        results = retriever.search("query")
        assert results == []
        engine.search.assert_not_called()

    def test_search_populates_last_stats(self):
        docs = ["a", "b"]
        retriever = self._build_retriever(docs, [0.9, 0.8], [0.5, 0.8])
        retriever.search("q", top_k=1)

        assert retriever.last_stats is not None
        assert retriever.last_stats.query == "q"
        assert retriever.last_stats.final_k == 1

    def test_search_stats_disabled(self):
        docs = ["a", "b"]
        engine = _make_engine(docs, [0.9, 0.8])
        reranker = _make_reranker([0.5, 0.8])
        retriever = TwoStageRetriever(engine, reranker=reranker, collect_stats=False)
        retriever.search("q")

        assert retriever.last_stats is None

    def test_search_threshold_filters_results(self):
        docs = ["a", "b", "c"]
        retriever = self._build_retriever(docs, [0.9, 0.8, 0.7], [0.2, 0.8, 0.5])
        results = retriever.search("q", top_k=3, threshold=0.6)

        # Only docs with ce_score >= 0.6 should be returned
        assert all(r.score >= 0.6 for r in results)

    def test_search_batch(self):
        docs = ["a", "b"]
        retriever = self._build_retriever(docs, [0.9, 0.8], [0.5, 0.8])
        all_results = retriever.search_batch(["q1", "q2"], top_k=1)

        assert len(all_results) == 2
        assert all(len(r) == 1 for r in all_results)

    def test_compare_ranking_returns_both_views(self):
        docs = ["ml doc", "python doc", "java doc"]
        bi_scores = [0.9, 0.8, 0.7]
        ce_scores = [0.3, 0.9, 0.5]

        retriever = self._build_retriever(docs, bi_scores, ce_scores)
        comparison = retriever.compare_ranking("query", top_k=2)

        assert "biencoder" in comparison
        assert "crossencoder" in comparison
        assert len(comparison["biencoder"]) == 2
        assert len(comparison["crossencoder"]) == 2

    def test_compare_ranking_empty_engine(self):
        engine = MagicMock()
        engine.__len__ = MagicMock(return_value=0)
        reranker = _make_reranker([])
        retriever = TwoStageRetriever(engine, reranker=reranker)

        comparison = retriever.compare_ranking("query")
        assert comparison == {"biencoder": [], "crossencoder": []}

    def test_recall_k_capped_at_index_size(self):
        # Engine has 3 docs; recall_k defaults would be capped
        docs = ["a", "b", "c"]
        engine = _make_engine(docs, [0.9, 0.8, 0.7])
        reranker = _make_reranker([0.3, 0.6, 0.9])
        retriever = TwoStageRetriever(engine, reranker=reranker, default_recall_k=100)
        retriever.search("q", top_k=2)

        # engine.search should be called with at most len(docs) = 3
        called_top_k = engine.search.call_args[1].get("top_k") or engine.search.call_args[0][1]
        assert called_top_k <= 3

    def test_repr(self):
        docs = ["a"]
        retriever = self._build_retriever(docs, [0.9], [0.8])
        r = repr(retriever)
        assert "TwoStageRetriever" in r
        assert "default_recall_k" in r
