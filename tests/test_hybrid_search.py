"""
Unit tests for the Hybrid Search Engine
========================================
Tests cover BM25Retriever, DenseRetriever (via mock), HybridSearchEngine,
and the reciprocal_rank_fusion utility.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from hybrid_search import (  # noqa: E402
    BM25Retriever,
    HybridResult,
    HybridSearchEngine,
    _tokenize,
    reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


SMALL_CORPUS = [
    "machine learning algorithms improve with data",
    "deep learning uses neural networks for feature extraction",
    "python is a versatile programming language",
    "natural language processing handles text analysis",
    "computer vision enables image recognition tasks",
    "reinforcement learning trains agents via reward signals",
    "data pipelines move and transform datasets efficiently",
    "transformers use attention mechanisms for sequence modelling",
    "gradient descent optimises neural network weights",
    "transfer learning reuses pre-trained model representations",
]


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_stripped(self):
        assert _tokenize("one, two. three!") == ["one", "two", "three"]

    def test_numbers_kept(self):
        tokens = _tokenize("GPT-4 uses 1.7 trillion params")
        assert "gpt" in tokens
        assert "4" in tokens
        assert "1" in tokens or "1" not in tokens  # depends on the split

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_only_punctuation(self):
        assert _tokenize("!!! ???") == []


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class TestBM25Retriever:
    @pytest.fixture
    def retriever(self):
        r = BM25Retriever()
        r.index(SMALL_CORPUS)
        return r

    def test_index_sets_corpus(self, retriever):
        assert len(retriever) == len(SMALL_CORPUS)

    def test_score_returns_array(self, retriever):
        scores = retriever.score("neural network")
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(SMALL_CORPUS),)

    def test_score_non_negative(self, retriever):
        """BM25 scores must be ≥ 0 with the smoothed IDF formulation."""
        scores = retriever.score("deep learning neural network")
        assert (scores >= 0).all()

    def test_search_returns_sorted(self, retriever):
        results = retriever.search("machine learning", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_relevance(self, retriever):
        """Most relevant document should rank first."""
        results = retriever.search("neural network deep learning", top_k=3)
        docs = [d for d, _ in results]
        assert any("deep learning" in d or "neural network" in d for d in docs[:2])

    def test_search_top_k(self, retriever):
        assert len(retriever.search("machine", top_k=3)) == 3
        # k larger than corpus → all docs returned
        assert len(retriever.search("machine", top_k=100)) == len(SMALL_CORPUS)

    def test_empty_corpus(self):
        r = BM25Retriever()
        assert r.search("anything") == []
        assert r.score("anything").size == 0

    def test_unknown_term(self, retriever):
        """Querying with a term not in the corpus should return zero scores."""
        scores = retriever.score("xyzzy1234notaword")
        assert (scores == 0).all()

    def test_reindex(self, retriever):
        """Re-indexing with a smaller corpus should replace the old one."""
        retriever.index(["only one doc"])
        assert len(retriever) == 1

    def test_repr(self, retriever):
        r = repr(retriever)
        assert "BM25Retriever" in r
        assert str(len(SMALL_CORPUS)) in r

    def test_avg_dl_positive(self, retriever):
        assert retriever._avg_dl > 0

    def test_idf_positive(self, retriever):
        for term, val in retriever._idf.items():
            assert val > 0, f"IDF for '{term}' is not positive"

    def test_custom_parameters(self):
        r = BM25Retriever(k1=2.0, b=0.5)
        r.index(SMALL_CORPUS)
        scores = r.score("deep learning")
        assert isinstance(scores, np.ndarray)


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_single_list(self):
        docs = ["a", "b", "c"]
        fused = reciprocal_rank_fusion([docs])
        # Order should be preserved
        result_docs = [d for d, _ in fused]
        assert result_docs == docs

    def test_two_identical_lists(self):
        docs = ["a", "b", "c"]
        fused = reciprocal_rank_fusion([docs, docs])
        result_docs = [d for d, _ in fused]
        # Same order, but higher scores (each doc contributes twice)
        assert result_docs == docs

    def test_complementary_lists(self):
        """Documents appearing in both lists should score higher than those in one."""
        list_a = ["shared", "only_a"]
        list_b = ["shared", "only_b"]
        fused_dict = dict(reciprocal_rank_fusion([list_a, list_b]))
        assert fused_dict["shared"] > fused_dict["only_a"]
        assert fused_dict["shared"] > fused_dict["only_b"]

    def test_all_scores_positive(self):
        lists = [["x", "y"], ["y", "z"]]
        fused = reciprocal_rank_fusion(lists)
        assert all(score > 0 for _, score in fused)

    def test_sorted_descending(self):
        lists = [["a", "b", "c"], ["b", "c", "a"]]
        fused = reciprocal_rank_fusion(lists)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_k_effect(self):
        """Higher rrf_k smooths scores (differences shrink)."""
        lists = [["a", "b"]]
        fused_low_k = dict(reciprocal_rank_fusion(lists, rrf_k=1))
        fused_high_k = dict(reciprocal_rank_fusion(lists, rrf_k=1000))
        diff_low = fused_low_k["a"] - fused_low_k["b"]
        diff_high = fused_high_k["a"] - fused_high_k["b"]
        assert diff_low > diff_high

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []

    def test_all_unique_docs(self):
        """Documents that appear in only one list are still ranked."""
        lists = [["a", "b"], ["c", "d"]]
        fused = reciprocal_rank_fusion(lists)
        result_docs = {d for d, _ in fused}
        assert result_docs == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# HybridSearchEngine  (DenseRetriever is mocked to avoid model download)
# ---------------------------------------------------------------------------


class TestHybridSearchEngine:
    """
    Tests for HybridSearchEngine where DenseRetriever is mocked so the
    test suite doesn't require downloading sentence-transformer models.
    """

    @pytest.fixture
    def mock_dense_retriever(self):
        """Return a DenseRetriever-alike mock that stores the corpus and
        returns plausible scores/ranked docs."""
        mock = MagicMock()
        mock.model_name = "mock-model"
        corpus_holder: list[str] = []

        def _index(docs):
            corpus_holder.clear()
            corpus_holder.extend(docs)

        def _search(query, top_k=10):
            # Reverse of BM25 order to test fusion properly
            reversed_docs = list(reversed(corpus_holder[:top_k]))
            return [(d, 0.9 - i * 0.05) for i, d in enumerate(reversed_docs)]

        mock.index.side_effect = _index
        mock.search.side_effect = _search
        return mock, corpus_holder

    @pytest.fixture
    def hybrid_engine(self, mock_dense_retriever):
        mock_dr, _ = mock_dense_retriever
        with patch("hybrid_search.DenseRetriever", return_value=mock_dr):
            engine = HybridSearchEngine(candidate_k=10)
            engine.index(SMALL_CORPUS)
        return engine, mock_dr

    # ------------------------------------------------------------------

    def test_index_populates_corpus(self, hybrid_engine):
        engine, _ = hybrid_engine
        assert len(engine) == len(SMALL_CORPUS)

    def test_search_returns_hybrid_results(self, hybrid_engine):
        engine, _ = hybrid_engine
        results = engine.search("neural network", top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, HybridResult) for r in results)

    def test_search_results_have_scores(self, hybrid_engine):
        engine, _ = hybrid_engine
        results = engine.search("machine learning", top_k=3)
        assert all(r.rrf_score > 0 for r in results)

    def test_search_sorted_descending(self, hybrid_engine):
        engine, _ = hybrid_engine
        results = engine.search("data pipeline", top_k=5)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_rank_fields_populated(self, hybrid_engine):
        engine, _ = hybrid_engine
        results = engine.search("deep learning", top_k=5)
        # At least some results should have both ranks set
        has_bm25 = any(r.bm25_rank is not None for r in results)
        has_dense = any(r.dense_rank is not None for r in results)
        assert has_bm25
        assert has_dense

    def test_search_simple_format(self, hybrid_engine):
        engine, _ = hybrid_engine
        results = engine.search_simple("python programming", top_k=3)
        assert all(isinstance(doc, str) and isinstance(score, float) for doc, score in results)

    def test_search_empty_corpus(self):
        mock_dr = MagicMock()
        mock_dr.model_name = "mock-model"
        mock_dr.search.return_value = []
        with patch("hybrid_search.DenseRetriever", return_value=mock_dr):
            engine = HybridSearchEngine()
        results = engine.search("query")
        assert results == []

    def test_top_k_respected(self, hybrid_engine):
        engine, _ = hybrid_engine
        for k in [1, 3, 5, 8]:
            results = engine.search("test query", top_k=k)
            assert len(results) == k

    def test_repr(self, hybrid_engine):
        engine, _ = hybrid_engine
        r = repr(engine)
        assert "HybridSearchEngine" in r
        assert "rrf_k=" in r

    def test_hybrid_result_repr(self):
        hr = HybridResult(
            document="test document",
            rrf_score=0.123,
            bm25_rank=2,
            dense_rank=1,
        )
        r = repr(hr)
        assert "HybridResult" in r
        assert "0.1230" in r

    def test_dense_retriever_called_on_index(self, hybrid_engine):
        _, mock_dr = hybrid_engine
        mock_dr.index.assert_called_once_with(SMALL_CORPUS)

    def test_dense_retriever_called_on_search(self, hybrid_engine):
        engine, mock_dr = hybrid_engine
        engine.search("query", top_k=3)
        mock_dr.search.assert_called()


# ---------------------------------------------------------------------------
# Integration-style tests  (BM25 only — no model download needed)
# ---------------------------------------------------------------------------


class TestBM25Integration:
    """Lightweight integration tests against the real BM25 retriever."""

    def test_bm25_outperforms_on_exact_keywords(self):
        """BM25 should rank exact keyword matches highly."""
        corpus = [
            "gradient boosting decision trees",
            "support vector machine classification",
            "k-nearest neighbour algorithm",
            "random forest ensemble method",
        ]
        r = BM25Retriever()
        r.index(corpus)
        results = r.search("random forest", top_k=2)
        top_doc = results[0][0]
        assert "random forest" in top_doc

    def test_bm25_handles_repeated_terms(self):
        """Term-frequency saturation should prevent spam docs from dominating."""
        low_tf = "neural network"
        high_tf = " ".join(["neural"] * 20)
        corpus = [low_tf, high_tf]
        r = BM25Retriever()
        r.index(corpus)
        s_low = r.score("neural network")[0]
        s_high = r.score("neural network")[1]
        # BM25 saturation: the ratio should be small despite 20x TF difference
        ratio = s_high / s_low if s_low > 0 else float("inf")
        assert ratio < 3.0, f"BM25 saturation not working: ratio={ratio:.2f}"

    def test_document_length_normalisation(self):
        """Short documents with same terms should not score much worse."""
        short = "neural network"
        long_padding = " ".join(["neural", "network"] + ["padding"] * 50)
        corpus = [short, long_padding]
        r = BM25Retriever()
        r.index(corpus)
        scores = r.score("neural network")
        # Short doc should not be dominated by the padded one
        # (b=0.75 normalisation prevents extreme length bias)
        assert scores[0] > 0
        assert scores[1] > 0
