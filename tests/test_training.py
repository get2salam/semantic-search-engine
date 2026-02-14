"""
Tests for training and evaluation modules.
============================================
Unit tests for the fine-tuning pipeline, metric functions,
and evaluation harness. Uses synthetic data to avoid model downloads.
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from evaluation import (
    EvalQuery,
    EvalReport,
    ModelBenchmark,
    RetrievalEvaluator,
    average_precision,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from training import (
    FineTuner,
    TrainingConfig,
    TrainingPair,
    TrainingResult,
    TrainingTriplet,
)


# =========================================================================
# Metric Unit Tests
# =========================================================================


class TestReciprocalRank:
    """Tests for reciprocal_rank metric."""

    def test_first_result_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_second_result_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5

    def test_third_result_relevant(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_relevant_results(self):
        assert reciprocal_rank(["a", "b", "c"], {"d"}) == 0.0

    def test_empty_results(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(["a", "b"], set()) == 0.0

    def test_multiple_relevant_returns_first(self):
        """RR uses the rank of the FIRST relevant result."""
        assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5


class TestAveragePrecision:
    """Tests for average_precision metric."""

    def test_perfect_ranking(self):
        """All relevant docs at the top."""
        assert average_precision(["a", "b", "c"], {"a", "b"}) == 1.0

    def test_reversed_ranking(self):
        """Relevant docs at positions 2 and 3."""
        ap = average_precision(["x", "a", "b"], {"a", "b"})
        # P@2 * 1 + P@3 * 1 = (1/2 + 2/3) / 2
        expected = (1 / 2 + 2 / 3) / 2
        assert ap == pytest.approx(expected)

    def test_single_relevant(self):
        ap = average_precision(["x", "y", "a"], {"a"})
        assert ap == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert average_precision(["a", "b"], set()) == 0.0

    def test_empty_results(self):
        assert average_precision([], {"a"}) == 0.0

    def test_all_relevant(self):
        """Every result is relevant."""
        ap = average_precision(["a", "b", "c"], {"a", "b", "c"})
        assert ap == pytest.approx(1.0)


class TestPrecisionAtK:
    """Tests for precision_at_k metric."""

    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x", "b", "y"], {"a", "b"}, 4) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["x", "y"], {"a"}, 2) == 0.0

    def test_k_larger_than_results(self):
        """k exceeds number of results — denominator is still k."""
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(1 / 5)

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, 0) == 0.0


class TestRecallAtK:
    """Tests for recall_at_k metric."""

    def test_all_found(self):
        assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0

    def test_partial_found(self):
        assert recall_at_k(["a", "x", "y"], {"a", "b"}, 3) == 0.5

    def test_none_found(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), 1) == 0.0

    def test_k_1(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 1) == 0.5


class TestNDCG:
    """Tests for DCG and NDCG metrics."""

    def test_perfect_binary_ndcg(self):
        """All relevant docs at top = NDCG@k = 1.0."""
        eq = EvalQuery(query="q", relevant_docs=["a", "b"])
        assert ndcg_at_k(["a", "b", "c"], eq, 2) == pytest.approx(1.0)

    def test_reversed_binary_ndcg(self):
        """Relevant docs at bottom — lower NDCG."""
        eq = EvalQuery(query="q", relevant_docs=["a", "b"])
        score = ndcg_at_k(["x", "a", "b"], eq, 3)
        assert 0 < score < 1.0

    def test_graded_relevance(self):
        """Test with graded relevance scores."""
        eq = EvalQuery(
            query="q",
            relevant_docs=["a", "b", "c"],
            relevance_grades={"a": 3, "b": 2, "c": 1},
        )
        # Perfect ranking: a(3), b(2), c(1)
        perfect = ndcg_at_k(["a", "b", "c"], eq, 3)
        assert perfect == pytest.approx(1.0)

        # Reversed: c(1), b(2), a(3)
        reversed_score = ndcg_at_k(["c", "b", "a"], eq, 3)
        assert reversed_score < 1.0

    def test_no_relevant_docs(self):
        eq = EvalQuery(query="q", relevant_docs=[])
        assert ndcg_at_k(["a", "b"], eq, 2) == 0.0

    def test_dcg_computation(self):
        """Verify DCG formula: sum(rel_i / log2(i+1))."""
        eq = EvalQuery(
            query="q",
            relevant_docs=["a", "b"],
            relevance_grades={"a": 3, "b": 1},
        )
        dcg = dcg_at_k(["a", "b"], eq, 2)
        expected = 3 / math.log2(2) + 1 / math.log2(3)
        assert dcg == pytest.approx(expected)


# =========================================================================
# Training Module Tests
# =========================================================================


class TestTrainingConfig:
    """Tests for TrainingConfig defaults and fields."""

    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.base_model == "all-MiniLM-L6-v2"
        assert cfg.epochs == 3
        assert cfg.learning_rate == 2e-5
        assert cfg.loss_type == "cosine"
        assert cfg.cv_folds == 0
        assert cfg.seed == 42

    def test_custom_config(self):
        cfg = TrainingConfig(
            base_model="all-mpnet-base-v2",
            epochs=10,
            batch_size=32,
            loss_type="triplet",
            cv_folds=5,
        )
        assert cfg.base_model == "all-mpnet-base-v2"
        assert cfg.epochs == 10
        assert cfg.cv_folds == 5


class TestTrainingResult:
    """Tests for TrainingResult serialization."""

    def test_to_dict(self):
        result = TrainingResult(
            model_path="/tmp/model",
            epochs_completed=3,
            training_samples=100,
            final_loss=0.15,
            best_score=0.92,
            elapsed_seconds=45.3,
        )
        d = result.to_dict()
        assert d["model_path"] == "/tmp/model"
        assert d["best_score"] == 0.92

    def test_save_and_load(self, tmp_path):
        result = TrainingResult(
            model_path=str(tmp_path / "model"),
            epochs_completed=2,
            training_samples=50,
            final_loss=0.2,
            best_score=0.88,
            elapsed_seconds=30.0,
        )
        out = tmp_path / "result.json"
        result.save(out)

        loaded = json.loads(out.read_text())
        assert loaded["epochs_completed"] == 2
        assert loaded["best_score"] == 0.88

    def test_repr(self):
        result = TrainingResult(
            model_path="x",
            epochs_completed=1,
            training_samples=10,
            final_loss=0.5,
            best_score=0.7,
            elapsed_seconds=5.0,
        )
        assert "0.5000" in repr(result)
        assert "0.7" in repr(result)


class TestFineTuner:
    """Tests for the FineTuner class (data loading, no actual model training)."""

    def test_add_pairs(self):
        tuner = FineTuner()
        tuner.add_pairs([
            TrainingPair("q1", "d1"),
            TrainingPair("q2", "d2"),
        ])
        assert len(tuner._pairs) == 2

    def test_add_triplets(self):
        tuner = FineTuner()
        tuner.add_triplets([
            TrainingTriplet("q1", "d1", "d2"),
        ])
        assert len(tuner._triplets) == 1

    def test_load_pairs_jsonl(self, tmp_path):
        data = tmp_path / "pairs.jsonl"
        data.write_text(
            '{"query": "q1", "positive": "d1"}\n'
            '{"query": "q2", "positive": "d2"}\n'
        )
        tuner = FineTuner()
        loaded = tuner.load_pairs_jsonl(data)
        assert loaded == 2
        assert tuner._pairs[0].query == "q1"

    def test_load_triplets_jsonl(self, tmp_path):
        data = tmp_path / "triplets.jsonl"
        data.write_text(
            '{"query": "q1", "positive": "d1", "negative": "d3"}\n'
        )
        tuner = FineTuner()
        loaded = tuner.load_triplets_jsonl(data)
        assert loaded == 1
        assert tuner._triplets[0].negative == "d3"

    def test_train_no_data_raises(self):
        tuner = FineTuner()
        with pytest.raises(ValueError, match="No training data"):
            tuner.train()

    def test_load_skips_blank_lines(self, tmp_path):
        data = tmp_path / "pairs.jsonl"
        data.write_text(
            '{"query": "q1", "positive": "d1"}\n'
            "\n"
            '{"query": "q2", "positive": "d2"}\n'
            "  \n"
        )
        tuner = FineTuner()
        loaded = tuner.load_pairs_jsonl(data)
        assert loaded == 2


# =========================================================================
# Evaluator Tests
# =========================================================================


class TestRetrievalEvaluator:
    """Tests for the RetrievalEvaluator class."""

    @staticmethod
    def _dummy_search(rankings: dict):
        """Create a mock search function from a predefined ranking dict."""

        def search_fn(query: str, k: int) -> list:
            return rankings.get(query, [])[:k]

        return search_fn

    def test_perfect_retrieval(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([
            EvalQuery("q1", ["d1", "d2"]),
            EvalQuery("q2", ["d3"]),
        ])
        search = self._dummy_search({
            "q1": ["d1", "d2", "d4"],
            "q2": ["d3", "d5"],
        })
        report = evaluator.evaluate(search, k_values=[1, 3])

        assert report.mrr == 1.0
        assert report.map_score == 1.0
        assert report.precision[1] == 1.0
        assert report.num_queries == 2

    def test_no_relevant_found(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([
            EvalQuery("q1", ["d1"]),
        ])
        search = self._dummy_search({"q1": ["d2", "d3"]})
        report = evaluator.evaluate(search, k_values=[1, 3])

        assert report.mrr == 0.0
        assert report.map_score == 0.0
        assert report.recall[3] == 0.0

    def test_mixed_results(self):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([
            EvalQuery("q1", ["d2"]),  # relevant at position 2
        ])
        search = self._dummy_search({"q1": ["d1", "d2", "d3"]})
        report = evaluator.evaluate(search, k_values=[1, 3])

        assert report.mrr == 0.5
        assert report.precision[1] == 0.0
        assert report.recall[3] == 1.0

    def test_evaluate_no_queries_raises(self):
        evaluator = RetrievalEvaluator()
        with pytest.raises(ValueError, match="No evaluation queries"):
            evaluator.evaluate(lambda q, k: [])

    def test_load_queries_jsonl(self, tmp_path):
        data = tmp_path / "queries.jsonl"
        data.write_text(
            '{"query": "q1", "relevant_docs": ["d1"]}\n'
            '{"query": "q2", "relevant_docs": ["d2", "d3"]}\n'
        )
        evaluator = RetrievalEvaluator()
        loaded = evaluator.load_queries_jsonl(data)
        assert loaded == 2
        assert evaluator._queries[1].relevant_docs == ["d2", "d3"]

    def test_report_save_and_summary(self, tmp_path):
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([EvalQuery("q1", ["d1"])])
        search = self._dummy_search({"q1": ["d1"]})
        report = evaluator.evaluate(search, k_values=[1, 5], model_name="test-model")

        # Save
        out = tmp_path / "report.json"
        report.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["model_name"] == "test-model"
        assert loaded["mrr"] == 1.0

        # Summary (just check it doesn't crash)
        report.print_summary()

    def test_graded_ndcg(self):
        """Graded relevance passes through to NDCG computation."""
        evaluator = RetrievalEvaluator()
        evaluator.add_queries([
            EvalQuery(
                query="q1",
                relevant_docs=["d1", "d2"],
                relevance_grades={"d1": 3, "d2": 1},
            )
        ])
        # Perfect order: d1(3), d2(1)
        perfect_search = self._dummy_search({"q1": ["d1", "d2", "d3"]})
        report = evaluator.evaluate(perfect_search, k_values=[2])
        assert report.ndcg[2] == pytest.approx(1.0)

        # Reversed order: d2(1), d1(3)
        reversed_search = self._dummy_search({"q1": ["d2", "d1", "d3"]})
        report2 = evaluator.evaluate(reversed_search, k_values=[2])
        assert report2.ndcg[2] < 1.0


class TestEvalReport:
    """Tests for EvalReport methods."""

    def test_to_dict_structure(self):
        report = EvalReport(
            num_queries=5,
            k_values=[1, 5],
            mrr=0.8,
            map_score=0.75,
            ndcg={1: 0.7, 5: 0.85},
            precision={1: 0.6, 5: 0.5},
            recall={1: 0.3, 5: 0.9},
            per_query=[],
            elapsed_seconds=1.5,
            model_name="test",
        )
        d = report.to_dict()
        assert d["mrr"] == 0.8
        assert d["model_name"] == "test"
        assert 5 in d["ndcg"]


# =========================================================================
# Integration-like Tests (with mock models)
# =========================================================================


class TestEvalQueryHelpers:
    """Test EvalQuery helper methods."""

    def test_binary_relevance(self):
        eq = EvalQuery("q", ["d1", "d2"])
        assert eq.get_grade("d1") == 1
        assert eq.get_grade("d3") == 0

    def test_graded_relevance(self):
        eq = EvalQuery("q", ["d1"], relevance_grades={"d1": 5, "d2": 2})
        assert eq.get_grade("d1") == 5
        assert eq.get_grade("d2") == 2
        assert eq.get_grade("d3") == 0


class TestBenchmarkResult:
    """Test BenchmarkResult serialization."""

    def test_save(self, tmp_path):
        result = BenchmarkResult(
            models=["m1", "m2"],
            reports={"m1": {"mrr": 0.8}, "m2": {"mrr": 0.6}},
            ranking=[("m1", 0.8), ("m2", 0.6)],
            best_model="m1",
            elapsed_seconds=10.0,
        )
        out = tmp_path / "bench.json"
        result.save(out)
        loaded = json.loads(out.read_text())
        assert loaded["best_model"] == "m1"
        assert len(loaded["ranking"]) == 2

    def test_print_comparison(self, capsys):
        result = BenchmarkResult(
            models=["m1"],
            reports={"m1": {"mrr": 0.9, "map": 0.85, "ndcg": {5: 0.88}, "recall": {10: 0.95}}},
            ranking=[("m1", 0.9)],
            best_model="m1",
            elapsed_seconds=5.0,
        )
        result.print_comparison()
        captured = capsys.readouterr()
        assert "m1" in captured.out
        assert "🏆" in captured.out
