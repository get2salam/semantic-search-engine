"""Tests for search benchmarking suite."""

from __future__ import annotations

import time

from benchmark import (
    BenchmarkReport,
    IndexBenchmark,
    LatencyStats,
    QualityMetrics,
    SearchBenchmark,
    _dcg,
    _ndcg,
)


def dummy_search(query: str, k: int = 10) -> list[dict]:
    time.sleep(0.001)
    return [{"id": f"doc_{i}", "score": 1.0 - i * 0.1} for i in range(k)]


class TestLatencyStats:
    def test_from_samples(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        stats = LatencyStats.from_samples(samples)
        assert stats.min == 1.0
        assert stats.max == 10.0
        assert 5.0 <= stats.mean <= 6.0
        assert 5.0 <= stats.p50 <= 6.0

    def test_empty(self) -> None:
        stats = LatencyStats.from_samples([])
        assert stats.p50 == 0.0
        assert stats.mean == 0.0

    def test_single(self) -> None:
        stats = LatencyStats.from_samples([5.0])
        assert stats.p50 == 5.0
        assert stats.std == 0.0


class TestQualityMetrics:
    def test_defaults(self) -> None:
        q = QualityMetrics()
        assert q.mrr == 0.0
        assert q.queries_evaluated == 0


class TestDCG:
    def test_basic(self) -> None:
        assert _dcg([1.0, 0.0, 1.0], 3) > 0

    def test_perfect(self) -> None:
        assert _dcg([1.0, 1.0, 1.0], 3) > _dcg([0.0, 0.0, 1.0], 3)

    def test_empty(self) -> None:
        assert _dcg([], 5) == 0.0


class TestNDCG:
    def test_perfect(self) -> None:
        assert _ndcg([1.0, 1.0], [1.0, 1.0], 2) == 1.0

    def test_zero(self) -> None:
        assert _ndcg([0.0, 0.0], [1.0, 1.0], 2) == 0.0

    def test_no_ideal(self) -> None:
        assert _ndcg([1.0], [], 1) == 0.0


class TestSearchBenchmark:
    def test_basic_run(self) -> None:
        bench = SearchBenchmark(dummy_search, name="test")
        report = bench.run(["query1", "query2"], k=5, iterations=2)
        assert report.total_queries == 4
        assert report.qps > 0
        assert report.latency.p50 > 0

    def test_warmup(self) -> None:
        bench = SearchBenchmark(dummy_search)
        report = bench.run(["q1"], k=5, warmup=2)
        assert report.total_queries == 1

    def test_summary(self) -> None:
        bench = SearchBenchmark(dummy_search, name="my_bench")
        report = bench.run(["q1", "q2"], k=5)
        s = report.summary()
        assert "my_bench" in s
        assert "QPS" in s
        assert "p50" in s

    def test_quality_metrics(self) -> None:
        def search_with_ids(query, k):
            return [{"id": f"doc_{i}"} for i in range(k)]

        bench = SearchBenchmark(search_with_ids)
        gt = {"q1": ["doc_0", "doc_1"], "q2": ["doc_0"]}
        report = bench.run(["q1", "q2"], k=5, ground_truth=gt)
        assert report.quality.mrr > 0
        assert report.quality.recall_at_k > 0
        assert report.quality.queries_evaluated == 2

    def test_no_ground_truth(self) -> None:
        bench = SearchBenchmark(dummy_search)
        report = bench.run(["q1"], k=5)
        assert report.quality.queries_evaluated == 0


class TestIndexBenchmark:
    def test_basic(self) -> None:
        def build_index(docs):
            time.sleep(0.01)

        bench = IndexBenchmark(build_index, name="idx_test")
        result = bench.run(["doc1", "doc2", "doc3"], iterations=2)
        assert result["documents"] == 3
        assert result["iterations"] == 2
        assert result["docs_per_second"] > 0
        assert result["name"] == "idx_test"


class TestBenchmarkReport:
    def test_defaults(self) -> None:
        r = BenchmarkReport()
        assert r.total_queries == 0
        assert r.qps == 0.0

    def test_summary_with_quality(self) -> None:
        r = BenchmarkReport(
            name="test",
            latency=LatencyStats(p50=5.0, p95=10.0, p99=15.0, mean=6.0, min=1.0, max=20.0, std=3.0),
            qps=100.0,
            total_queries=1000,
            total_time_seconds=10.0,
            quality=QualityMetrics(mrr=0.85, ndcg_at_k=0.9, recall_at_k=0.75, queries_evaluated=50),
        )
        s = r.summary()
        assert "MRR" in s
        assert "NDCG" in s
