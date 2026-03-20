"""Search engine benchmarking suite.

Measures latency, throughput, and quality metrics for semantic search
pipelines. Generates reproducible benchmark reports.

Metrics:
    - p50/p95/p99 latency
    - Queries per second (QPS)
    - Index build time
    - Memory usage
    - Result quality (MRR, NDCG, Recall@K)

Usage::

    bench = SearchBenchmark(engine)
    report = bench.run(queries, k=10, iterations=100)
    print(report.summary())
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LatencyStats:
    """Latency distribution statistics.

    Attributes:
        p50: Median latency in milliseconds.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        mean: Average latency.
        min: Minimum latency.
        max: Maximum latency.
        std: Standard deviation.
    """

    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    std: float = 0.0

    @staticmethod
    def from_samples(samples_ms: list[float]) -> LatencyStats:
        """Compute stats from a list of latency samples in milliseconds."""
        if not samples_ms:
            return LatencyStats()
        s = sorted(samples_ms)
        n = len(s)
        return LatencyStats(
            p50=s[int(n * 0.50)],
            p95=s[int(n * 0.95)] if n >= 20 else s[-1],
            p99=s[int(n * 0.99)] if n >= 100 else s[-1],
            mean=statistics.mean(s),
            min=s[0],
            max=s[-1],
            std=statistics.stdev(s) if n >= 2 else 0.0,
        )


@dataclass
class QualityMetrics:
    """Search quality metrics.

    Attributes:
        mrr: Mean Reciprocal Rank.
        ndcg_at_k: Normalised Discounted Cumulative Gain at K.
        recall_at_k: Recall at K (fraction of relevant docs retrieved).
        precision_at_k: Precision at K.
        queries_evaluated: Number of queries with ground truth.
    """

    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    queries_evaluated: int = 0


@dataclass
class BenchmarkReport:
    """Complete benchmark report.

    Attributes:
        name: Benchmark name/description.
        latency: Latency distribution stats.
        qps: Queries per second throughput.
        total_queries: Total queries executed.
        total_time_seconds: Wall clock time.
        quality: Optional quality metrics.
        metadata: Additional context.
    """

    name: str = "benchmark"
    latency: LatencyStats = field(default_factory=LatencyStats)
    qps: float = 0.0
    total_queries: int = 0
    total_time_seconds: float = 0.0
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== {self.name} ===",
            f"Queries: {self.total_queries} in {self.total_time_seconds:.2f}s",
            f"QPS: {self.qps:.1f}",
            f"Latency: p50={self.latency.p50:.1f}ms p95={self.latency.p95:.1f}ms p99={self.latency.p99:.1f}ms",
            f"Range: {self.latency.min:.1f}ms - {self.latency.max:.1f}ms (std={self.latency.std:.1f}ms)",
        ]
        if self.quality.queries_evaluated > 0:
            lines.extend(
                [
                    f"MRR: {self.quality.mrr:.4f}",
                    f"NDCG@K: {self.quality.ndcg_at_k:.4f}",
                    f"Recall@K: {self.quality.recall_at_k:.4f}",
                ]
            )
        return "\n".join(lines)


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg(relevances: list[float], ideal: list[float], k: int) -> float:
    """Normalised DCG."""
    idcg = _dcg(sorted(ideal, reverse=True), k)
    if idcg == 0:
        return 0.0
    return _dcg(relevances, k) / idcg


class SearchBenchmark:
    """Benchmark harness for search engines.

    Args:
        search_fn: Function that takes (query: str, k: int) and returns results.
        name: Benchmark name.

    Example::

        def my_search(query, k):
            return engine.search(query, top_k=k)

        bench = SearchBenchmark(my_search)
        report = bench.run(["contract law", "property rights"], k=10)
        print(report.summary())
    """

    def __init__(self, search_fn: Callable, name: str = "search_benchmark") -> None:
        """Initialise with a search function."""
        self.search_fn = search_fn
        self.name = name

    def run(
        self,
        queries: list[str],
        k: int = 10,
        iterations: int = 1,
        warmup: int = 0,
        ground_truth: dict[str, list[str]] | None = None,
    ) -> BenchmarkReport:
        """Run the benchmark.

        Args:
            queries: List of search queries.
            k: Number of results per query.
            iterations: Times to repeat each query (for stable latency).
            warmup: Warmup iterations (not measured).
            ground_truth: Optional {query: [relevant_doc_ids]} for quality metrics.

        Returns:
            BenchmarkReport with latency, throughput, and quality stats.
        """
        # Warmup
        for _ in range(warmup):
            for q in queries[:5]:
                self.search_fn(q, k)

        gc.collect()

        # Measure
        latencies: list[float] = []
        all_results: dict[str, list[Any]] = {}
        start_total = time.perf_counter()

        for _ in range(iterations):
            for query in queries:
                start = time.perf_counter()
                results = self.search_fn(query, k)
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)
                if query not in all_results:
                    all_results[query] = results

        total_time = time.perf_counter() - start_total
        total_queries = len(queries) * iterations
        qps = total_queries / total_time if total_time > 0 else 0

        # Quality metrics
        quality = QualityMetrics()
        if ground_truth:
            quality = self._compute_quality(all_results, ground_truth, k)

        return BenchmarkReport(
            name=self.name,
            latency=LatencyStats.from_samples(latencies),
            qps=qps,
            total_queries=total_queries,
            total_time_seconds=total_time,
            quality=quality,
            metadata={"k": k, "iterations": iterations, "unique_queries": len(queries)},
        )

    def _compute_quality(
        self,
        results: dict[str, list[Any]],
        ground_truth: dict[str, list[str]],
        k: int,
    ) -> QualityMetrics:
        """Compute MRR, NDCG, Recall, Precision from results vs ground truth."""
        mrr_scores = []
        ndcg_scores = []
        recall_scores = []
        precision_scores = []

        for query, relevant_ids in ground_truth.items():
            if query not in results:
                continue

            result_ids = []
            for r in results[query][:k]:
                if hasattr(r, "chunk") and hasattr(r.chunk, "document_id"):
                    result_ids.append(r.chunk.document_id)
                elif hasattr(r, "id"):
                    result_ids.append(r.id)
                elif isinstance(r, dict):
                    result_ids.append(r.get("id", r.get("document_id", str(r))))
                else:
                    result_ids.append(str(r))

            relevant_set = set(relevant_ids)

            # MRR
            rr = 0.0
            for i, rid in enumerate(result_ids):
                if rid in relevant_set:
                    rr = 1.0 / (i + 1)
                    break
            mrr_scores.append(rr)

            # NDCG
            relevances = [1.0 if rid in relevant_set else 0.0 for rid in result_ids]
            ideal = [1.0] * len(relevant_ids) + [0.0] * (k - len(relevant_ids))
            ndcg_scores.append(_ndcg(relevances, ideal, k))

            # Recall & Precision
            retrieved_relevant = sum(1 for rid in result_ids if rid in relevant_set)
            recall = retrieved_relevant / len(relevant_ids) if relevant_ids else 0.0
            precision = retrieved_relevant / len(result_ids) if result_ids else 0.0
            recall_scores.append(recall)
            precision_scores.append(precision)

        n = len(mrr_scores)
        return QualityMetrics(
            mrr=statistics.mean(mrr_scores) if mrr_scores else 0.0,
            ndcg_at_k=statistics.mean(ndcg_scores) if ndcg_scores else 0.0,
            recall_at_k=statistics.mean(recall_scores) if recall_scores else 0.0,
            precision_at_k=statistics.mean(precision_scores) if precision_scores else 0.0,
            queries_evaluated=n,
        )


class IndexBenchmark:
    """Benchmark harness for index build operations.

    Args:
        index_fn: Function that builds an index from documents.
        name: Benchmark name.
    """

    def __init__(self, index_fn: Callable, name: str = "index_benchmark") -> None:
        """Initialise with an index build function."""
        self.index_fn = index_fn
        self.name = name

    def run(self, documents: list[Any], iterations: int = 1) -> dict[str, Any]:
        """Benchmark index build time.

        Args:
            documents: Documents to index.
            iterations: Number of times to repeat.

        Returns:
            Dict with timing and throughput metrics.
        """
        times = []
        for _ in range(iterations):
            gc.collect()
            start = time.perf_counter()
            self.index_fn(documents)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = statistics.mean(times)
        docs_per_sec = len(documents) / mean_time if mean_time > 0 else 0

        return {
            "name": self.name,
            "documents": len(documents),
            "iterations": iterations,
            "mean_time_seconds": round(mean_time, 3),
            "min_time_seconds": round(min(times), 3),
            "max_time_seconds": round(max(times), 3),
            "docs_per_second": round(docs_per_sec, 1),
        }
