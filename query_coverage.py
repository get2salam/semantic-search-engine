"""
Query Coverage Probe
====================
For a given query set and indexed corpus, classify each query into one of
three buckets and report aggregate coverage:

* **uncovered** — the top-1 similarity is below ``coverage_threshold``.
  No document in the corpus is a plausible answer; either the corpus has
  a blind spot or the query is out-of-domain. These are the queries that
  most often surface as "I don't know" / hallucination triggers in RAG.
* **ambiguous** — the top-1 is above threshold but ``query_difficulty``
  clarity is low (top-1 doesn't dominate the rest). Several documents
  look equally plausible, which is a hallmark of vague queries or
  queries that touch overlapping subdomains. RAG systems often need a
  reranker — or query rewriting — to disambiguate.
* **confident** — strong top-1 with a clear margin.

This is the companion piece to :mod:`corpus_profile` /
:mod:`near_duplicates` — together they let teams pre-flight a RAG
deployment without ever shipping a hallucinated answer to a user.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from diagnostics import query_difficulty

__all__ = [
    "QueryVerdict",
    "QueryCoverageReport",
    "classify_query",
    "query_coverage_report",
]


# Search function contract: (query: str, top_k: int) -> list[(doc, score)].
SearchFn = Callable[[str, int], list[tuple[str, float]]]


@dataclass(frozen=True)
class QueryVerdict:
    """Per-query classification produced by the coverage probe."""

    query: str
    bucket: str  # "uncovered" | "ambiguous" | "confident"
    top1: float
    top1_text: str | None
    clarity: float
    score_drop: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryCoverageReport:
    """Aggregate coverage stats across a query set.

    Attributes:
        n_queries: Total number of queries probed.
        coverage_threshold: Minimum top-1 to count as covered.
        clarity_threshold: Minimum clarity to count as confident.
        n_uncovered / n_ambiguous / n_confident: Bucket counts.
        coverage_rate: ``(n_ambiguous + n_confident) / n_queries``. The
            fraction of queries the corpus has *some* plausible answer for.
        confidence_rate: ``n_confident / n_queries``. The fraction the
            engine answers without ambiguity.
        uncovered_examples: Up to ``examples_per_bucket`` representative
            uncovered queries (lowest top-1 first).
        ambiguous_examples: Up to ``examples_per_bucket`` representative
            ambiguous queries (lowest clarity first).
    """

    n_queries: int
    coverage_threshold: float
    clarity_threshold: float
    n_uncovered: int
    n_ambiguous: int
    n_confident: int
    coverage_rate: float
    confidence_rate: float
    uncovered_examples: list[QueryVerdict] = field(default_factory=list)
    ambiguous_examples: list[QueryVerdict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uncovered_examples"] = [v.to_dict() for v in self.uncovered_examples]
        d["ambiguous_examples"] = [v.to_dict() for v in self.ambiguous_examples]
        return d


def classify_query(
    query: str,
    results: list[tuple[str, float]],
    *,
    coverage_threshold: float = 0.30,
    clarity_threshold: float = 0.10,
) -> QueryVerdict:
    """Classify a single query's results into one of three buckets.

    Args:
        query: The query text (for the verdict record).
        results: Engine output: ``[(doc, score), ...]`` ordered best-first.
        coverage_threshold: Minimum top-1 score to count as covered. The
            default ``0.30`` is calibrated to ``all-MiniLM-L6-v2`` cosine
            scores — tune for other encoders.
        clarity_threshold: Minimum :func:`diagnostics.query_difficulty`
            clarity for the verdict to be ``confident``. Below this,
            the result list is too uniform to commit to a single best.

    Returns:
        :class:`QueryVerdict` carrying the bucket and the diagnostics that
        drove it.
    """
    if not results:
        return QueryVerdict(
            query=query,
            bucket="uncovered",
            top1=0.0,
            top1_text=None,
            clarity=0.0,
            score_drop=0.0,
        )

    scores = [s for _, s in results]
    top1, top1_text = scores[0], results[0][0]
    diff = query_difficulty(scores)

    if top1 < coverage_threshold:
        bucket = "uncovered"
    elif diff.clarity < clarity_threshold:
        bucket = "ambiguous"
    else:
        bucket = "confident"

    return QueryVerdict(
        query=query,
        bucket=bucket,
        top1=float(top1),
        top1_text=top1_text,
        clarity=float(diff.clarity),
        score_drop=float(diff.score_drop),
    )


def query_coverage_report(
    queries: list[str],
    search_fn: SearchFn,
    *,
    top_k: int = 5,
    coverage_threshold: float = 0.30,
    clarity_threshold: float = 0.10,
    examples_per_bucket: int = 10,
) -> QueryCoverageReport:
    """Run the coverage probe across a query set.

    Args:
        queries: List of query strings.
        search_fn: A retrieval callable: ``(query, top_k) -> [(doc, score)]``.
            The engine's ``SemanticSearchEngine.search`` is the canonical
            implementation — but any backend works.
        top_k: Number of results to pull per query.
        coverage_threshold: See :func:`classify_query`.
        clarity_threshold: See :func:`classify_query`.
        examples_per_bucket: Cap on the per-bucket example list.

    Returns:
        :class:`QueryCoverageReport` with bucket counts, coverage rate,
        confidence rate and representative example queries.
    """
    n = len(queries)
    if n == 0:
        return QueryCoverageReport(
            n_queries=0,
            coverage_threshold=coverage_threshold,
            clarity_threshold=clarity_threshold,
            n_uncovered=0,
            n_ambiguous=0,
            n_confident=0,
            coverage_rate=0.0,
            confidence_rate=0.0,
        )

    verdicts: list[QueryVerdict] = []
    for q in queries:
        results = search_fn(q, top_k)
        verdicts.append(
            classify_query(
                q,
                results,
                coverage_threshold=coverage_threshold,
                clarity_threshold=clarity_threshold,
            )
        )

    n_unc = sum(1 for v in verdicts if v.bucket == "uncovered")
    n_amb = sum(1 for v in verdicts if v.bucket == "ambiguous")
    n_con = sum(1 for v in verdicts if v.bucket == "confident")

    uncovered = sorted(
        (v for v in verdicts if v.bucket == "uncovered"),
        key=lambda v: v.top1,
    )[:examples_per_bucket]
    ambiguous = sorted(
        (v for v in verdicts if v.bucket == "ambiguous"),
        key=lambda v: v.clarity,
    )[:examples_per_bucket]

    return QueryCoverageReport(
        n_queries=n,
        coverage_threshold=coverage_threshold,
        clarity_threshold=clarity_threshold,
        n_uncovered=n_unc,
        n_ambiguous=n_amb,
        n_confident=n_con,
        coverage_rate=(n_amb + n_con) / n,
        confidence_rate=n_con / n,
        uncovered_examples=uncovered,
        ambiguous_examples=ambiguous,
    )
