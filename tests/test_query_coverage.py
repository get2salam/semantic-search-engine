"""Tests for the query_coverage probe."""

from __future__ import annotations

import json

from query_coverage import (
    QueryCoverageReport,
    QueryVerdict,
    classify_query,
    query_coverage_report,
)

# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------


def test_classify_query_empty_results_is_uncovered():
    verdict = classify_query("anything", [])
    assert verdict.bucket == "uncovered"
    assert verdict.top1 == 0.0
    assert verdict.top1_text is None


def test_classify_query_low_top1_is_uncovered():
    results = [("doc", 0.10), ("other", 0.05)]
    verdict = classify_query("q", results, coverage_threshold=0.30)
    assert verdict.bucket == "uncovered"
    assert verdict.top1 == 0.10


def test_classify_query_uniform_top_results_is_ambiguous():
    # Five near-identical scores → clarity ≈ 0
    results = [("a", 0.51), ("b", 0.50), ("c", 0.50), ("d", 0.50), ("e", 0.50)]
    verdict = classify_query("q", results, coverage_threshold=0.30, clarity_threshold=0.10)
    assert verdict.bucket == "ambiguous"
    assert verdict.clarity < 0.10


def test_classify_query_clear_winner_is_confident():
    results = [("a", 0.85), ("b", 0.40), ("c", 0.35), ("d", 0.30), ("e", 0.25)]
    verdict = classify_query("q", results, coverage_threshold=0.30, clarity_threshold=0.10)
    assert verdict.bucket == "confident"
    assert verdict.top1_text == "a"


def test_classify_query_serialises():
    verdict = classify_query("q", [("doc", 0.7), ("other", 0.4)])
    payload = json.loads(json.dumps(verdict.to_dict()))
    assert payload["query"] == "q"
    assert payload["bucket"] == "confident"


# ---------------------------------------------------------------------------
# query_coverage_report
# ---------------------------------------------------------------------------


def _make_search_fn(answers: dict[str, list[tuple[str, float]]]):
    def _search(q: str, k: int) -> list[tuple[str, float]]:
        return answers.get(q, [])[:k]

    return _search


def test_query_coverage_report_empty_queries():
    report = query_coverage_report([], _make_search_fn({}))
    assert isinstance(report, QueryCoverageReport)
    assert report.n_queries == 0
    assert report.coverage_rate == 0.0


def test_query_coverage_report_aggregates_three_buckets():
    answers = {
        "good": [("doc-1", 0.85), ("doc-2", 0.40), ("doc-3", 0.30)],
        "ambiguous": [("a", 0.51), ("b", 0.50), ("c", 0.50)],
        "uncovered": [("nope", 0.10)],
    }
    report = query_coverage_report(
        list(answers.keys()),
        _make_search_fn(answers),
        top_k=3,
        coverage_threshold=0.30,
        clarity_threshold=0.10,
    )
    assert report.n_queries == 3
    assert report.n_confident == 1
    assert report.n_ambiguous == 1
    assert report.n_uncovered == 1
    assert report.coverage_rate == 2 / 3
    assert report.confidence_rate == 1 / 3


def test_query_coverage_report_examples_are_sorted():
    answers = {f"q-{i}": [("doc", score)] for i, score in enumerate([0.05, 0.10, 0.15, 0.20])}
    report = query_coverage_report(
        list(answers.keys()),
        _make_search_fn(answers),
        top_k=1,
        coverage_threshold=0.30,
        examples_per_bucket=2,
    )
    assert report.n_uncovered == 4
    assert len(report.uncovered_examples) == 2
    # Lowest top1 first.
    assert [v.top1 for v in report.uncovered_examples] == [0.05, 0.10]


def test_query_coverage_report_serialises_examples():
    answers = {"missing": [("doc", 0.05)]}
    report = query_coverage_report(["missing"], _make_search_fn(answers))
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["uncovered_examples"][0]["query"] == "missing"
    assert payload["coverage_rate"] == 0.0


def test_query_verdict_dataclass_round_trip():
    v = QueryVerdict("q", "confident", 0.9, "doc", 0.8, 0.5)
    d = v.to_dict()
    assert d["query"] == "q"
    assert d["bucket"] == "confident"
