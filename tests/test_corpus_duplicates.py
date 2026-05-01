"""Tests for the exact-duplicate report in corpus_profile."""

from __future__ import annotations

import json

from corpus_profile import (
    DuplicateGroup,
    DuplicateReport,
    exact_duplicate_report,
)


def test_exact_duplicate_report_empty_corpus():
    report = exact_duplicate_report([])
    assert isinstance(report, DuplicateReport)
    assert report.n_documents == 0
    assert report.duplication_ratio == 0.0
    assert report.groups == []


def test_exact_duplicate_report_no_duplicates():
    report = exact_duplicate_report(["alpha", "beta", "gamma"])
    assert report.n_documents == 3
    assert report.n_unique == 3
    assert report.n_duplicate_documents == 0
    assert report.n_groups == 0


def test_exact_duplicate_report_finds_groups():
    docs = [
        "Hello world",
        "hello world",  # case + whitespace normalised match
        "Hello   world",  # multi-space normalised match
        "Different text",
        "Different text",
    ]
    report = exact_duplicate_report(docs)
    assert report.n_groups == 2
    # 5 docs, 2 groups of (3, 2) → 3 duplicates total (n - groups)
    assert report.n_duplicate_documents == 3
    assert report.duplication_ratio == 0.6
    largest = report.groups[0]
    assert isinstance(largest, DuplicateGroup)
    assert len(largest.indices) == 3
    assert largest.representative == 0


def test_exact_duplicate_report_caps_groups():
    # Build 30 distinct duplicate pairs.
    docs = []
    for i in range(30):
        docs.append(f"text {i}")
        docs.append(f"text {i}")
    report = exact_duplicate_report(docs, max_groups=10)
    assert report.n_groups == 30  # aggregate count is honest
    assert len(report.groups) == 10  # only the cap is rendered


def test_exact_duplicate_report_ignore_empty_default():
    docs = ["", "   ", "real document", "", "real document"]
    report = exact_duplicate_report(docs)
    # Two empties are excluded from the duplicate scan.
    assert report.n_groups == 1
    assert report.groups[0].text == "real document"


def test_exact_duplicate_report_serialises():
    docs = ["foo", "foo", "bar"]
    payload = json.loads(json.dumps(exact_duplicate_report(docs).to_dict()))
    assert payload["n_documents"] == 3
    assert isinstance(payload["groups"], list)
    assert payload["groups"][0]["indices"] == [0, 1]
