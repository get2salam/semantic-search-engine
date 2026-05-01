"""Tests for the audit_report aggregator."""

from __future__ import annotations

import json

import numpy as np

from audit_report import RagReadinessReport, build_report
from corpus_profile import (
    exact_duplicate_report,
    length_stats,
    vocabulary_stats,
)
from embedding_stats import embedding_stats
from near_duplicates import near_duplicate_report
from query_coverage import query_coverage_report


def _normed(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def _build_minimal(docs: list[str]) -> RagReadinessReport:
    return build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
    )


def test_build_report_minimal_payload_serialises():
    report = _build_minimal(["alpha", "beta", "gamma"])
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["n_documents"] == 3
    assert "length" in payload and "vocabulary" in payload
    # Optional sections should be omitted, not serialised as nulls.
    assert "near_duplicates" not in payload
    assert "embedding" not in payload
    assert "coverage" not in payload


def test_build_report_attaches_optional_sections():
    docs = ["alpha doc", "beta doc", "gamma doc"]
    embs = _normed([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    report = build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
        near_duplicates=near_duplicate_report(embs, threshold=0.99),
        embedding=embedding_stats(embs),
        coverage=query_coverage_report(
            ["q"],
            lambda q, k: [("alpha doc", 0.9), ("beta doc", 0.2)],
        ),
    )
    payload = report.to_dict()
    assert "near_duplicates" in payload
    assert "embedding" in payload
    assert "coverage" in payload


def test_headline_status_flags_empty_documents():
    report = _build_minimal(["good doc", ""])
    assert report.headline_status() == "needs_attention"


def test_headline_status_flags_high_exact_duplicate_ratio():
    docs = ["dup"] * 5 + ["unique-1", "unique-2"]
    report = _build_minimal(docs)
    assert report.headline_status() == "needs_attention"


def test_headline_status_ready_for_clean_corpus():
    docs = ["alpha document", "beta document with more words", "gamma document content"]
    report = _build_minimal(docs)
    assert report.headline_status() == "ready"


def test_notes_recommend_chunking_long_docs():
    docs = ["short doc"] + ["x" * 10_000]
    report = _build_minimal(docs)
    assert any("chunk" in n.lower() for n in report.notes)


def test_notes_recommend_dedup_when_ratio_high():
    docs = ["dup"] * 10 + ["unique"]
    report = _build_minimal(docs)
    assert any("deduplicate" in n.lower() for n in report.notes)


def test_notes_flag_low_query_coverage():
    docs = ["alpha", "beta"]
    report = build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
        coverage=query_coverage_report(
            ["q1", "q2"],
            lambda q, k: [("nope", 0.05)],
        ),
    )
    assert any("covered" in n.lower() for n in report.notes)


def test_summary_lines_renders_optional_sections():
    docs = ["alpha doc", "beta doc"]
    embs = _normed([[1.0, 0.0], [0.0, 1.0]])
    report = build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
        near_duplicates=near_duplicate_report(embs, threshold=0.99),
        embedding=embedding_stats(embs),
    )
    rendered = "\n".join(report.summary_lines())
    assert "Near duplicates" in rendered
    assert "Embedding health" in rendered
    assert "Vocabulary" in rendered


def test_generated_at_is_utc_iso8601():
    report = _build_minimal(["doc"])
    assert report.generated_at.endswith("Z")
    assert "T" in report.generated_at
