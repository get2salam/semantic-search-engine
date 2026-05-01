"""Tests for the audit_markdown renderer."""

from __future__ import annotations

import numpy as np

from audit_markdown import render_markdown
from audit_report import build_report
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


def _minimal(docs: list[str]):
    return build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
    )


def test_render_markdown_returns_clean_string():
    out = render_markdown(_minimal(["alpha", "beta", "gamma"]))
    assert out.startswith("## RAG Readiness Audit")
    assert out.endswith("\n")
    assert "Length distribution" in out
    assert "Vocabulary" in out
    assert "Exact duplicates" in out


def test_render_markdown_status_badge_for_clean_corpus_is_ready():
    out = render_markdown(_minimal(["alpha doc", "beta doc", "gamma document"]))
    assert "READY" in out


def test_render_markdown_status_badge_for_unhealthy_corpus():
    out = render_markdown(_minimal(["empty placeholder", ""]))
    assert "NEEDS ATTENTION" in out


def test_render_markdown_omits_optional_sections_when_absent():
    out = render_markdown(_minimal(["alpha", "beta"]))
    assert "Near duplicates" not in out
    assert "Embedding-space health" not in out
    assert "Query coverage" not in out


def test_render_markdown_includes_optional_sections_when_present():
    docs = ["alpha doc", "beta doc"]
    embs = _normed([[1.0, 0.0], [0.0, 1.0]])
    report = build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
        near_duplicates=near_duplicate_report(embs, threshold=0.99),
        embedding=embedding_stats(embs),
        coverage=query_coverage_report(["q"], lambda q, k: [("alpha doc", 0.9)]),
    )
    out = render_markdown(report)
    assert "Near duplicates" in out
    assert "Embedding-space health" in out
    assert "Query coverage" in out


def test_render_markdown_no_action_items_when_corpus_clean():
    # Build a corpus with overlapping vocabulary and adequate length so
    # the hapax / very-short / dedup notes all stay quiet.
    base = (
        "machine learning models learn from training data using gradient descent "
        "and tuned hyperparameters that adapt the model to the data distribution"
    )
    docs = [
        base + " for classification problems",
        base + " for regression tasks and forecasting workloads",
        base + " when applied to natural language understanding pipelines",
        base + " across large-scale information retrieval systems",
    ]
    out = render_markdown(_minimal(docs))
    assert "No action items" in out


def test_render_markdown_action_items_present_when_notes_exist():
    docs = ["dup"] * 10 + ["unique"]
    out = render_markdown(_minimal(docs))
    assert "Action items" in out
    assert "deduplicate" in out.lower()


def test_render_markdown_custom_title_used():
    out = render_markdown(_minimal(["doc"]), title="Custom Audit")
    assert out.startswith("## Custom Audit")


def test_render_markdown_lists_worst_uncovered_queries():
    docs = ["alpha"]
    report = build_report(
        length=length_stats(docs),
        vocabulary=vocabulary_stats(docs),
        exact_duplicates=exact_duplicate_report(docs),
        coverage=query_coverage_report(
            ["bad-q-1", "bad-q-2"],
            lambda q, k: [("none", 0.05)],
        ),
    )
    out = render_markdown(report)
    assert "Worst uncovered queries" in out
    assert "bad-q-1" in out
