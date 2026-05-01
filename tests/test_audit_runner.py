"""Tests for the audit_runner loaders + stdlib-only run path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audit_runner import load_documents, load_queries, run_audit

# ---------------------------------------------------------------------------
# load_documents
# ---------------------------------------------------------------------------


def test_load_documents_text_file_one_per_line(tmp_path: Path):
    path = tmp_path / "docs.txt"
    path.write_text("alpha\nbeta\n\ngamma\n", encoding="utf-8")
    assert load_documents(path) == ["alpha", "beta", "gamma"]


def test_load_documents_jsonl_string_rows(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    path.write_text('"alpha"\n"beta"\n', encoding="utf-8")
    assert load_documents(path) == ["alpha", "beta"]


def test_load_documents_jsonl_object_rows_with_text_field(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    path.write_text(
        '{"text": "alpha"}\n{"text": "beta"}\n',
        encoding="utf-8",
    )
    assert load_documents(path) == ["alpha", "beta"]


def test_load_documents_rejects_invalid_jsonl(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_documents(path)


def test_load_documents_rejects_object_without_text(tmp_path: Path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"body": "wrong key"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be a string or object"):
        load_documents(path)


# ---------------------------------------------------------------------------
# load_queries
# ---------------------------------------------------------------------------


def test_load_queries_text_file(tmp_path: Path):
    path = tmp_path / "queries.txt"
    path.write_text("query one\nquery two\n", encoding="utf-8")
    assert load_queries(path) == ["query one", "query two"]


def test_load_queries_jsonl_query_field(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text(
        '{"query": "first"}\n{"query": "second"}\n',
        encoding="utf-8",
    )
    assert load_queries(path) == ["first", "second"]


def test_load_queries_jsonl_text_field_supported(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"text": "alpha"}\n', encoding="utf-8")
    assert load_queries(path) == ["alpha"]


def test_load_queries_jsonl_string_rows(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text('"first"\n"second"\n', encoding="utf-8")
    assert load_queries(path) == ["first", "second"]


def test_load_queries_rejects_object_without_query_or_text(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    path.write_text('{"body": "wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must carry 'text' or 'query'"):
        load_queries(path)


# ---------------------------------------------------------------------------
# run_audit (stdlib-only path)
# ---------------------------------------------------------------------------


def test_run_audit_no_embedding_stats_returns_minimal_report():
    docs = ["alpha doc", "beta doc", "gamma doc"]
    report = run_audit(docs, include_embedding_stats=False)
    assert report.length.n == 3
    assert report.vocabulary.unique_tokens > 0
    assert report.embedding is None
    assert report.near_duplicates is None
    assert report.coverage is None


def test_run_audit_serialises_round_trip():
    docs = ["one document", "two document"]
    report = run_audit(docs, include_embedding_stats=False)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["n_documents"] == 2
    # Optional sections are omitted when not computed.
    assert "embedding" not in payload
    assert "coverage" not in payload


def test_run_audit_handles_empty_documents_gracefully():
    report = run_audit([], include_embedding_stats=False)
    assert report.length.n == 0
    assert report.vocabulary.total_tokens == 0
    assert report.exact_duplicates.n_documents == 0
