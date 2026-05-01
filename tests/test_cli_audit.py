"""Tests for the CLI 'audit' subcommand.

These tests use ``--no-embedding-stats`` so they never load the
sentence-transformer model, keeping the suite fast and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import cli


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def clean_corpus(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "corpus.txt",
        [
            "machine learning models learn from training data using gradient descent",
            "machine learning models adapt to data distribution using gradient descent",
            "machine learning models tuned with gradient descent generalise to new data",
            "machine learning models trained on diverse data outperform shallow baselines",
        ],
    )


@pytest.fixture
def messy_corpus(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "messy.txt",
        [
            "duplicate document",
            "duplicate document",
            "duplicate document",
            "duplicate document",
            "another doc",
            "yet another",
        ],
    )


def test_audit_runs_on_clean_corpus_and_exits_zero(clean_corpus, capsys):
    rc = cli.main(["audit", "--corpus", str(clean_corpus), "--no-embedding-stats"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RAG Readiness Audit" in out
    assert "READY" in out


def test_audit_returns_nonzero_for_unhealthy_corpus(messy_corpus, capsys):
    rc = cli.main(["audit", "--corpus", str(messy_corpus), "--no-embedding-stats"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "NEEDS_ATTENTION" in out or "NEEDS ATTENTION" in out


def test_audit_writes_json_report(clean_corpus, tmp_path: Path):
    out_path = tmp_path / "audit.json"
    rc = cli.main(
        [
            "audit",
            "--corpus",
            str(clean_corpus),
            "--no-embedding-stats",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["n_documents"] == 4
    assert "length" in payload
    assert "vocabulary" in payload


def test_audit_writes_markdown_report(clean_corpus, tmp_path: Path):
    md_path = tmp_path / "audit.md"
    rc = cli.main(
        [
            "audit",
            "--corpus",
            str(clean_corpus),
            "--no-embedding-stats",
            "--markdown",
            str(md_path),
        ]
    )
    assert rc == 0
    body = md_path.read_text(encoding="utf-8")
    assert body.startswith("## RAG Readiness Audit")
    assert "Length distribution" in body


def test_audit_json_stdout_emits_parseable_payload(clean_corpus, capsys):
    rc = cli.main(["audit", "--corpus", str(clean_corpus), "--no-embedding-stats", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["n_documents"] == 4
    assert rc == 0


def test_audit_missing_corpus_returns_usage_error(tmp_path: Path, capsys):
    rc = cli.main(
        [
            "audit",
            "--corpus",
            str(tmp_path / "nope.txt"),
            "--no-embedding-stats",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "not found" in err


def test_audit_missing_queries_returns_usage_error(clean_corpus, tmp_path: Path, capsys):
    rc = cli.main(
        [
            "audit",
            "--corpus",
            str(clean_corpus),
            "--queries",
            str(tmp_path / "nope.jsonl"),
            "--no-embedding-stats",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "not found" in err


def test_audit_empty_corpus_rejected(tmp_path: Path, capsys):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n\n", encoding="utf-8")
    rc = cli.main(["audit", "--corpus", str(empty), "--no-embedding-stats"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "no documents" in err
