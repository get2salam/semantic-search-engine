"""
Tests for the CLI wrapper.

We stub out SemanticSearchEngine to avoid downloading a model during
unit tests — these tests exercise argument parsing, file I/O, and
output formatting, not the actual embedding model.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

import cli

# ---------------------------------------------------------------------------
# Stub engine
# ---------------------------------------------------------------------------


class _StubEngine:
    model_name = "stub-model"
    embedding_dim = 3
    use_faiss = False

    def __init__(self, model_name: str | None = None, use_faiss: bool = True):
        _StubEngine.model_name = model_name or "stub-model"
        self.docs: list[str] = []
        self._saved_to: Path | None = None

    def add_documents(self, docs, batch_size=64, show_progress=True):
        self.docs.extend(docs)

    def search(self, query, top_k=5, threshold=None, mmr_lambda=None):
        # Return the first top_k docs with decreasing synthetic scores
        out = []
        for i, doc in enumerate(self.docs[:top_k]):
            score = 1.0 - (i * 0.1)
            if threshold is not None and score < threshold:
                continue
            out.append((doc, score))
        return out

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "marker").write_text("ok", encoding="utf-8")
        self._saved_to = Path(path)

    @classmethod
    def load(cls, path):
        engine = cls()
        engine.docs = ["alpha doc", "beta doc", "gamma doc"]
        return engine

    def __len__(self):
        return len(self.docs)


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch):
    """Make the CLI import see a fast stub instead of the real engine."""
    import semantic_search

    monkeypatch.setattr(semantic_search, "SemanticSearchEngine", _StubEngine, raising=True)
    yield


# ---------------------------------------------------------------------------
# _load_documents
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_plain_text(self, tmp_path: Path):
        p = tmp_path / "d.txt"
        p.write_text("first\nsecond\n\nthird\n", encoding="utf-8")
        assert cli._load_documents(p) == ["first", "second", "third"]

    def test_jsonl_strings(self, tmp_path: Path):
        p = tmp_path / "d.jsonl"
        p.write_text('"a"\n"b"\n', encoding="utf-8")
        assert cli._load_documents(p) == ["a", "b"]

    def test_jsonl_objects_with_text(self, tmp_path: Path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"text": "one"}\n{"text": "two"}\n', encoding="utf-8")
        assert cli._load_documents(p) == ["one", "two"]

    def test_jsonl_missing_text_raises(self, tmp_path: Path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"body": "no text"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="must be a string or object"):
            cli._load_documents(p)

    def test_jsonl_bad_json_raises(self, tmp_path: Path):
        p = tmp_path / "d.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            cli._load_documents(p)


# ---------------------------------------------------------------------------
# End-to-end subcommand tests
# ---------------------------------------------------------------------------


class TestIndexCommand:
    def test_index_from_text(self, tmp_path: Path, capsys):
        src = tmp_path / "in.txt"
        src.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        out = tmp_path / "idx"
        rc = cli.main(["index", "--input", str(src), "--output", str(out), "--quiet"])
        assert rc == 0
        assert (out / "marker").exists()
        assert "Indexed 3 documents" in capsys.readouterr().out

    def test_index_missing_input(self, tmp_path: Path, capsys):
        rc = cli.main(
            ["index", "--input", str(tmp_path / "nope.txt"), "--output", str(tmp_path / "idx")]
        )
        assert rc == 2
        assert "input file not found" in capsys.readouterr().err

    def test_index_empty_input(self, tmp_path: Path, capsys):
        src = tmp_path / "empty.txt"
        src.write_text("\n\n", encoding="utf-8")
        rc = cli.main(["index", "--input", str(src), "--output", str(tmp_path / "idx")])
        assert rc == 2
        assert "no documents found" in capsys.readouterr().err


class TestSearchCommand:
    def test_search_text_output(self, tmp_path: Path, capsys):
        idx = tmp_path / "idx"
        idx.mkdir()
        rc = cli.main(["search", "--index", str(idx), "--query", "hello", "--top-k", "2"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Query: 'hello'" in out
        assert "alpha doc" in out
        assert "beta doc" in out

    def test_search_json_output(self, tmp_path: Path, capsys):
        idx = tmp_path / "idx"
        idx.mkdir()
        rc = cli.main(
            ["search", "--index", str(idx), "--query", "hello", "--top-k", "2", "--json"]
        )
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["query"] == "hello"
        assert len(payload["results"]) == 2
        assert payload["results"][0]["rank"] == 1
        assert payload["results"][0]["score"] >= payload["results"][1]["score"]

    def test_search_stdin(self, tmp_path: Path, capsys, monkeypatch):
        idx = tmp_path / "idx"
        idx.mkdir()
        monkeypatch.setattr(sys, "stdin", io.StringIO("first\nsecond\n"))
        rc = cli.main(["search", "--index", str(idx), "--stdin", "--top-k", "1"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Query: 'first'" in out
        assert "Query: 'second'" in out

    def test_search_missing_query(self, tmp_path: Path, capsys):
        idx = tmp_path / "idx"
        idx.mkdir()
        rc = cli.main(["search", "--index", str(idx)])
        assert rc == 2
        assert "provide --query or --stdin" in capsys.readouterr().err

    def test_search_missing_index_dir(self, tmp_path: Path, capsys):
        rc = cli.main(
            ["search", "--index", str(tmp_path / "missing"), "--query", "x"]
        )
        assert rc == 2
        assert "index directory not found" in capsys.readouterr().err


class TestInfoCommand:
    def test_info_text(self, tmp_path: Path, capsys):
        idx = tmp_path / "idx"
        idx.mkdir()
        rc = cli.main(["info", "--index", str(idx)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Documents:" in out
        assert "Model:" in out

    def test_info_json(self, tmp_path: Path, capsys):
        idx = tmp_path / "idx"
        idx.mkdir()
        rc = cli.main(["info", "--index", str(idx), "--json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out.strip())
        assert payload["documents"] == 3
        assert payload["faiss_enabled"] is False


class TestVersionCommand:
    def test_version_prints_something(self, capsys):
        rc = cli.main(["version"])
        assert rc == 0
        assert capsys.readouterr().out.strip()
