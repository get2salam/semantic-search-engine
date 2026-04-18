"""
Tests for the TREC/BEIR-style TSV dataset loader.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval_data import (
    load_beir_like,
    load_corpus_tsv,
    load_qrels_tsv,
    load_queries_tsv,
)

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


class TestLoadCorpus:
    def test_two_column(self, tmp_path: Path):
        p = tmp_path / "corpus.tsv"
        p.write_text("d1\tfirst doc\nd2\tsecond doc\n", encoding="utf-8")
        corpus = load_corpus_tsv(p)
        assert corpus == {"d1": "first doc", "d2": "second doc"}

    def test_three_column_title_joined(self, tmp_path: Path):
        p = tmp_path / "corpus.tsv"
        p.write_text("d1\tIntro\tbody text\nd2\t\tno title here\n", encoding="utf-8")
        corpus = load_corpus_tsv(p)
        assert corpus["d1"] == "Intro body text"
        assert corpus["d2"] == "no title here"

    def test_blank_lines_skipped(self, tmp_path: Path):
        p = tmp_path / "corpus.tsv"
        p.write_text("\nd1\thello\n\n", encoding="utf-8")
        assert load_corpus_tsv(p) == {"d1": "hello"}

    def test_duplicate_doc_id_raises(self, tmp_path: Path):
        p = tmp_path / "corpus.tsv"
        p.write_text("d1\ta\nd1\tb\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate doc_id"):
            load_corpus_tsv(p)

    def test_too_few_columns_raises(self, tmp_path: Path):
        p = tmp_path / "corpus.tsv"
        p.write_text("justonecolumn\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected at least 2 columns"):
            load_corpus_tsv(p)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


class TestLoadQueries:
    def test_basic(self, tmp_path: Path):
        p = tmp_path / "queries.tsv"
        p.write_text("q1\twhat is AI\nq2\tclimate change\n", encoding="utf-8")
        assert load_queries_tsv(p) == {"q1": "what is AI", "q2": "climate change"}

    def test_query_containing_tab_preserved(self, tmp_path: Path):
        # Extra tabs in the query text are joined back together
        p = tmp_path / "queries.tsv"
        p.write_text("q1\tpart A\tpart B\n", encoding="utf-8")
        assert load_queries_tsv(p) == {"q1": "part A\tpart B"}

    def test_duplicate_query_id_raises(self, tmp_path: Path):
        p = tmp_path / "queries.tsv"
        p.write_text("q1\ta\nq1\tb\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate query_id"):
            load_queries_tsv(p)


# ---------------------------------------------------------------------------
# Qrels
# ---------------------------------------------------------------------------


class TestLoadQrels:
    def test_trec_format(self, tmp_path: Path):
        p = tmp_path / "qrels.tsv"
        p.write_text(
            "q1\t0\td1\t1\n"
            "q1\t0\td2\t3\n"
            "q2\t0\td3\t2\n",
            encoding="utf-8",
        )
        qrels = load_qrels_tsv(p)
        assert qrels == {"q1": {"d1": 1, "d2": 3}, "q2": {"d3": 2}}

    def test_drops_zero_relevance_by_default(self, tmp_path: Path):
        p = tmp_path / "qrels.tsv"
        p.write_text("q1\t0\td1\t1\nq1\t0\td2\t0\n", encoding="utf-8")
        assert load_qrels_tsv(p) == {"q1": {"d1": 1}}

    def test_keep_zero_when_requested(self, tmp_path: Path):
        p = tmp_path / "qrels.tsv"
        p.write_text("q1\t0\td1\t0\n", encoding="utf-8")
        assert load_qrels_tsv(p, drop_zero_relevance=False) == {"q1": {"d1": 0}}

    def test_invalid_grade_raises(self, tmp_path: Path):
        p = tmp_path / "qrels.tsv"
        p.write_text("q1\t0\td1\tnot-a-number\n", encoding="utf-8")
        with pytest.raises(ValueError, match="relevance grade must be int"):
            load_qrels_tsv(p)


# ---------------------------------------------------------------------------
# Full loader
# ---------------------------------------------------------------------------


def _mk_beir(tmp_path: Path):
    (tmp_path / "corpus.tsv").write_text(
        "d1\tArticle about cats\nd2\tArticle about dogs\nd3\tRecipe for pasta\n",
        encoding="utf-8",
    )
    (tmp_path / "queries.tsv").write_text(
        "q1\tfeline companions\nq2\tcanine friends\nq3\ttasty lunch\n",
        encoding="utf-8",
    )
    (tmp_path / "qrels.tsv").write_text(
        "q1\t0\td1\t2\n"
        "q2\t0\td2\t3\n"
        "q3\t0\td3\t1\n"
        "q3\t0\tdNOPE\t2\n",  # non-existent doc should be dropped
        encoding="utf-8",
    )
    return tmp_path


class TestLoadBeirLike:
    def test_end_to_end(self, tmp_path: Path):
        ds = load_beir_like(_mk_beir(tmp_path))
        assert len(ds.corpus) == 3
        assert len(ds.queries) == 3
        assert ds.name == tmp_path.name

        # Queries are keyed by the original query text, with graded relevance
        q_by_text = {q.query: q for q in ds.queries}
        assert q_by_text["feline companions"].relevant_docs == ["d1"]
        assert q_by_text["feline companions"].relevance_grades == {"d1": 2}

    def test_missing_docs_dropped_from_grades(self, tmp_path: Path):
        ds = load_beir_like(_mk_beir(tmp_path))
        q_by_text = {q.query: q for q in ds.queries}
        # d3 is valid, dNOPE is not
        assert q_by_text["tasty lunch"].relevant_docs == ["d3"]
        assert "dNOPE" not in q_by_text["tasty lunch"].relevance_grades

    def test_query_with_no_judgments_skipped(self, tmp_path: Path):
        _mk_beir(tmp_path)
        # Add a 4th query with no qrels entry
        (tmp_path / "queries.tsv").write_text(
            "q1\tfeline companions\nq2\tcanine friends\nq3\ttasty lunch\nq4\tnone\n",
            encoding="utf-8",
        )
        ds = load_beir_like(tmp_path)
        texts = {q.query for q in ds.queries}
        assert "none" not in texts
        assert len(ds.queries) == 3

    def test_corpus_helpers(self, tmp_path: Path):
        ds = load_beir_like(_mk_beir(tmp_path))
        assert ds.corpus_ids() == ["d1", "d2", "d3"]
        assert len(ds.corpus_texts()) == 3

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_beir_like(tmp_path)
