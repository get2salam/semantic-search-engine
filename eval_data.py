"""
Evaluation dataset loaders (TREC / BEIR-style TSV)
===================================================
Most public retrieval benchmarks ship three tab-separated files:

    corpus.tsv   doc_id \\t text  (optional title column)
    queries.tsv  query_id \\t query_text
    qrels.tsv    query_id \\t iteration \\t doc_id \\t relevance  (TREC)

This module provides a small ``BeirDataset`` container plus a
``load_beir_like`` function that reads the three files and produces
ready-to-use ``EvalQuery`` objects and a corpus mapping. It intentionally
supports the *convention* (TREC 4-col qrels, 2-col queries, 2-or-3 col
corpus) so you can point it at BEIR downloads, CISI, or hand-crafted
datasets with no external dependencies.

Example:
    ds = load_beir_like("datasets/nfcorpus")
    print(f"{len(ds.corpus)} docs, {len(ds.queries)} queries")
    report = evaluator.evaluate(search_fn=my_search, k_values=[1, 5, 10])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluation import EvalQuery


@dataclass
class BeirDataset:
    """A loaded evaluation dataset."""

    corpus: dict[str, str]  # doc_id -> text
    queries: list[EvalQuery]
    name: str | None = None

    def corpus_ids(self) -> list[str]:
        return list(self.corpus.keys())

    def corpus_texts(self) -> list[str]:
        return list(self.corpus.values())


# ---------------------------------------------------------------------------
# TSV parsing helpers
# ---------------------------------------------------------------------------


def _iter_tsv(path: Path, expected_min_cols: int):
    """Yield non-blank TSV rows split on the tab char."""
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < expected_min_cols:
                raise ValueError(
                    f"{path}:{lineno}: expected at least {expected_min_cols} columns, "
                    f"got {len(cols)}"
                )
            yield lineno, cols


def load_corpus_tsv(path: str | Path) -> dict[str, str]:
    """
    Parse a corpus TSV.

    Accepts two forms::

        doc_id \\t text
        doc_id \\t title \\t text

    When three columns are present, ``title`` is prepended to ``text``
    with a space separator (matching the BEIR convention).
    """
    path = Path(path)
    corpus: dict[str, str] = {}
    for lineno, cols in _iter_tsv(path, expected_min_cols=2):
        doc_id = cols[0]
        if len(cols) == 2:
            text = cols[1]
        else:
            title, body = cols[1], "\t".join(cols[2:])
            text = f"{title} {body}".strip() if title else body
        if doc_id in corpus:
            raise ValueError(f"{path}:{lineno}: duplicate doc_id {doc_id!r}")
        corpus[doc_id] = text
    return corpus


def load_queries_tsv(path: str | Path) -> dict[str, str]:
    """Parse a 2-column ``query_id \\t query_text`` TSV."""
    path = Path(path)
    queries: dict[str, str] = {}
    for lineno, cols in _iter_tsv(path, expected_min_cols=2):
        qid, text = cols[0], "\t".join(cols[1:])
        if qid in queries:
            raise ValueError(f"{path}:{lineno}: duplicate query_id {qid!r}")
        queries[qid] = text
    return queries


def load_qrels_tsv(
    path: str | Path, *, drop_zero_relevance: bool = True
) -> dict[str, dict[str, int]]:
    """
    Parse a TREC-format qrels TSV.

    Expected columns: ``query_id \\t iteration \\t doc_id \\t relevance``.
    The ``iteration`` column is ignored (convention places 0 there).

    Returns a nested dict: ``query_id -> doc_id -> relevance_grade``.
    When ``drop_zero_relevance`` is True, rows with grade <= 0 are
    excluded from the result.
    """
    path = Path(path)
    qrels: dict[str, dict[str, int]] = {}
    for lineno, cols in _iter_tsv(path, expected_min_cols=4):
        qid, _iteration, doc_id, grade_str = cols[0], cols[1], cols[2], cols[3]
        try:
            grade = int(grade_str)
        except ValueError as exc:
            raise ValueError(
                f"{path}:{lineno}: relevance grade must be int, got {grade_str!r}"
            ) from exc
        if drop_zero_relevance and grade <= 0:
            continue
        qrels.setdefault(qid, {})[doc_id] = grade
    return qrels


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------


def load_beir_like(
    directory: str | Path,
    *,
    corpus_name: str = "corpus.tsv",
    queries_name: str = "queries.tsv",
    qrels_name: str = "qrels.tsv",
) -> BeirDataset:
    """
    Load a TREC / BEIR-style retrieval dataset from a directory.

    Args:
        directory: Directory containing ``corpus.tsv``, ``queries.tsv``
            and ``qrels.tsv``.
        corpus_name / queries_name / qrels_name: override the file
            names (some BEIR datasets nest qrels under ``qrels/test.tsv``).

    Returns:
        A ``BeirDataset`` with the corpus mapping and a list of
        ``EvalQuery`` objects, one per query that has at least one
        relevant judgment.
    """
    directory = Path(directory)
    corpus = load_corpus_tsv(directory / corpus_name)
    queries = load_queries_tsv(directory / queries_name)
    qrels = load_qrels_tsv(directory / qrels_name)

    eval_queries: list[EvalQuery] = []
    for qid, text in queries.items():
        grades = qrels.get(qid)
        if not grades:
            continue  # skip queries with no relevance judgments
        # Validate that judged docs exist in the corpus (warn silently by
        # dropping the missing ones — matches BEIR tolerance for trec-eval).
        valid_grades = {d: g for d, g in grades.items() if d in corpus}
        if not valid_grades:
            continue
        eval_queries.append(
            EvalQuery(
                query=text,
                relevant_docs=list(valid_grades.keys()),
                relevance_grades=dict(valid_grades),
            )
        )

    return BeirDataset(corpus=corpus, queries=eval_queries, name=directory.name)
