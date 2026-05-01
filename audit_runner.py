"""
Audit Runner
============
Composes the corpus + coverage analyses end-to-end and returns a
``RagReadinessReport``. Pulled out of the CLI module so it can be
imported from notebooks, tests, or a future REST endpoint without
re-implementing the orchestration.

The runner is deliberately conservative about *what* it loads:

* Corpus is loaded with the same JSONL/text helper the existing
  ``index`` subcommand uses (one doc per line; JSONL with optional
  ``"text"`` field).
* The embedding model is loaded only when the caller asks for the
  embedding-space stats or near-duplicate clustering — keeping a "fast
  text-only audit" path under a second on small corpora.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import json
from pathlib import Path

from audit_report import RagReadinessReport, build_report
from corpus_profile import (
    exact_duplicate_report,
    length_stats,
    vocabulary_stats,
)

__all__ = [
    "load_documents",
    "load_queries",
    "run_audit",
]


def load_documents(path: Path) -> list[str]:
    """Load documents from a text file (one per line) or JSONL.

    JSONL rows may be raw strings or objects with a ``"text"`` field.
    Blank lines are skipped. Errors surface a 1-based line number so the
    caller can pinpoint malformed input.
    """
    docs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
                if isinstance(obj, str):
                    docs.append(obj)
                elif isinstance(obj, dict) and "text" in obj:
                    docs.append(str(obj["text"]))
                else:
                    raise ValueError(
                        f"{path}:{lineno}: JSONL entry must be a string or object with 'text'"
                    )
            else:
                docs.append(line)
    return docs


def load_queries(path: Path) -> list[str]:
    """Load a query set from a text file (one per line) or JSONL.

    JSONL rows may be raw strings, objects with a ``"text"`` field, or
    objects with a ``"query"`` field — matching the conventions used by
    most evaluation toolkits.
    """
    queries: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
                if isinstance(obj, str):
                    queries.append(obj)
                elif isinstance(obj, dict):
                    text = obj.get("text") or obj.get("query")
                    if not text:
                        raise ValueError(
                            f"{path}:{lineno}: JSONL entry must carry 'text' or 'query'"
                        )
                    queries.append(str(text))
                else:
                    raise ValueError(f"{path}:{lineno}: unsupported JSONL row")
            else:
                queries.append(line)
    return queries


def run_audit(
    documents: list[str],
    *,
    queries: list[str] | None = None,
    near_dup_threshold: float = 0.92,
    coverage_threshold: float = 0.30,
    clarity_threshold: float = 0.10,
    top_k: int = 5,
    include_embedding_stats: bool = True,
    model_name: str | None = None,
) -> RagReadinessReport:
    """Run the full RAG-readiness audit pipeline.

    Args:
        documents: Corpus to profile.
        queries: Optional query set for the coverage probe. Encoder is
            always loaded when queries are provided.
        near_dup_threshold: Cosine cut-off for embedding-based near-dup
            clustering. ``None`` would skip near-dup detection but the
            current pipeline always runs it when embeddings are computed.
        coverage_threshold: Minimum top-1 score for a query to count as
            covered. Defaults to ``0.30`` (calibrated to ``all-MiniLM``).
        clarity_threshold: Minimum :func:`diagnostics.query_difficulty`
            clarity for a query to count as confident.
        top_k: Top-k passed to the search function during the coverage
            probe.
        include_embedding_stats: When ``True`` (default) the encoder is
            loaded and embedding-space + near-duplicate stats are
            attached to the report. When ``False`` the audit is purely
            stdlib + numpy and runs in milliseconds.
        model_name: Optional sentence-transformer model name. Defaults
            to whatever the engine picks up from settings.

    Returns:
        :class:`audit_report.RagReadinessReport` carrying every
        configured analysis section.
    """
    length = length_stats(documents)
    vocab = vocabulary_stats(documents)
    exact = exact_duplicate_report(documents)

    near = None
    embedding = None
    coverage = None

    needs_embedder = include_embedding_stats or queries is not None
    if needs_embedder and documents:
        # Local imports keep the no-encoder path import-time cheap.
        from embedding_stats import embedding_stats
        from near_duplicates import near_duplicate_report
        from semantic_search import SemanticSearchEngine

        kwargs = {}
        if model_name:
            kwargs["model_name"] = model_name
        engine = SemanticSearchEngine(**kwargs)
        engine.add_documents(documents, show_progress=False)

        if engine.embeddings is not None and include_embedding_stats:
            embedding = embedding_stats(engine.embeddings)
            near = near_duplicate_report(engine.embeddings, threshold=near_dup_threshold)

        if queries:
            from query_coverage import query_coverage_report

            def search(q: str, k: int) -> list[tuple[str, float]]:
                return engine.search(q, top_k=k)

            coverage = query_coverage_report(
                queries,
                search,
                top_k=top_k,
                coverage_threshold=coverage_threshold,
                clarity_threshold=clarity_threshold,
            )

    return build_report(
        length=length,
        vocabulary=vocab,
        exact_duplicates=exact,
        near_duplicates=near,
        embedding=embedding,
        coverage=coverage,
    )
