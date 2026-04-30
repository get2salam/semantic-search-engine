"""
End-to-End Result Explanation
=============================
Combines query LOO and document contribution analyses into a single
``ExplainedResult`` record per (query, result) pair.

This is the primary entry point for the workbench: a caller wires up an
encoder once and then calls :func:`explain_result` (or :func:`explain_results`
for batches) to get a flat, JSON-serialisable explanation that can be
rendered, logged, or sent over the wire.

The companion modules :mod:`explain` and :mod:`contributions` stay focused on
the underlying perturbation passes and remain useful on their own.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from contributions import SpanContribution, TokenContribution, contributing_spans, contributing_tokens
from explain import QueryExplanation, _encode, explain_query, tokenize

__all__ = [
    "ExplainedResult",
    "explain_result",
    "explain_results",
]


# Documents above this token count default to span-level contribution analysis
# rather than per-token LOO. Empirically, short docs (titles, snippets) deserve
# fine-grained attribution; long docs (paragraphs) are easier to read at the
# phrase level and the per-token cost stops being worth it.
_LONG_DOC_THRESHOLD = 30


@dataclass
class ExplainedResult:
    """A single result row plus its full explanation."""

    rank: int
    query: str
    document: str
    score: float
    query_tokens: list[TokenContribution | dict[str, Any]] = field(default_factory=list)
    document_tokens: list[TokenContribution | dict[str, Any]] = field(default_factory=list)
    document_spans: list[SpanContribution | dict[str, Any]] = field(default_factory=list)

    # Lightweight metadata so downstream renderers don't need a second pass.
    n_query_tokens: int = 0
    n_document_tokens: int = 0
    contribution_mode: str = "tokens"  # "tokens" | "spans"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "query": self.query,
            "document": self.document,
            "score": self.score,
            "n_query_tokens": self.n_query_tokens,
            "n_document_tokens": self.n_document_tokens,
            "contribution_mode": self.contribution_mode,
            "query_tokens": [
                t.to_dict() if hasattr(t, "to_dict") else t for t in self.query_tokens
            ],
            "document_tokens": [
                t.to_dict() if hasattr(t, "to_dict") else t for t in self.document_tokens
            ],
            "document_spans": [
                s.to_dict() if hasattr(s, "to_dict") else s for s in self.document_spans
            ],
        }


def _choose_mode(n_doc_tokens: int, mode: str | None) -> str:
    if mode in {"tokens", "spans"}:
        return mode
    if mode == "auto" or mode is None:
        return "spans" if n_doc_tokens > _LONG_DOC_THRESHOLD else "tokens"
    raise ValueError(f"Unknown contribution mode: {mode!r}")


def explain_result(
    query: str,
    document: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    rank: int = 1,
    score: float | None = None,
    document_embedding: np.ndarray | None = None,
    contribution_mode: str | None = "auto",
    span_window: int = 5,
    span_stride: int | None = None,
) -> ExplainedResult:
    """Explain a single (query, document) match.

    Args:
        query: User query.
        document: Result document.
        encoder: Embedding callable.
        rank: 1-based rank of this result in the original list (purely for
            display — has no effect on the explanation arithmetic).
        score: Pre-computed query-document similarity. Reused as the base
            score for both LOO passes.
        document_embedding: Pre-computed doc embedding. Saves one encode call.
        contribution_mode: ``"tokens"``, ``"spans"``, or ``"auto"`` (default).
            ``"auto"`` switches to span mode for long documents.
        span_window: Span width when contribution_mode resolves to ``"spans"``.
        span_stride: Stride between spans (default = window → non-overlapping).

    Returns:
        :class:`ExplainedResult` ready for serialisation or rendering.
    """
    n_doc = len(tokenize(document))
    mode = _choose_mode(n_doc, contribution_mode)

    # Run the query side first; reuse its base_score everywhere downstream so
    # we never pay for a duplicate query-document encode.
    query_exp: QueryExplanation = explain_query(
        query,
        document,
        encoder,
        document_embedding=document_embedding,
        base_score=score,
    )
    base = query_exp.base_score

    # Encode the query once for the document-side passes (the query LOO
    # already paid this cost internally; we redo it cheaply because the
    # explain_query API doesn't return the query embedding).
    query_emb = _encode(encoder, [query])[0]

    doc_tokens: list[TokenContribution] = []
    doc_spans: list[SpanContribution] = []
    if mode == "tokens":
        doc_tokens = contributing_tokens(
            query,
            document,
            encoder,
            query_embedding=query_emb,
            base_score=base,
        )
    else:
        doc_spans = contributing_spans(
            query,
            document,
            encoder,
            window=span_window,
            stride=span_stride,
            query_embedding=query_emb,
            base_score=base,
        )

    return ExplainedResult(
        rank=rank,
        query=query,
        document=document,
        score=base,
        query_tokens=list(query_exp.tokens),
        document_tokens=doc_tokens,
        document_spans=doc_spans,
        n_query_tokens=len(query_exp.tokens),
        n_document_tokens=n_doc,
        contribution_mode=mode,
    )


def explain_results(
    query: str,
    results: Sequence[tuple[str, float] | str],
    encoder: Callable[[Sequence[str]], np.ndarray],
    **kwargs: Any,
) -> list[ExplainedResult]:
    """Explain a ranked list of results.

    ``results`` accepts either ``(document, score)`` tuples (the engine's
    native format) or plain strings (in which case the score is recomputed).
    """
    out: list[ExplainedResult] = []
    for rank, item in enumerate(results, start=1):
        if isinstance(item, str):
            doc, score = item, None
        else:
            doc, score = item[0], float(item[1])
        out.append(explain_result(query, doc, encoder, rank=rank, score=score, **kwargs))
    return out
