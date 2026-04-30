"""
Document Contribution Analysis
==============================
Ranks the tokens of a *result document* by how much each contributes to the
query-document similarity score.

The query side is covered by ``explain.py`` (leave-one-out on the query).
This module mirrors the same idea on the *document* side: drop each document
token, re-encode, and observe the score drop. Together they let users see
both halves of the match — "which words in my query mattered" and "which
words in this result earned the match".

For long documents the per-token cost adds up, so the API exposes both:

    * ``contributing_tokens`` — full LOO over every token (best fidelity)
    * ``contributing_spans``  — LOO over fixed-width windows (cheaper, coarser)

Window mode is the default for documents above ~30 tokens because per-token
LOO scales linearly with document length and a 5-token window typically
preserves the top contributing region while cutting cost ~5×.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np

from explain import _cosine, _encode, tokenize

__all__ = [
    "TokenContribution",
    "SpanContribution",
    "contributing_tokens",
    "contributing_spans",
]


@dataclass(frozen=True)
class TokenContribution:
    """One document token's contribution to the match score."""

    token: str
    position: int
    contribution: float
    score_with: float
    score_without: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpanContribution:
    """A contiguous span of document tokens treated as one unit."""

    span: str
    start: int
    end: int  # exclusive
    contribution: float
    score_with: float
    score_without: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_perturbations(tokens: list[str], drops: list[tuple[int, int]]) -> list[str]:
    """Return one perturbed text per (start, end) drop, joined by spaces."""
    perturbed: list[str] = []
    for start, end in drops:
        kept = tokens[:start] + tokens[end:]
        perturbed.append(" ".join(kept))
    return perturbed


def contributing_tokens(
    query: str,
    document: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    query_embedding: np.ndarray | None = None,
    base_score: float | None = None,
) -> list[TokenContribution]:
    """Rank every token in ``document`` by its contribution to sim(query, doc).

    Args:
        query: The original query (unchanged across perturbations).
        document: The result document whose tokens we attribute.
        encoder: Callable ``list[str] → np.ndarray`` (n, d).
        query_embedding: Pre-computed query embedding. Skips one encode call
            when callers already have it (typical inside the search loop).
        base_score: Pre-computed base similarity. Skips a second encode call.

    Returns:
        One :class:`TokenContribution` per document token, in original order.
    """
    tokens = tokenize(document)
    if not tokens:
        return []

    drops = [(i, i + 1) for i in range(len(tokens))]
    perturbed = _build_perturbations(tokens, drops)

    to_encode: list[str] = list(perturbed)
    need_query = query_embedding is None
    need_base = base_score is None

    if need_query:
        to_encode.append(query)
    if need_base or query_embedding is None:
        to_encode.append(document)

    embeddings = _encode(encoder, to_encode)

    perturbed_emb = embeddings[: len(perturbed)]
    cursor = len(perturbed)
    if need_query:
        q_emb = embeddings[cursor]
        cursor += 1
    else:
        q_emb = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

    if need_base:
        doc_emb = embeddings[cursor]
        base = _cosine(q_emb, doc_emb)
    else:
        base = float(base_score)

    out: list[TokenContribution] = []
    for i, token in enumerate(tokens):
        s_without = _cosine(q_emb, perturbed_emb[i])
        out.append(
            TokenContribution(
                token=token,
                position=i,
                contribution=base - s_without,
                score_with=base,
                score_without=s_without,
            )
        )
    return out


def contributing_spans(
    query: str,
    document: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    window: int = 5,
    stride: int | None = None,
    query_embedding: np.ndarray | None = None,
    base_score: float | None = None,
) -> list[SpanContribution]:
    """Coarse-grained contribution analysis over fixed-width spans.

    Useful for long documents where per-token LOO is wasteful — a 5-token
    window typically isolates phrases (not single words) which is the level
    most users want to inspect anyway.

    Args:
        window: Span width in tokens. Must be ≥ 1.
        stride: Step between window starts. Defaults to ``window`` (non-
            overlapping). Use a smaller stride for finer-grained heatmaps at
            extra cost.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if stride is None:
        stride = window
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    tokens = tokenize(document)
    if not tokens:
        return []

    drops: list[tuple[int, int]] = []
    for start in range(0, len(tokens), stride):
        end = min(start + window, len(tokens))
        drops.append((start, end))
        if end == len(tokens):
            break
    perturbed = _build_perturbations(tokens, drops)

    to_encode: list[str] = list(perturbed)
    need_query = query_embedding is None
    need_base = base_score is None
    if need_query:
        to_encode.append(query)
    if need_base or query_embedding is None:
        to_encode.append(document)

    embeddings = _encode(encoder, to_encode)
    perturbed_emb = embeddings[: len(perturbed)]
    cursor = len(perturbed)
    if need_query:
        q_emb = embeddings[cursor]
        cursor += 1
    else:
        q_emb = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if need_base:
        doc_emb = embeddings[cursor]
        base = _cosine(q_emb, doc_emb)
    else:
        base = float(base_score)

    out: list[SpanContribution] = []
    for (start, end), emb in zip(drops, perturbed_emb, strict=True):
        s_without = _cosine(q_emb, emb)
        out.append(
            SpanContribution(
                span=" ".join(tokens[start:end]),
                start=start,
                end=end,
                contribution=base - s_without,
                score_with=base,
                score_without=s_without,
            )
        )
    return out
