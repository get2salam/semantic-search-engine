"""
Search Explainability
=====================
Black-box explanations for semantic search results.

The bi-encoder architecture used by the engine produces a single dense vector
per query — there are no attention weights or word-level scores to expose
directly. This module reconstructs *post-hoc* explanations by perturbing the
query and measuring the change in similarity on the target document:

    importance(token_i)  =  sim(q, d)  −  sim(q \\ {token_i}, d)

Tokens whose removal collapses the similarity were "carrying" the match.
Tokens whose removal *increases* it were noise from the encoder's perspective.

This is the same family of methods as LIME / occlusion analysis: model-agnostic,
deterministic, and trivially interpretable. The trade-off is cost — one
re-encoding per token. For typical query lengths (< 20 tokens) this is
sub-second on CPU and acceptable as an on-demand diagnostic.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "TokenAttribution",
    "QueryExplanation",
    "tokenize",
    "explain_query",
]


# Word tokenizer that preserves alphanumerics and apostrophes (don't, it's, …).
# Keeps explanations readable; we never feed these tokens back to the encoder
# directly — we always reconstruct the perturbed query as a whitespace-joined
# string of *kept* tokens, so the encoder still sees natural text.
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


@dataclass(frozen=True)
class TokenAttribution:
    """Single token's contribution to a query-document similarity score."""

    token: str
    position: int
    importance: float
    score_with: float
    score_without: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryExplanation:
    """Full explanation: per-token attributions plus metadata."""

    query: str
    document: str
    base_score: float
    tokens: list[TokenAttribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "document": self.document,
            "base_score": self.base_score,
            "tokens": [t.to_dict() for t in self.tokens],
        }

    def top_positive(self, n: int = 3) -> list[TokenAttribution]:
        """Tokens whose presence helps the match the most (largest drop on removal)."""
        return sorted(self.tokens, key=lambda t: t.importance, reverse=True)[:n]

    def top_negative(self, n: int = 3) -> list[TokenAttribution]:
        """Tokens that *hurt* the match (removal would raise the score)."""
        return sorted(self.tokens, key=lambda t: t.importance)[:n]


def tokenize(text: str) -> list[str]:
    """Word-level tokenisation used by the explainer.

    Public so callers can preview tokenisation before paying for an
    explanation pass.
    """
    return _WORD_RE.findall(text)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Falls back to dot product if
    inputs are already L2-normalised (the common case for this codebase)."""
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _encode(encoder: Callable[[Sequence[str]], np.ndarray], texts: Sequence[str]) -> np.ndarray:
    """Call the encoder and coerce the result to a 2-D float array."""
    arr = np.asarray(encoder(list(texts)), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def explain_query(
    query: str,
    document: str,
    encoder: Callable[[Sequence[str]], np.ndarray],
    *,
    document_embedding: np.ndarray | None = None,
    base_score: float | None = None,
) -> QueryExplanation:
    """Compute leave-one-out token importances for ``query`` against ``document``.

    Args:
        query: The user's original query.
        document: The result document being explained.
        encoder: Any callable that maps ``list[str] → np.ndarray`` of shape
            ``(n, d)``. Must produce embeddings comparable by cosine similarity.
            For this engine, ``engine.model.encode`` works directly.
        document_embedding: Optional pre-computed document embedding. Saves one
            encoder call when the doc is already in the index.
        base_score: Optional pre-computed query-document similarity. Saves one
            encoder call.

    Returns:
        A ``QueryExplanation`` with one ``TokenAttribution`` per query token.
        Tokens are returned in original order; sort with ``top_positive`` /
        ``top_negative`` for ranked views.
    """
    tokens = tokenize(query)
    if not tokens:
        return QueryExplanation(query=query, document=document, base_score=0.0, tokens=[])

    # Build all perturbations in one batch so the encoder amortises overhead.
    perturbed = [" ".join(tokens[:i] + tokens[i + 1 :]) for i in range(len(tokens))]
    to_encode: list[str] = list(perturbed)

    # Append the original query / document only when we don't already have them.
    need_query = base_score is None
    need_doc = document_embedding is None
    if need_query:
        to_encode.append(query)
    if need_doc:
        to_encode.append(document)

    embeddings = _encode(encoder, to_encode)
    perturbed_emb = embeddings[: len(perturbed)]

    cursor = len(perturbed)
    if need_query:
        query_emb = embeddings[cursor]
        cursor += 1
    else:
        query_emb = None  # not used; we have base_score directly
    if need_doc:
        doc_emb = embeddings[cursor]
    else:
        doc_emb = np.asarray(document_embedding, dtype=np.float32).reshape(-1)

    if base_score is None:
        base = _cosine(query_emb, doc_emb)
    else:
        base = float(base_score)

    attributions: list[TokenAttribution] = []
    for i, token in enumerate(tokens):
        score_without = _cosine(perturbed_emb[i], doc_emb)
        attributions.append(
            TokenAttribution(
                token=token,
                position=i,
                importance=base - score_without,
                score_with=base,
                score_without=score_without,
            )
        )

    return QueryExplanation(
        query=query,
        document=document,
        base_score=base,
        tokens=attributions,
    )
