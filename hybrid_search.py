"""
Hybrid Search Engine
====================
Combines BM25 sparse retrieval with dense semantic embeddings using
Reciprocal Rank Fusion (RRF) for improved recall and precision.

BM25 excels at exact-match and keyword relevance; dense embeddings handle
semantic similarity.  Fusing both signals with RRF consistently outperforms
either method alone across real-world IR benchmarks.

Architecture::

    query
      ↓
    BM25Retriever ──────────┐
                             ├──► RecipRankFusion ──► ranked results
    DenseRetriever ─────────┘

Usage::

    from hybrid_search import HybridSearchEngine

    engine = HybridSearchEngine()
    engine.index(["Machine learning is powerful", "Python for data science"])
    results = engine.search("AI and ML techniques", top_k=5)
    for doc, score in results:
        print(f"[{score:.4f}] {doc}")

References:
    - Robertson et al. (1994) – Okapi BM25
    - Cormack, Clarke & Buettcher (2009) – Reciprocal Rank Fusion

Author: get2salam
License: MIT
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridSearchEngine",
    "HybridResult",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase-split *text* on non-alphanumeric boundaries."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    Okapi BM25 sparse retriever.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter (default ``1.5``).
    b:
        Document-length normalisation parameter (default ``0.75``).

    Notes
    -----
    IDF uses the standard smoothed formulation:

    .. code-block:: text

        IDF(t) = log((N - df_t + 0.5) / (df_t + 0.5) + 1)

    This always stays positive, making BM25 scores non-negative.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        self._corpus: list[str] = []
        self._tokenized: list[list[str]] = []
        self._avg_dl: float = 0.0
        self._idf: dict[str, float] = {}
        self._tf: list[Counter] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: list[str]) -> None:
        """
        Index *documents* for BM25 retrieval.

        Replaces any previously indexed content.

        Parameters
        ----------
        documents:
            Corpus to index. Documents are tokenised by whitespace /
            punctuation boundaries.
        """
        self._corpus = list(documents)
        self._tokenized = [_tokenize(d) for d in documents]
        self._tf = [Counter(tokens) for tokens in self._tokenized]

        n = len(documents)
        total_len = sum(len(t) for t in self._tokenized)
        self._avg_dl = total_len / n if n > 0 else 1.0

        # IDF for every unique term
        df: dict[str, int] = {}
        for tokens in self._tokenized:
            for term in set(tokens):
                df[term] = df.get(term, 0) + 1

        self._idf = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1.0) for term, freq in df.items()
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, query: str) -> np.ndarray:
        """
        Compute BM25 scores for *query* against the indexed corpus.

        Parameters
        ----------
        query:
            Raw query string.

        Returns
        -------
        np.ndarray
            1-D array of BM25 scores, one per document.
        """
        if not self._corpus:
            return np.array([])

        q_tokens = _tokenize(query)
        scores = np.zeros(len(self._corpus))

        for term in q_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue

            for i, (tf, tokens) in enumerate(zip(self._tf, self._tokenized, strict=False)):
                dl = len(tokens)
                freq = tf.get(term, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
                scores[i] += idf * numerator / denominator

        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return the *top_k* documents most relevant to *query*.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[str, float]]
            ``(document, bm25_score)`` pairs sorted by descending score.
        """
        scores = self.score(query)
        if scores.size == 0:
            return []

        k = min(top_k, len(self._corpus))
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self._corpus[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._corpus)

    def __repr__(self) -> str:
        return f"BM25Retriever(k1={self.k1}, b={self.b}, docs={len(self)})"


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """
    Dense semantic retriever backed by sentence-transformers.

    Parameters
    ----------
    model_name:
        Name of the sentence-transformer model.
    normalize:
        L2-normalise embeddings (enables cosine similarity via dot product).
    batch_size:
        Encoding batch size.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        normalize: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._corpus: list[str] = []
        self._embeddings: np.ndarray | None = None

    # ------------------------------------------------------------------

    def index(self, documents: list[str]) -> None:
        """
        Encode and index *documents*.

        Replaces any previously indexed content.
        """
        self._corpus = list(documents)
        if documents:
            self._embeddings = self._model.encode(
                documents,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        else:
            self._embeddings = None

    # ------------------------------------------------------------------

    def score(self, query: str) -> np.ndarray:
        """
        Return cosine-similarity scores for *query* vs the indexed corpus.

        Returns
        -------
        np.ndarray
            1-D array of scores in ``[-1, 1]``.
        """
        if self._embeddings is None or len(self._corpus) == 0:
            return np.array([])

        q_emb = self._model.encode(
            query,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        ).reshape(1, -1)

        return np.dot(self._embeddings, q_emb.T).flatten()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Return the *top_k* documents most similar to *query*.

        Returns
        -------
        list[tuple[str, float]]
            ``(document, cosine_score)`` pairs sorted by descending score.
        """
        scores = self.score(query)
        if scores.size == 0:
            return []

        k = min(top_k, len(self._corpus))
        top_indices = np.argsort(scores)[::-1][:k]

        return [(self._corpus[i], float(scores[i])) for i in top_indices]

    def __len__(self) -> int:
        return len(self._corpus)

    def __repr__(self) -> str:
        return f"DenseRetriever(model='{self.model_name}', docs={len(self)})"


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document *d*:

    .. code-block:: text

        RRF(d) = Σ_i  1 / (k + rank_i(d))

    where *rank_i(d)* is the 1-based position of *d* in the *i*-th list.

    Documents not present in a list contribute nothing from that list.

    Parameters
    ----------
    ranked_lists:
        Each inner list is an ordered sequence of document strings
        (most relevant first).
    rrf_k:
        Smoothing constant that reduces the influence of high-ranked
        documents. ``60`` is the commonly recommended default.

    Returns
    -------
    list[tuple[str, float]]
        ``(document, rrf_score)`` pairs sorted by descending RRF score.

    References
    ----------
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    *Reciprocal rank fusion outperforms condorcet and individual rank
    learning methods.*  SIGIR '09, pp. 758-759.
    """
    scores: dict[str, float] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class HybridResult:
    """A single result from :class:`HybridSearchEngine`."""

    document: str
    """The matched document text."""

    rrf_score: float
    """Fused RRF score (higher = more relevant)."""

    bm25_rank: int | None
    """Rank in the BM25 result list (1-based), or ``None`` if absent."""

    dense_rank: int | None
    """Rank in the dense result list (1-based), or ``None`` if absent."""

    def __repr__(self) -> str:
        return (
            f"HybridResult("
            f"rrf={self.rrf_score:.4f}, "
            f"bm25_rank={self.bm25_rank}, "
            f"dense_rank={self.dense_rank}, "
            f"doc={self.document[:60]!r})"
        )


# ---------------------------------------------------------------------------
# HybridSearchEngine
# ---------------------------------------------------------------------------


class HybridSearchEngine:
    """
    Hybrid search engine that fuses BM25 and dense retrieval.

    Parameters
    ----------
    model_name:
        Sentence-transformer model for dense retrieval.
    bm25_k1:
        BM25 term-frequency saturation parameter.
    bm25_b:
        BM25 document-length normalisation parameter.
    rrf_k:
        RRF smoothing constant (default ``60``).
    candidate_k:
        Number of candidates retrieved from each sub-retriever before
        fusion (default ``50``).  Increase for higher recall at the cost
        of slightly more computation.

    Examples
    --------
    >>> engine = HybridSearchEngine()
    >>> engine.index(["deep learning is powerful", "python is easy to learn"])
    >>> results = engine.search("neural network training", top_k=1)
    >>> results[0].document
    'deep learning is powerful'
    """

    def __init__(
        self,
        model_name: str = DenseRetriever.DEFAULT_MODEL,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        candidate_k: int = 50,
    ) -> None:
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        self._bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self._dense = DenseRetriever(model_name=model_name)
        self._corpus: list[str] = []

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, documents: list[str]) -> None:
        """
        Index *documents* in both retrievers.

        Parameters
        ----------
        documents:
            Corpus to index.
        """
        self._corpus = list(documents)
        self._bm25.index(documents)
        self._dense.index(documents)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[HybridResult]:
        """
        Search for *query* using hybrid BM25 + dense retrieval.

        Both retrievers return up to ``candidate_k`` results.  Their ranked
        lists are fused with Reciprocal Rank Fusion, and the top *top_k*
        fused results are returned.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Number of results to return.

        Returns
        -------
        list[HybridResult]
            Results sorted by descending RRF score.
        """
        if not self._corpus:
            return []

        k = min(self.candidate_k, len(self._corpus))

        # Retrieve from each sub-retriever
        bm25_hits = self._bm25.search(query, top_k=k)
        dense_hits = self._dense.search(query, top_k=k)

        bm25_ranked = [doc for doc, _ in bm25_hits]
        dense_ranked = [doc for doc, _ in dense_hits]

        # Build rank lookup tables (1-based)
        bm25_rank_map = {doc: i + 1 for i, doc in enumerate(bm25_ranked)}
        dense_rank_map = {doc: i + 1 for i, doc in enumerate(dense_ranked)}

        # Fuse
        fused = reciprocal_rank_fusion(
            [bm25_ranked, dense_ranked],
            rrf_k=self.rrf_k,
        )

        results = []
        for doc, rrf_score in fused[:top_k]:
            results.append(
                HybridResult(
                    document=doc,
                    rrf_score=rrf_score,
                    bm25_rank=bm25_rank_map.get(doc),
                    dense_rank=dense_rank_map.get(doc),
                )
            )

        return results

    def search_simple(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Simplified search returning ``(document, rrf_score)`` tuples.

        This is a convenience wrapper compatible with the
        :class:`~evaluation.RetrievalEvaluator` interface.

        Parameters
        ----------
        query:
            Raw query string.
        top_k:
            Number of results to return.
        """
        return [(r.document, r.rrf_score) for r in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._corpus)

    def __repr__(self) -> str:
        return (
            f"HybridSearchEngine("
            f"model='{self._dense.model_name}', "
            f"rrf_k={self.rrf_k}, "
            f"docs={len(self)})"
        )
