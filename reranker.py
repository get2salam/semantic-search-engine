"""
Cross-Encoder Re-Ranker
========================
Two-stage retrieval pipeline: fast bi-encoder recall + precise cross-encoder
re-ranking. Cross-encoders score (query, document) pairs jointly, producing
significantly more accurate rankings at the cost of higher latency.

Typical pipeline:
    1. Bi-encoder retrieval — fast, retrieves top-N candidates (N=50–200)
    2. Cross-encoder re-ranking — precise, re-scores top-N and returns top-K

Cross-encoder models (sentence-transformers hub):
    - "cross-encoder/ms-marco-MiniLM-L-6-v2"   fast, good quality
    - "cross-encoder/ms-marco-MiniLM-L-12-v2"  slower, higher quality
    - "cross-encoder/stsb-roberta-large"        semantic similarity tasks

Usage:
    from semantic_search import SemanticSearchEngine
    from reranker import CrossEncoderReranker, TwoStageRetriever

    # Standalone re-ranker
    reranker = CrossEncoderReranker()
    candidates = [("doc text here", 0.72), ("another doc", 0.68)]
    reranked = reranker.rerank(query="my query", candidates=candidates, top_k=1)

    # Full two-stage pipeline
    engine = SemanticSearchEngine()
    engine.add_documents([...])
    retriever = TwoStageRetriever(engine)
    results = retriever.search("my query", top_k=5, recall_k=50)

Author: get2salam
License: MIT
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from semantic_search import SemanticSearchEngine

logger = logging.getLogger(__name__)

__all__ = [
    "CrossEncoderReranker",
    "TwoStageRetriever",
    "RerankResult",
    "RetrievalStats",
]


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class RerankResult:
    """A single result from the re-ranking pipeline."""

    document: str
    score: float
    original_rank: int
    reranked_rank: int

    @property
    def rank_change(self) -> int:
        """Positive means the document moved up in ranking."""
        return self.original_rank - self.reranked_rank


@dataclass
class RetrievalStats:
    """Timing and size statistics for a two-stage retrieval call."""

    query: str
    recall_k: int
    final_k: int
    recall_ms: float
    rerank_ms: float
    total_ms: float
    candidates_returned: int

    @property
    def rerank_overhead_pct(self) -> float:
        """Percentage of total time spent in re-ranking."""
        if self.total_ms == 0:
            return 0.0
        return (self.rerank_ms / self.total_ms) * 100

    def __str__(self) -> str:
        return (
            f"RetrievalStats(recall={self.recall_ms:.1f}ms, "
            f"rerank={self.rerank_ms:.1f}ms [{self.rerank_overhead_pct:.0f}%], "
            f"total={self.total_ms:.1f}ms, "
            f"candidates={self.candidates_returned}/{self.recall_k})"
        )


# ---------------------------------------------------------------------------
# Cross-Encoder Re-Ranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """
    Re-ranks candidate documents using a cross-encoder model.

    A cross-encoder takes a (query, document) pair as a single input and
    produces a relevance score. This is more accurate than bi-encoder
    dot-product similarity but O(N) slower (no index).

    Attributes:
        model_name: Name of the cross-encoder model from sentence-transformers hub
        model: Loaded CrossEncoder instance
        max_length: Maximum token length for truncation

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> candidates = [("Python programming tutorial", 0.8), ("Java basics", 0.75)]
        >>> results = reranker.rerank("learn Python", candidates, top_k=1)
        >>> print(results[0].document)
        "Python programming tutorial"
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
        device: str | None = None,
    ):
        """
        Initialize the cross-encoder re-ranker.

        Args:
            model_name: HuggingFace model name for the cross-encoder.
                        Must be compatible with sentence-transformers CrossEncoder.
            max_length: Maximum sequence length for tokenisation.
            device: Torch device ("cpu", "cuda", "mps"). Auto-detects if None.
        """
        self.model_name = model_name
        self.max_length = max_length

        logger.info("Loading cross-encoder: %s", model_name)
        try:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {"max_length": max_length}
            if device is not None:
                kwargs["device"] = device
            self.model = CrossEncoder(model_name, **kwargs)
            logger.info("Cross-encoder loaded")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        top_k: int | None = None,
        batch_size: int = 32,
    ) -> list[RerankResult]:
        """
        Re-rank a list of (document, score) candidates for the given query.

        Args:
            query: The search query.
            candidates: List of (document_text, initial_score) from bi-encoder.
            top_k: Number of results to return. Returns all if None.
            batch_size: Batch size for cross-encoder inference.

        Returns:
            List of RerankResult sorted by cross-encoder score (descending).
        """
        if not candidates:
            return []

        k = top_k if top_k is not None else len(candidates)

        # Build (query, document) pairs for the cross-encoder
        pairs = [[query, doc] for doc, _ in candidates]

        # Score all pairs
        scores: list[float] = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        ).tolist()

        # Build results with original ranks
        results = [
            RerankResult(
                document=doc,
                score=float(score),
                original_rank=i + 1,
                reranked_rank=0,  # assigned below
            )
            for i, ((doc, _), score) in enumerate(zip(candidates, scores, strict=False))
        ]

        # Sort by cross-encoder score (descending)
        results.sort(key=lambda r: r.score, reverse=True)

        # Assign final ranks
        for rank, result in enumerate(results, start=1):
            result.reranked_rank = rank

        return results[:k]

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.

        Args:
            query: The search query.
            document: The document text.

        Returns:
            Relevance score (typically a logit or probability, model-dependent).
        """
        scores = self.model.predict([[query, document]], show_progress_bar=False)
        return float(scores[0])

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model='{self.model_name}', max_length={self.max_length})"


# ---------------------------------------------------------------------------
# Two-Stage Retriever
# ---------------------------------------------------------------------------


class TwoStageRetriever:
    """
    Full two-stage retrieval pipeline combining bi-encoder + cross-encoder.

    Stage 1 (Recall): Bi-encoder retrieves ``recall_k`` candidates fast.
    Stage 2 (Precision): Cross-encoder re-ranks candidates and returns top-K.

    This balances the speed of approximate retrieval with the accuracy of
    pairwise relevance scoring.

    Attributes:
        engine: Bi-encoder semantic search engine.
        reranker: Cross-encoder re-ranker.
        default_recall_k: Default number of candidates for stage 1.
        collect_stats: If True, stats from the last search are stored in
                       ``last_stats``.

    Example:
        >>> engine = SemanticSearchEngine()
        >>> engine.add_documents(["doc 1", "doc 2", ...])
        >>> retriever = TwoStageRetriever(engine)
        >>> results = retriever.search("my query", top_k=5)
        >>> for r in results:
        ...     print(f"[{r.score:.3f}] {r.document[:60]}")
    """

    def __init__(
        self,
        engine: SemanticSearchEngine,
        reranker: CrossEncoderReranker | None = None,
        default_recall_k: int = 50,
        collect_stats: bool = True,
    ):
        """
        Initialize the two-stage retriever.

        Args:
            engine: A SemanticSearchEngine instance with documents already indexed.
            reranker: A CrossEncoderReranker instance. Loads the default model if None.
            default_recall_k: How many candidates to retrieve in stage 1.
            collect_stats: Whether to store timing stats after each search.
        """
        self.engine = engine
        self.reranker = reranker if reranker is not None else CrossEncoderReranker()
        self.default_recall_k = default_recall_k
        self.collect_stats = collect_stats
        self.last_stats: RetrievalStats | None = None

    def search(
        self,
        query: str,
        top_k: int = 5,
        recall_k: int | None = None,
        threshold: float | None = None,
    ) -> list[RerankResult]:
        """
        Execute a two-stage search and return re-ranked results.

        Args:
            query: The search query text.
            top_k: Number of final results to return.
            recall_k: Override the number of stage-1 candidates.
                      Defaults to ``default_recall_k`` or 3× top_k, whichever is larger.
            threshold: Optional minimum cross-encoder score filter.

        Returns:
            List of RerankResult sorted by cross-encoder relevance.
        """
        n_docs = len(self.engine)
        if n_docs == 0:
            logger.warning("No documents indexed in the engine")
            return []

        # Determine how many candidates to fetch in stage 1
        effective_recall_k = recall_k or max(self.default_recall_k, top_k * 3)
        effective_recall_k = min(effective_recall_k, n_docs)

        # Stage 1 — bi-encoder retrieval
        t0 = time.perf_counter()
        candidates = self.engine.search(query, top_k=effective_recall_k)
        recall_ms = (time.perf_counter() - t0) * 1000

        if not candidates:
            return []

        # Stage 2 — cross-encoder re-ranking
        t1 = time.perf_counter()
        results = self.reranker.rerank(query, candidates, top_k=top_k)
        rerank_ms = (time.perf_counter() - t1) * 1000

        total_ms = recall_ms + rerank_ms

        # Optional threshold filter
        if threshold is not None:
            results = [r for r in results if r.score >= threshold]

        # Store stats
        if self.collect_stats:
            self.last_stats = RetrievalStats(
                query=query,
                recall_k=effective_recall_k,
                final_k=top_k,
                recall_ms=recall_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
                candidates_returned=len(candidates),
            )
            logger.debug("Search stats: %s", self.last_stats)

        return results

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        recall_k: int | None = None,
    ) -> list[list[RerankResult]]:
        """
        Run two-stage search for multiple queries.

        Args:
            queries: List of query strings.
            top_k: Number of results per query.
            recall_k: Number of stage-1 candidates per query.

        Returns:
            List of result lists, one per query.
        """
        return [self.search(q, top_k=top_k, recall_k=recall_k) for q in queries]

    def compare_ranking(
        self,
        query: str,
        top_k: int = 5,
        recall_k: int | None = None,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Compare bi-encoder vs cross-encoder rankings for a query.

        Useful for debugging and understanding reranking behaviour.

        Args:
            query: The search query.
            top_k: Number of results to compare.
            recall_k: Number of initial candidates.

        Returns:
            Dict with "biencoder" and "crossencoder" keys, each containing
            a list of (document_snippet, score) tuples.
        """
        n_docs = len(self.engine)
        if n_docs == 0:
            return {"biencoder": [], "crossencoder": []}

        effective_recall_k = recall_k or max(self.default_recall_k, top_k * 3)
        effective_recall_k = min(effective_recall_k, n_docs)

        candidates = self.engine.search(query, top_k=effective_recall_k)
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        biencoder = [(doc[:80], score) for doc, score in candidates[:top_k]]
        crossencoder = [(r.document[:80], r.score) for r in reranked]

        return {"biencoder": biencoder, "crossencoder": crossencoder}

    def __repr__(self) -> str:
        return (
            f"TwoStageRetriever("
            f"engine={self.engine!r}, "
            f"reranker={self.reranker!r}, "
            f"default_recall_k={self.default_recall_k})"
        )
