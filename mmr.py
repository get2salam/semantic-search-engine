"""
Maximal Marginal Relevance (MMR) Diversification
================================================
Re-ranks an initial candidate set to balance relevance with diversity, reducing
near-duplicate results that are common when retrieving from dense sentence
embeddings.

The classic MMR formulation (Carbonell & Goldstein, SIGIR 1998) selects
documents iteratively:

    MMR = argmax_{d ∈ R \\ S}  [ λ · sim(q, d)  −  (1 − λ) · max_{s ∈ S} sim(d, s) ]

where ``R`` is the candidate pool, ``S`` is the set of already-selected
documents, and ``λ ∈ [0, 1]`` trades off relevance (``λ → 1``) against
diversity (``λ → 0``).

Because MMR needs pair-wise document similarities in addition to query-document
scores, it is applied on top of the bi-encoder's L2-normalised embedding
matrix: doc-doc similarity is then a plain dot product.

Usage::

    from semantic_search import SemanticSearchEngine

    engine = SemanticSearchEngine()
    engine.add_documents([...])

    # diversified top-5 (re-ranks a 25-candidate pool)
    results = engine.search("climate change", top_k=5, mmr_lambda=0.5)

Reference:
    Carbonell, J. & Goldstein, J. (1998).
    "The use of MMR, diversity-based reranking for reordering documents and
    producing summaries." SIGIR '98.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import numpy as np

__all__ = ["mmr_select", "mmr_rerank"]


def mmr_select(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """
    Run the MMR selection loop and return the chosen candidate indices.

    The function expects L2-normalised embeddings; with unit vectors, dot
    products equal cosine similarity, which keeps the λ/(1−λ) convex
    combination on a comparable scale. Non-normalised inputs are accepted but
    will shift the relevance/diversity balance in ways the caller must account
    for.

    Args:
        query_embedding: 1-D array of shape ``(d,)`` or 2-D ``(1, d)``.
        candidate_embeddings: 2-D array of shape ``(n, d)`` — one row per
            candidate, ordered by the upstream retriever.
        top_k: Number of documents to select. Clamped to ``n``.
        lambda_mult: Relevance weight in ``[0, 1]``.
            * ``1.0`` → behaves identically to pure relevance ranking.
            * ``0.0`` → maximises diversity, ignoring the query after the
              first pick.
            * ``0.5`` (default) → balanced.

    Returns:
        List of selected indices (into ``candidate_embeddings``) in selection
        order. Always returns at most ``top_k`` indices; may be shorter when
        the candidate pool is smaller.

    Raises:
        ValueError: If ``lambda_mult`` is outside ``[0, 1]`` or if inputs have
            mismatched dimensionalities.
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError(f"lambda_mult must be in [0, 1], got {lambda_mult}")

    if candidate_embeddings.ndim != 2:
        raise ValueError(
            f"candidate_embeddings must be 2-D, got shape {candidate_embeddings.shape}"
        )

    n_candidates = candidate_embeddings.shape[0]
    if n_candidates == 0 or top_k <= 0:
        return []

    query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if query_vec.shape[0] != candidate_embeddings.shape[1]:
        raise ValueError(
            f"query dim {query_vec.shape[0]} != candidate dim {candidate_embeddings.shape[1]}"
        )

    cand = candidate_embeddings.astype(np.float32, copy=False)
    relevance = cand @ query_vec  # shape (n,)

    k = min(top_k, n_candidates)
    selected: list[int] = []
    # Running max similarity to any already-selected doc; −inf so the first
    # pick is driven purely by relevance.
    max_sim_to_selected = np.full(n_candidates, -np.inf, dtype=np.float32)

    # First pick: highest-relevance candidate (diversity term is −inf·(1−λ) =
    # 0 by convention for an empty selection set).
    first = int(np.argmax(relevance))
    selected.append(first)

    for _ in range(1, k):
        # Update running max with similarity to the most recently added doc.
        last_vec = cand[selected[-1]]
        sim_to_last = cand @ last_vec
        np.maximum(max_sim_to_selected, sim_to_last, out=max_sim_to_selected)

        mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * max_sim_to_selected
        # Mask already-selected candidates so they can't be re-picked.
        mmr_score[selected] = -np.inf

        nxt = int(np.argmax(mmr_score))
        # Safety: all remaining scores are −inf (e.g. ties exhaust the pool).
        if not np.isfinite(mmr_score[nxt]):
            break
        selected.append(nxt)

    return selected


def mmr_rerank(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_scores: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5,
) -> tuple[list[int], list[float]]:
    """
    MMR-rerank a candidate pool and return ``(indices, relevance_scores)``.

    Relevance scores are the **original** query-doc similarities (not the MMR
    objective), so they stay interpretable to callers used to cosine
    similarity. MMR only controls *which* docs are returned and in which order.

    Args:
        query_embedding: Query vector, ``(d,)`` or ``(1, d)``.
        candidate_embeddings: Candidate embedding matrix, ``(n, d)``.
        candidate_scores: Per-candidate relevance scores from the retriever.
            Must align row-wise with ``candidate_embeddings``.
        top_k: Number of results to return.
        lambda_mult: Relevance/diversity trade-off in ``[0, 1]``.

    Returns:
        Tuple ``(indices, scores)`` where ``indices`` are MMR-selected row
        indices (length ≤ ``top_k``) and ``scores`` are the corresponding
        entries from ``candidate_scores``.
    """
    indices = mmr_select(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        top_k=top_k,
        lambda_mult=lambda_mult,
    )
    scores = [float(candidate_scores[i]) for i in indices]
    return indices, scores
