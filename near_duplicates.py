"""
Near-Duplicate Detection
========================
Embedding-based near-duplicate clustering for corpus auditing.

Whereas :func:`corpus_profile.exact_duplicate_report` only catches
*literal* repetition (after whitespace + case normalisation), this module
catches *paraphrases*: documents whose embeddings sit above a cosine
threshold of one another. That covers boilerplate variations, scraped
content with minor edits, and machine-translated near-twins — all of
which inflate the index without adding signal.

The clustering is a union-find sweep over a single similarity-matrix
pass. That's O(n²) memory which is fine for the small / medium corpora
that audits typically run on; for very large indices, swap the backend
to FAISS-Range without changing the report contract.

The functions accept pre-computed embeddings rather than running the
encoder themselves. That keeps the module composable with any embedding
backend in the codebase (dense, hybrid, fine-tuned).

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "NearDuplicateGroup",
    "NearDuplicateReport",
    "near_duplicate_report",
]


@dataclass(frozen=True)
class NearDuplicateGroup:
    """A near-duplicate cluster of corpus indices.

    Attributes:
        representative: Index of the canonical document (the one most
            similar to the cluster centroid — i.e., the medoid).
        indices: All corpus indices in the cluster, including the
            representative. Sorted ascending so reports are deterministic.
        mean_similarity: Mean off-diagonal cosine similarity within the
            cluster. Always ≥ ``threshold`` by construction.
        size: Cluster size. Always ≥ 2 for groups in the report.
    """

    representative: int
    indices: list[int]
    mean_similarity: float
    size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NearDuplicateReport:
    """Aggregate near-duplicate stats for a corpus."""

    n_documents: int
    n_groups: int
    n_duplicate_documents: int
    threshold: float
    duplication_ratio: float
    groups: list[NearDuplicateGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["groups"] = [g.to_dict() for g in self.groups]
        return d


class _UnionFind:
    """Tiny union-find with path compression for clustering by threshold."""

    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        # Iterative path compression — recursion blows the stack on long chains.
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def near_duplicate_report(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.92,
    max_groups: int = 25,
) -> NearDuplicateReport:
    """Cluster embeddings whose pairwise cosine ≥ ``threshold``.

    Args:
        embeddings: ``(n, d)`` float matrix. Assumed L2-normalised — the
            engine normalises by default — so dot products equal cosines.
            Non-normalised inputs are accepted and renormalised in place
            for safety.
        threshold: Minimum cosine for two docs to be linked. The default
            ``0.92`` is a sentence-transformer rule of thumb that catches
            paraphrases without flagging merely related sentences.
        max_groups: Cap on the number of groups returned in the report.
            Aggregate counts always reflect the whole corpus.

    Returns:
        :class:`NearDuplicateReport` with cluster counts and the largest
        clusters (representative is the medoid of each cluster).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")

    n = embeddings.shape[0]
    if n == 0:
        return NearDuplicateReport(0, 0, 0, threshold, 0.0)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1]; got {threshold}")

    arr = embeddings.astype(np.float64, copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    normed = arr / safe

    sim = normed @ normed.T
    # Mask the diagonal so a doc isn't flagged as its own duplicate.
    np.fill_diagonal(sim, -1.0)

    uf = _UnionFind(n)
    # Iterate the upper triangle only — sim is symmetric, save half the work.
    rows, cols = np.where(np.triu(sim >= threshold, k=1))
    for i, j in zip(rows.tolist(), cols.tolist(), strict=False):
        uf.union(i, j)

    # Bucket indices by cluster root.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    groups: list[NearDuplicateGroup] = []
    n_dup_docs = 0
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        n_dup_docs += len(indices) - 1
        sub = normed[indices]
        # Mean off-diagonal similarity within the cluster.
        sub_sim = sub @ sub.T
        size = len(indices)
        off = float(sub_sim.sum() - np.trace(sub_sim))
        mean_sim = off / (size * (size - 1)) if size > 1 else 1.0
        # Medoid: the row with the highest mean similarity to the rest.
        per_doc = (sub_sim.sum(axis=1) - 1.0) / (size - 1)
        medoid_local = int(np.argmax(per_doc))
        groups.append(
            NearDuplicateGroup(
                representative=indices[medoid_local],
                indices=sorted(indices),
                mean_similarity=mean_sim,
                size=size,
            )
        )

    groups.sort(key=lambda g: (-g.size, -g.mean_similarity))

    return NearDuplicateReport(
        n_documents=n,
        n_groups=len(groups),
        n_duplicate_documents=n_dup_docs,
        threshold=threshold,
        duplication_ratio=n_dup_docs / n,
        groups=groups[:max_groups],
    )
