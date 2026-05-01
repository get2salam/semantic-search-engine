"""
RAG Readiness Audit Report
==========================
Aggregates the corpus and query-coverage signals from
:mod:`corpus_profile`, :mod:`near_duplicates` and :mod:`query_coverage`
into a single ``RagReadinessReport`` plus serialisers.

The point of the aggregate is *narrative*: a client browsing the JSON
or Markdown should be able to answer three questions in under a minute:

    1. Is the corpus healthy enough to embed? (length, vocab, dup ratio)
    2. Is the embedding space well-behaved? (centroid, hubness, eff-rank)
    3. Will real queries get answered? (coverage, confidence)

The aggregator is intentionally *just* a container + renderer — every
underlying analysis stays independently runnable so callers can pick
just the bits they need from a notebook.

Author: get2salam
License: MIT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from corpus_profile import DuplicateReport, LengthStats, VocabularyStats
from embedding_stats import EmbeddingStats
from near_duplicates import NearDuplicateReport
from query_coverage import QueryCoverageReport

__all__ = [
    "RagReadinessReport",
    "build_report",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RagReadinessReport:
    """Bundle of the four audit signals plus optional metadata."""

    generated_at: str
    n_documents: int
    length: LengthStats
    vocabulary: VocabularyStats
    exact_duplicates: DuplicateReport
    near_duplicates: NearDuplicateReport | None = None
    embedding: EmbeddingStats | None = None
    coverage: QueryCoverageReport | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "generated_at": self.generated_at,
            "n_documents": self.n_documents,
            "length": self.length.to_dict(),
            "vocabulary": self.vocabulary.to_dict(),
            "exact_duplicates": self.exact_duplicates.to_dict(),
            "notes": list(self.notes),
        }
        if self.near_duplicates is not None:
            d["near_duplicates"] = self.near_duplicates.to_dict()
        if self.embedding is not None:
            d["embedding"] = self.embedding.to_dict()
        if self.coverage is not None:
            d["coverage"] = self.coverage.to_dict()
        return d

    # -- narrative helpers ---------------------------------------------------

    def headline_status(self) -> str:
        """One-word verdict suitable for a Markdown badge / dashboard."""
        if self.length.empty_count > 0:
            return "needs_attention"
        if self.exact_duplicates.duplication_ratio > 0.20:
            return "needs_attention"
        if self.near_duplicates and self.near_duplicates.duplication_ratio > 0.30:
            return "needs_attention"
        if self.coverage and self.coverage.coverage_rate < 0.5:
            return "needs_attention"
        return "ready"

    def summary_lines(self) -> list[str]:
        lines = [
            f"RAG Readiness Audit  ({self.headline_status().upper()})",
            f"  Documents:           {self.n_documents}",
            (
                f"  Length (chars):      "
                f"median={self.length.char_median:.0f} "
                f"p90={self.length.char_p90:.0f} "
                f"p99={self.length.char_p99:.0f}"
            ),
            (f"  Empty / very short:  {self.length.empty_count} / {self.length.very_short_count}"),
            f"  Very long:           {self.length.very_long_count}",
            (
                f"  Vocabulary:          "
                f"unique={self.vocabulary.unique_tokens} "
                f"ttr={self.vocabulary.type_token_ratio:.3f} "
                f"hapax={self.vocabulary.hapax_ratio:.3f}"
            ),
            (
                f"  Exact duplicates:    "
                f"{self.exact_duplicates.n_duplicate_documents} "
                f"({self.exact_duplicates.duplication_ratio:.1%})"
            ),
        ]
        if self.near_duplicates is not None:
            lines.append(
                f"  Near duplicates:     "
                f"{self.near_duplicates.n_duplicate_documents} "
                f"@ ≥{self.near_duplicates.threshold:.2f} "
                f"({self.near_duplicates.duplication_ratio:.1%})"
            )
        if self.embedding is not None:
            lines.append(
                f"  Embedding health:    "
                f"centroid_norm={self.embedding.centroid_norm:.3f} "
                f"eff_rank={self.embedding.effective_rank:.1f} "
                f"hubness={self.embedding.hubness_skewness:.2f}"
            )
        if self.coverage is not None:
            lines.append(
                f"  Query coverage:      "
                f"covered={self.coverage.coverage_rate:.1%} "
                f"confident={self.coverage.confidence_rate:.1%}"
            )
        if self.notes:
            lines.append("  Notes:")
            lines.extend(f"   - {n}" for n in self.notes)
        return lines


def _build_notes(
    length: LengthStats,
    vocab: VocabularyStats,
    exact: DuplicateReport,
    near: NearDuplicateReport | None,
    coverage: QueryCoverageReport | None,
) -> list[str]:
    """Generate human-friendly notes that turn raw numbers into action items."""
    notes: list[str] = []
    if length.empty_count:
        notes.append(
            f"{length.empty_count} empty document(s) found — drop or backfill before indexing."
        )
    if length.very_short_count:
        notes.append(
            f"{length.very_short_count} very short document(s) "
            f"(< {length.very_short_threshold} chars) — consider merging with neighbours."
        )
    if length.very_long_count:
        notes.append(
            f"{length.very_long_count} very long document(s) "
            f"(> {length.very_long_threshold} chars) — chunk before encoding."
        )
    if exact.duplication_ratio > 0.05:
        notes.append(
            f"Exact-duplicate ratio {exact.duplication_ratio:.1%} — deduplicate to "
            "cut embedding cost and index size."
        )
    if near is not None and near.duplication_ratio > 0.20:
        notes.append(
            f"Near-duplicate ratio {near.duplication_ratio:.1%} at "
            f"≥{near.threshold:.2f} cosine — consider semantic dedup or MMR diversification."
        )
    if vocab.hapax_ratio > 0.6:
        notes.append(
            f"Hapax ratio {vocab.hapax_ratio:.1%} — heavy domain vocabulary, "
            "a domain-tuned encoder may outperform the default."
        )
    if coverage is not None and coverage.coverage_rate < 0.6:
        notes.append(
            f"Only {coverage.coverage_rate:.1%} of queries have a covered answer — "
            "expand the corpus or rewrite ambiguous queries."
        )
    return notes


def build_report(
    *,
    length: LengthStats,
    vocabulary: VocabularyStats,
    exact_duplicates: DuplicateReport,
    near_duplicates: NearDuplicateReport | None = None,
    embedding: EmbeddingStats | None = None,
    coverage: QueryCoverageReport | None = None,
    n_documents: int | None = None,
) -> RagReadinessReport:
    """Bundle the audit signals into a :class:`RagReadinessReport`.

    Args:
        length: Output of :func:`corpus_profile.length_stats`.
        vocabulary: Output of :func:`corpus_profile.vocabulary_stats`.
        exact_duplicates: Output of :func:`corpus_profile.exact_duplicate_report`.
        near_duplicates: Optional output of
            :func:`near_duplicates.near_duplicate_report`.
        embedding: Optional output of :func:`embedding_stats.embedding_stats`.
        coverage: Optional output of
            :func:`query_coverage.query_coverage_report`.
        n_documents: Override for the document count in the bundle. Falls
            back to ``length.n``.

    Returns:
        :class:`RagReadinessReport` with the auto-derived notes attached.
    """
    notes = _build_notes(length, vocabulary, exact_duplicates, near_duplicates, coverage)
    return RagReadinessReport(
        generated_at=_utc_now(),
        n_documents=n_documents if n_documents is not None else length.n,
        length=length,
        vocabulary=vocabulary,
        exact_duplicates=exact_duplicates,
        near_duplicates=near_duplicates,
        embedding=embedding,
        coverage=coverage,
        notes=notes,
    )
