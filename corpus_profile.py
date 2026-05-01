"""
Corpus Profile
==============
Lightweight, pre-RAG corpus health checks. Profiles the *content* of a
document collection without ever touching the embedding model:

* **Length distribution** — character / word / token-ish stats across the
  corpus. Outliers (very long or very short docs) are the single most
  common cause of poor retrieval, so the profile reports both percentiles
  and the index of the worst offenders.
* **Vocabulary** — unique-token coverage, type/token ratio, and the
  long-tail share. Useful for spotting jargon-heavy corpora that need
  domain-specific encoders.
* **Exact duplicates** — hash-based detection so callers can deduplicate
  before paying to embed identical documents twice.

Everything here is stdlib-only and streams over the corpus once. The
embedding-based near-duplicate analysis lives in :mod:`near_duplicates`
so callers can opt into the heavier work explicitly.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "LengthStats",
    "VocabularyStats",
    "DuplicateGroup",
    "DuplicateReport",
    "length_stats",
    "vocabulary_stats",
    "exact_duplicate_report",
]


# Tokeniser: lower-cased word characters. Deliberately simple so the
# vocabulary numbers are deterministic across platforms; users wanting
# linguistic-grade tokenisation should bring their own.
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def _tokenise(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


@dataclass(frozen=True)
class LengthStats:
    """Length-distribution summary for a corpus.

    Attributes:
        n: Number of documents profiled.
        char_mean / char_median / char_p90 / char_p99: Character-length stats.
        word_mean / word_median / word_p90 / word_p99: Word-count stats.
        empty_count: Documents with zero non-whitespace characters.
        very_short_count: Documents below ``very_short_chars`` characters.
        very_long_count: Documents above ``very_long_chars`` characters.
        very_short_threshold / very_long_threshold: The cut-offs used.
        longest_indices: Up to five indices of the longest documents (for
            quick triage of outliers).
        shortest_indices: Up to five indices of the shortest non-empty
            documents.
    """

    n: int
    char_mean: float
    char_median: float
    char_p90: float
    char_p99: float
    word_mean: float
    word_median: float
    word_p90: float
    word_p99: float
    empty_count: int
    very_short_count: int
    very_long_count: int
    very_short_threshold: int
    very_long_threshold: int
    longest_indices: list[int] = field(default_factory=list)
    shortest_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def length_stats(
    documents: list[str],
    *,
    very_short_chars: int = 20,
    very_long_chars: int = 4000,
    outlier_top_n: int = 5,
) -> LengthStats:
    """Compute character/word length distribution for a corpus.

    Args:
        documents: List of raw document strings.
        very_short_chars: Documents shorter than this (in characters) are
            counted as ``very_short``. Defaults to 20 — below this most
            embedders return near-noise vectors.
        very_long_chars: Documents longer than this are counted as
            ``very_long``. Defaults to 4000 chars (~ a transformer's typical
            context window after tokenisation).
        outlier_top_n: How many longest/shortest indices to retain for
            triage in the report.

    Returns:
        :class:`LengthStats` with means, medians, p90/p99, outlier counts
        and indices of the longest/shortest documents.
    """
    n = len(documents)
    if n == 0:
        return LengthStats(
            n=0,
            char_mean=0.0,
            char_median=0.0,
            char_p90=0.0,
            char_p99=0.0,
            word_mean=0.0,
            word_median=0.0,
            word_p90=0.0,
            word_p99=0.0,
            empty_count=0,
            very_short_count=0,
            very_long_count=0,
            very_short_threshold=very_short_chars,
            very_long_threshold=very_long_chars,
        )

    char_lens: list[int] = []
    word_lens: list[int] = []
    empty = 0
    very_short = 0
    very_long = 0

    for doc in documents:
        text = doc or ""
        c = len(text)
        w = len(_tokenise(text))
        char_lens.append(c)
        word_lens.append(w)
        if not text.strip():
            empty += 1
            continue
        if c < very_short_chars:
            very_short += 1
        if c > very_long_chars:
            very_long += 1

    sorted_chars = sorted(char_lens)
    sorted_words = sorted(word_lens)

    # Indices of longest/shortest meaningful docs (skip empties / whitespace-only).
    indexed = sorted(
        ((c, i) for i, c in enumerate(char_lens) if c > 0 and (documents[i] or "").strip()),
        key=lambda t: t[0],
    )
    shortest_indices = [i for _, i in indexed[:outlier_top_n]]
    longest_indices = [i for _, i in indexed[-outlier_top_n:][::-1]]

    return LengthStats(
        n=n,
        char_mean=sum(char_lens) / n,
        char_median=_percentile(sorted_chars, 50.0),
        char_p90=_percentile(sorted_chars, 90.0),
        char_p99=_percentile(sorted_chars, 99.0),
        word_mean=sum(word_lens) / n,
        word_median=_percentile(sorted_words, 50.0),
        word_p90=_percentile(sorted_words, 90.0),
        word_p99=_percentile(sorted_words, 99.0),
        empty_count=empty,
        very_short_count=very_short,
        very_long_count=very_long,
        very_short_threshold=very_short_chars,
        very_long_threshold=very_long_chars,
        longest_indices=longest_indices,
        shortest_indices=shortest_indices,
    )


@dataclass(frozen=True)
class VocabularyStats:
    """Vocabulary-richness summary for a corpus.

    Attributes:
        total_tokens: Sum of token counts across all documents.
        unique_tokens: Distinct token types.
        type_token_ratio: ``unique_tokens / total_tokens`` — a classic
            lexical-diversity proxy. Approaches 1.0 on tiny / disjoint
            corpora; small values (<0.05) usually mean heavy repetition.
        hapax_ratio: Share of types that occur exactly once. High ratios
            (>0.5) suggest a long lexical tail — common in technical /
            domain-specific corpora and a hint that domain-tuned embedders
            may help.
        top_tokens: Up to ``top_n`` most frequent tokens with their counts,
            ordered descending. Empty when the corpus has no tokens.
    """

    total_tokens: int
    unique_tokens: int
    type_token_ratio: float
    hapax_ratio: float
    top_tokens: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Tuples become lists in JSON; keep that shape explicit.
        d["top_tokens"] = [[tok, count] for tok, count in self.top_tokens]
        return d


def vocabulary_stats(
    documents: list[str],
    *,
    top_n: int = 20,
    stopwords: set[str] | None = None,
) -> VocabularyStats:
    """Compute lexical-diversity stats over a corpus.

    Args:
        documents: List of raw document strings.
        top_n: How many top-frequency tokens to report.
        stopwords: Optional set of lower-case tokens to drop before
            counting. Useful for English corpora where high-frequency
            stopwords would otherwise dominate ``top_tokens``.

    Returns:
        :class:`VocabularyStats` with totals, type/token ratio, hapax
        ratio, and the top-N tokens.
    """
    counter: Counter[str] = Counter()
    for doc in documents:
        if not doc:
            continue
        tokens = _tokenise(doc)
        if stopwords:
            tokens = [t for t in tokens if t not in stopwords]
        counter.update(tokens)

    total = sum(counter.values())
    unique = len(counter)
    if total == 0:
        return VocabularyStats(0, 0, 0.0, 0.0, [])

    hapax = sum(1 for c in counter.values() if c == 1)
    return VocabularyStats(
        total_tokens=total,
        unique_tokens=unique,
        type_token_ratio=unique / total,
        hapax_ratio=hapax / unique if unique else 0.0,
        top_tokens=counter.most_common(top_n),
    )


def _normalise_for_dedup(text: str) -> str:
    """Normalise whitespace + case for hash-based dedup."""
    return " ".join((text or "").lower().split())


@dataclass(frozen=True)
class DuplicateGroup:
    """A cluster of corpus indices that share an exact (normalised) text."""

    representative: int
    indices: list[int]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateReport:
    """Aggregate exact-duplicate stats for a corpus."""

    n_documents: int
    n_unique: int
    n_duplicate_documents: int
    n_groups: int
    duplication_ratio: float
    groups: list[DuplicateGroup] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["groups"] = [g.to_dict() for g in self.groups]
        return d


def exact_duplicate_report(
    documents: list[str],
    *,
    max_groups: int = 25,
    ignore_empty: bool = True,
) -> DuplicateReport:
    """Detect exact (whitespace/case-normalised) duplicates.

    Args:
        documents: List of raw document strings.
        max_groups: Cap on how many duplicate groups to retain in the
            report (largest groups first). The aggregate counts always
            cover the whole corpus.
        ignore_empty: When ``True`` (default), strings that normalise to
            an empty string are excluded from the duplicate report —
            empty docs are flagged separately by :func:`length_stats`.

    Returns:
        :class:`DuplicateReport` with totals and the largest duplicate
        clusters (each carrying the corpus indices that collide).
    """
    n = len(documents)
    if n == 0:
        return DuplicateReport(0, 0, 0, 0, 0.0)

    buckets: dict[str, list[int]] = {}
    for i, doc in enumerate(documents):
        norm = _normalise_for_dedup(doc)
        if ignore_empty and not norm:
            continue
        # Hashing keeps memory bounded for very long documents.
        key = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        buckets.setdefault(key, []).append(i)

    groups_all: list[DuplicateGroup] = []
    n_dup_docs = 0
    for indices in buckets.values():
        if len(indices) <= 1:
            continue
        n_dup_docs += len(indices) - 1
        rep = indices[0]
        groups_all.append(
            DuplicateGroup(
                representative=rep,
                indices=indices,
                text=documents[rep],
            )
        )
    groups_all.sort(key=lambda g: -len(g.indices))

    n_unique = len(buckets) + (n - sum(len(v) for v in buckets.values()))
    return DuplicateReport(
        n_documents=n,
        n_unique=n_unique,
        n_duplicate_documents=n_dup_docs,
        n_groups=len(groups_all),
        duplication_ratio=n_dup_docs / n,
        groups=groups_all[:max_groups],
    )
