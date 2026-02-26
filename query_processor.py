"""
Query Preprocessing Pipeline
=============================
A modular NLP pipeline for cleaning, normalising, and expanding search queries
before they are encoded by the embedding model.

Stages (all optional, enabled by default):
    1. Unicode normalisation (NFC) and whitespace collapse
    2. Case normalisation (configurable)
    3. Stop-word filtering  (preserves single-word queries)
    4. Query expansion via a lightweight synonym map

Author: get2salam
License: MIT
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "QueryNormalizer",
    "StopWordFilter",
    "QueryExpander",
    "QueryProcessor",
    "ProcessedQuery",
]

# ---------------------------------------------------------------------------
# Default stop-word list (English)
# ---------------------------------------------------------------------------
_DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "can",
        "could",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "as",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "which",
        "who",
        "whom",
        "what",
        "where",
        "when",
        "why",
        "how",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "us",
        "you",
        "your",
        "he",
        "she",
        "it",
        "its",
        "they",
        "them",
        "their",
    }
)

# ---------------------------------------------------------------------------
# Default synonym map (token → list of synonyms to append)
# ---------------------------------------------------------------------------
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "ml": ["machine learning"],
    "ai": ["artificial intelligence"],
    "nlp": ["natural language processing"],
    "dl": ["deep learning"],
    "nn": ["neural network"],
    "rag": ["retrieval augmented generation"],
    "llm": ["large language model"],
    "api": ["application programming interface"],
    "db": ["database"],
    "ocr": ["optical character recognition"],
    "cv": ["computer vision"],
    "ir": ["information retrieval"],
}


# ---------------------------------------------------------------------------
# Data class for pipeline output
# ---------------------------------------------------------------------------
@dataclass
class ProcessedQuery:
    """Result of running a query through the :class:`QueryProcessor`."""

    original: str
    """The raw query as supplied by the caller."""

    normalised: str
    """Query after Unicode + whitespace normalisation and case folding."""

    filtered: str
    """Query after stop-word removal (may equal *normalised* for short queries)."""

    expanded: str
    """Final query string after synonym expansion — use this for embedding."""

    tokens: list[str]
    """Whitespace-split tokens of the *filtered* query."""

    expansions: list[str] = field(default_factory=list)
    """Phrases appended by the :class:`QueryExpander` (for transparency)."""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ProcessedQuery("
            f"original={self.original!r}, "
            f"expanded={self.expanded!r}, "
            f"expansions={self.expansions!r})"
        )


# ---------------------------------------------------------------------------
# Stage 1 — Normaliser
# ---------------------------------------------------------------------------
class QueryNormalizer:
    """
    Normalise raw query strings.

    Operations (all enabled by default):
        - Unicode NFC normalisation to collapse composed/decomposed forms.
        - ASCII transliteration of common accented characters (optional).
        - Collapse runs of whitespace to a single space and strip ends.
        - Case folding to lowercase (can be disabled).

    Parameters:
        lowercase: Convert query to lowercase. Default ``True``.
        strip_accents: Map accented letters to their ASCII base. Default ``False``.
        max_length: Truncate query to this many characters (0 = unlimited).
    """

    def __init__(
        self,
        lowercase: bool = True,
        strip_accents: bool = False,
        max_length: int = 512,
    ) -> None:
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.max_length = max_length
        self._ws_re = re.compile(r"\s+")

    # ------------------------------------------------------------------
    def normalise(self, query: str) -> str:
        """Return the normalised query string."""
        if not isinstance(query, str):
            raise TypeError(f"Query must be str, got {type(query).__name__!r}")

        # 1. NFC Unicode form
        text = unicodedata.normalize("NFC", query)

        # 2. Optional accent stripping
        if self.strip_accents:
            text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")

        # 3. Collapse whitespace
        text = self._ws_re.sub(" ", text).strip()

        # 4. Lowercase
        if self.lowercase:
            text = text.lower()

        # 5. Truncate
        if self.max_length > 0 and len(text) > self.max_length:
            text = text[: self.max_length].rsplit(" ", 1)[0]

        return text

    # Alias so the class is callable as a pipeline stage
    __call__ = normalise


# ---------------------------------------------------------------------------
# Stage 2 — Stop-word filter
# ---------------------------------------------------------------------------
class StopWordFilter:
    """
    Remove stop words from a tokenised query.

    Short queries (≤ *min_tokens* tokens) are returned unchanged to avoid
    producing an empty string when the user types a single stop word such as
    ``"the"``.

    Parameters:
        stop_words: Set of lowercased stop words. Defaults to
                    :data:`_DEFAULT_STOP_WORDS`.
        min_tokens: Do not filter if the query has ≤ this many tokens.
                    Default ``1``.
        min_remaining: Always keep at least this many tokens. Default ``1``.
    """

    def __init__(
        self,
        stop_words: frozenset[str] | set[str] | None = None,
        min_tokens: int = 1,
        min_remaining: int = 1,
    ) -> None:
        self.stop_words: frozenset[str] = (
            frozenset(stop_words) if stop_words is not None else _DEFAULT_STOP_WORDS
        )
        self.min_tokens = min_tokens
        self.min_remaining = min_remaining

    # ------------------------------------------------------------------
    def filter(self, query: str) -> tuple[str, list[str]]:
        """
        Remove stop words from *query*.

        Returns:
            A ``(filtered_query, tokens)`` tuple where *tokens* are the
            non-stop-word tokens.
        """
        tokens = query.split()

        if len(tokens) <= self.min_tokens:
            return query, tokens

        content = [t for t in tokens if t.lower() not in self.stop_words]

        # Safety net — never return an empty query
        if len(content) < self.min_remaining:
            content = tokens[: self.min_remaining]

        return " ".join(content), content

    def __call__(self, query: str) -> tuple[str, list[str]]:  # pragma: no cover
        return self.filter(query)


# ---------------------------------------------------------------------------
# Stage 3 — Query expander
# ---------------------------------------------------------------------------
class QueryExpander:
    """
    Append synonyms / related terms for known abbreviations and jargon.

    The expansion is purely additive: original tokens are kept so that the
    embedding model still processes the user's exact wording.

    Parameters:
        synonyms: Mapping of ``token → [synonym_phrase, ...]``.
                  Keys are compared in a case-insensitive manner.
                  Defaults to :data:`_DEFAULT_SYNONYMS`.
        max_expansions: Maximum number of synonym phrases to append per
                        query. ``0`` means unlimited. Default ``3``.
    """

    def __init__(
        self,
        synonyms: dict[str, list[str]] | None = None,
        max_expansions: int = 3,
    ) -> None:
        raw = synonyms if synonyms is not None else _DEFAULT_SYNONYMS
        # Normalise keys to lowercase once at init time
        self.synonyms: dict[str, list[str]] = {k.lower(): v for k, v in raw.items()}
        self.max_expansions = max_expansions

    # ------------------------------------------------------------------
    def expand(self, query: str, tokens: Sequence[str]) -> tuple[str, list[str]]:
        """
        Expand *query* using matched synonyms for *tokens*.

        Parameters:
            query: The (filtered) query string.
            tokens: Tokens to look up in the synonym map.

        Returns:
            A ``(expanded_query, expansions)`` tuple.  *expansions* lists the
            phrases that were appended for transparency / logging.
        """
        seen: set[str] = set()
        additions: list[str] = []

        for token in tokens:
            key = token.lower()
            if key in self.synonyms:
                for phrase in self.synonyms[key]:
                    if phrase not in seen:
                        seen.add(phrase)
                        additions.append(phrase)
                        if self.max_expansions > 0 and len(additions) >= self.max_expansions:
                            break
            if self.max_expansions > 0 and len(additions) >= self.max_expansions:
                break

        expanded = query + " " + " ".join(additions) if additions else query

        return expanded, additions

    def __call__(
        self, query: str, tokens: Sequence[str]
    ) -> tuple[str, list[str]]:  # pragma: no cover
        return self.expand(query, tokens)


# ---------------------------------------------------------------------------
# Orchestrator — QueryProcessor
# ---------------------------------------------------------------------------
class QueryProcessor:
    """
    Orchestrate the full query preprocessing pipeline.

    The pipeline is::

        raw query
          │
          ▼
        QueryNormalizer     (Unicode NFC, whitespace, case)
          │
          ▼
        StopWordFilter      (remove stop words)
          │
          ▼
        QueryExpander       (synonym / abbreviation expansion)
          │
          ▼
        ProcessedQuery

    Parameters:
        normalizer: :class:`QueryNormalizer` instance (or ``None`` to skip).
        stop_filter: :class:`StopWordFilter` instance (or ``None`` to skip).
        expander: :class:`QueryExpander` instance (or ``None`` to skip).

    Example::

        >>> processor = QueryProcessor()
        >>> result = processor.process("What is NLP and ML?")
        >>> print(result.expanded)
        'nlp ml natural language processing machine learning'
    """

    def __init__(
        self,
        normalizer: QueryNormalizer | None = None,
        stop_filter: StopWordFilter | None = None,
        expander: QueryExpander | None = None,
    ) -> None:
        self.normalizer = normalizer or QueryNormalizer()
        self.stop_filter = stop_filter or StopWordFilter()
        self.expander = expander or QueryExpander()

    # ------------------------------------------------------------------
    def process(self, query: str) -> ProcessedQuery:
        """
        Run the full pipeline and return a :class:`ProcessedQuery`.

        Parameters:
            query: Raw user query string.

        Returns:
            :class:`ProcessedQuery` with all intermediate and final forms.
        """
        # Stage 1 — normalise
        normalised = self.normalizer.normalise(query)

        # Stage 2 — filter stop words
        filtered, tokens = self.stop_filter.filter(normalised)

        # Stage 3 — expand
        expanded, expansions = self.expander.expand(filtered, tokens)

        return ProcessedQuery(
            original=query,
            normalised=normalised,
            filtered=filtered,
            expanded=expanded,
            tokens=tokens,
            expansions=expansions,
        )

    def process_batch(self, queries: Sequence[str]) -> list[ProcessedQuery]:
        """
        Process a batch of queries and return a list of :class:`ProcessedQuery`.

        Parameters:
            queries: Sequence of raw query strings.

        Returns:
            List of :class:`ProcessedQuery` in the same order as input.
        """
        return [self.process(q) for q in queries]

    # Convenience shortcut
    __call__ = process
