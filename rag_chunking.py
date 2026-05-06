"""Deterministic text chunking helpers for retrieval-augmented generation.

The functions in this module are intentionally dependency-free so they can be
used in ingestion jobs, notebooks, CLIs, and tests without loading an embedding
model.  They produce stable chunk IDs, preserve word offsets, and keep metadata
attached to every chunk.
"""

from dataclasses import dataclass, field
import re
from typing import Any


_WORD_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class TextChunk:
    """A document slice ready to embed or pass into a retriever."""

    chunk_id: str
    text: str
    start_word: int
    end_word: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Return a cheap token estimate suitable for budget planning."""

        return max(1, round((self.end_word - self.start_word) * 1.33))


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def chunk_text(
    text: str,
    *,
    max_words: int = 180,
    overlap_words: int = 30,
    source_id: str = "doc",
    metadata: dict[str, Any] | None = None,
) -> list[TextChunk]:
    """Split text into overlapping word windows with stable chunk IDs.

    Args:
        text: Source text to split.
        max_words: Maximum number of words in each chunk.
        overlap_words: Number of words repeated between adjacent chunks.
        source_id: Prefix used to build stable chunk identifiers.
        metadata: Optional metadata copied onto each chunk.

    Returns:
        Ordered ``TextChunk`` objects. Empty input returns an empty list.
    """

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = _words(text)
    if not words:
        return []

    step = max_words - overlap_words
    chunks: list[TextChunk] = []
    base_metadata = dict(metadata or {})
    for start in range(0, len(words), step):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunks.append(
            TextChunk(
                chunk_id=f"{source_id}:{len(chunks) + 1:04d}",
                text=" ".join(chunk_words),
                start_word=start,
                end_word=end,
                metadata=base_metadata.copy(),
            )
        )
        if end == len(words):
            break
    return chunks


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    slug = _SLUG_RE.sub("-", value.lower()).strip("-")
    return slug or "section"


def chunk_markdown_sections(
    markdown: str,
    *,
    max_words: int = 180,
    overlap_words: int = 30,
    source_id: str = "doc",
    include_heading: bool = True,
) -> list[TextChunk]:
    """Chunk Markdown while preserving heading metadata.

    Each heading starts a new section. Long sections still use overlapping word
    windows, but every emitted chunk includes ``heading`` and ``heading_level``
    metadata so downstream RAG prompts can cite where the passage came from.
    """

    sections: list[tuple[str | None, int | None, list[str]]] = []
    current_heading: str | None = None
    current_level: int | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines or current_heading:
            sections.append((current_heading, current_level, current_lines))
        current_lines = []

    for line in (markdown or "").splitlines():
        match = _HEADING_RE.match(line)
        if match:
            flush()
            current_heading = match.group(2).strip()
            current_level = len(match.group(1))
        else:
            current_lines.append(line)
    flush()

    if not sections:
        return []

    output: list[TextChunk] = []
    for heading, level, lines in sections:
        body = "\n".join(line.strip() for line in lines).strip()
        if not body and not heading:
            continue
        section_text = f"{heading}\n{body}".strip() if include_heading and heading else body
        slug = _slugify(heading or f"section-{len(output) + 1}")
        section_chunks = chunk_text(
            section_text,
            max_words=max_words,
            overlap_words=overlap_words,
            source_id=f"{source_id}:{slug}",
            metadata={
                "heading": heading,
                "heading_level": level,
                "source_id": source_id,
            },
        )
        output.extend(section_chunks)
    return output
