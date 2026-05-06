"""Source attribution helpers for RAG answers."""

import re
from dataclasses import dataclass

_CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True)
class SourceCitation:
    """Display metadata for a source that can be cited in an answer."""

    label: str
    source: str
    title: str | None = None


def assign_citation_labels(sources: list[str], *, prefix: str = "S") -> list[SourceCitation]:
    """Assign stable display labels to source identifiers."""

    return [
        SourceCitation(label=f"{prefix}{index}", source=source, title=None)
        for index, source in enumerate(sources, start=1)
    ]


def extract_citation_labels(answer: str) -> set[str]:
    """Extract ``[S1]``-style citation labels from generated text."""

    return set(_CITATION_RE.findall(answer or ""))


def format_source_bibliography(citations: list[SourceCitation]) -> str:
    """Render a compact source list for answer footers or logs."""

    lines = []
    for citation in citations:
        title = f" — {citation.title}" if citation.title else ""
        lines.append(f"[{citation.label}] {citation.source}{title}")
    return "\n".join(lines)
