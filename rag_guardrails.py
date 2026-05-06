"""Lightweight guardrails for checking grounded RAG answers."""

import re

from rag_attribution import extract_citation_labels

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_CITATION_RE = re.compile(r"\[S\d+\]")


def split_claims(answer: str) -> list[str]:
    """Split an answer into citation-checkable sentence-like claims."""

    return [part.strip() for part in _SENTENCE_RE.split(answer.strip()) if part.strip()]


def find_uncited_claims(answer: str, *, min_words: int = 5) -> list[str]:
    """Return substantial claims that do not include a source citation."""

    uncited: list[str] = []
    for claim in split_claims(answer or ""):
        if len(claim.split()) < min_words:
            continue
        if not _CITATION_RE.search(claim):
            uncited.append(claim)
    return uncited


def validate_answer_citations(answer: str, available_labels: set[str]) -> dict[str, object]:
    """Check whether cited labels exist and substantial claims are cited."""

    cited = extract_citation_labels(answer)
    unknown = cited - set(available_labels)
    uncited = find_uncited_claims(answer)
    return {
        "ok": not unknown and not uncited,
        "cited_labels": sorted(cited),
        "unknown_labels": sorted(unknown),
        "uncited_claims": uncited,
    }
