"""Context-window assembly helpers for grounded AI answers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievedPassage:
    """A retriever hit normalized for prompt construction."""

    text: str
    source: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _format_passage(index: int, passage: RetrievedPassage, *, include_scores: bool) -> str:
    label = f"S{index}"
    title = passage.metadata.get("title") or passage.source
    score = f" score={passage.score:.3f}" if include_scores and passage.score is not None else ""
    return f"[{label}] {title}{score}\n{passage.text.strip()}"


def build_context_window(
    passages: list[RetrievedPassage],
    *,
    max_chars: int = 4000,
    include_scores: bool = False,
) -> str:
    """Build a citation-labelled context block within a character budget."""

    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    selected: list[str] = []
    remaining = max_chars
    for index, passage in enumerate(passages, start=1):
        block = _format_passage(index, passage, include_scores=include_scores)
        separator = "\n\n" if selected else ""
        budget = remaining - len(separator)
        if budget <= 0:
            break
        if len(block) > budget:
            if budget < 20:
                break
            block = block[: max(0, budget - 1)].rstrip() + "…"
        selected.append(separator + block)
        remaining = max_chars - sum(len(part) for part in selected)
        if remaining <= 0:
            break
    return "".join(selected)
