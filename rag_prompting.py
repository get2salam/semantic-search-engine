"""Prompt templates for grounded RAG answers."""

from __future__ import annotations

import re

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer using only the provided context. "
    "Cite sources with [S1]-style labels after each factual claim. "
    "If the context is insufficient, say what is missing instead of guessing."
)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_STYLE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,39}$")


def _clean_block(value: str | None, field_name: str) -> str:
    """Strip prompt-hostile control characters from a required text block."""

    cleaned = _CONTROL_CHARS.sub("", value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} is required")
    return cleaned


def _quote_block(value: str) -> str:
    """Indent untrusted text so embedded section labels cannot break the prompt."""

    return "\n".join(f"    {line}" for line in value.splitlines())


def _clean_answer_style(answer_style: str) -> str:
    """Accept only a short single-line style label, not arbitrary instructions."""

    style = _CONTROL_CHARS.sub("", answer_style or "").strip()
    if not _SAFE_STYLE.fullmatch(style):
        raise ValueError("answer_style must be a short single-line label")
    return style


def build_rag_prompt(
    question: str,
    context: str,
    *,
    instructions: str = DEFAULT_RAG_INSTRUCTIONS,
    answer_style: str = "concise",
) -> str:
    """Build a deterministic grounded-answer prompt.

    The retrieved context and user question are quoted under fixed headings so
    a malicious document cannot inject a new ``Question:``/``Grounded answer:``
    section by placing those labels at the start of a line.
    """

    question = _clean_block(question, "question")
    context = _clean_block(context, "context")
    instructions = _clean_block(instructions, "instructions")
    answer_style = _clean_answer_style(answer_style)

    return (
        f"Instructions:\n{instructions}\n\n"
        f"Answer style: {answer_style}\n\n"
        f"Context:\n{_quote_block(context)}\n\n"
        f"Question:\n{_quote_block(question)}\n\n"
        "Grounded answer:"
    )


def build_refusal_prompt(question: str, missing_context_reason: str) -> str:
    """Build a short fallback prompt for answer abstention flows."""

    question = _clean_block(question, "question")
    missing_context_reason = _clean_block(missing_context_reason, "missing_context_reason")
    return (
        "The retrieval context is not sufficient to answer safely.\n"
        f"Question:\n{_quote_block(question)}\n"
        f"Missing context:\n{_quote_block(missing_context_reason)}\n"
        "Respond with a brief explanation and suggest the next retrieval step."
    )
