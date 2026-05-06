"""Prompt templates for grounded RAG answers."""

DEFAULT_RAG_INSTRUCTIONS = (
    "Answer using only the provided context. "
    "Cite sources with [S1]-style labels after each factual claim. "
    "If the context is insufficient, say what is missing instead of guessing."
)


def build_rag_prompt(
    question: str,
    context: str,
    *,
    instructions: str = DEFAULT_RAG_INSTRUCTIONS,
    answer_style: str = "concise",
) -> str:
    """Build a deterministic grounded-answer prompt."""

    question = (question or "").strip()
    context = (context or "").strip()
    if not question:
        raise ValueError("question is required")
    if not context:
        raise ValueError("context is required")

    return (
        f"Instructions:\n{instructions}\n\n"
        f"Answer style: {answer_style}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Grounded answer:"
    )


def build_refusal_prompt(question: str, missing_context_reason: str) -> str:
    """Build a short fallback prompt for answer abstention flows."""

    return (
        "The retrieval context is not sufficient to answer safely.\n"
        f"Question: {(question or '').strip()}\n"
        f"Missing context: {missing_context_reason.strip()}\n"
        "Respond with a brief explanation and suggest the next retrieval step."
    )
