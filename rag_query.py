"""Deterministic query-planning helpers for AI retrieval flows."""

import re
from dataclasses import dataclass

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "how",
    "is",
    "of",
    "the",
    "to",
    "what",
    "with",
}
_EXPANSIONS = {
    "ai": "artificial intelligence",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
}


@dataclass(frozen=True)
class RetrievalPlan:
    """A small plan object that can drive dense, lexical, or hybrid retrieval."""

    original_query: str
    variants: list[str]
    top_k: int
    use_hybrid: bool


def normalize_query(query: str) -> str:
    """Normalize whitespace and casing without destroying user intent."""

    return re.sub(r"\s+", " ", (query or "").strip()).lower()


def generate_query_variants(query: str, *, max_variants: int = 4) -> list[str]:
    """Generate deterministic retrieval query variants."""

    normalized = normalize_query(query)
    if not normalized:
        return []

    variants: list[str] = [normalized]
    expanded_terms = [_EXPANSIONS[token] for token in normalized.split() if token in _EXPANSIONS]
    if expanded_terms:
        variants.append(f"{normalized} {' '.join(expanded_terms)}")

    keywords = " ".join(token for token in normalized.split() if token not in _STOPWORDS)
    if keywords and keywords not in variants:
        variants.append(keywords)

    questionless = re.sub(r"^(what|how|why|when|where)\s+", "", normalized)
    if questionless and questionless not in variants:
        variants.append(questionless)

    return variants[:max_variants]


def build_retrieval_plan(query: str, *, top_k: int = 5) -> RetrievalPlan:
    """Create a simple retrieval plan from a natural-language query."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    variants = generate_query_variants(query)
    return RetrievalPlan(
        original_query=query.strip(),
        variants=variants,
        top_k=top_k,
        use_hybrid=len(variants) > 1,
    )
