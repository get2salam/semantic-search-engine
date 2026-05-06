import pytest

from rag_query import build_retrieval_plan, generate_query_variants, normalize_query


def test_normalize_query_collapses_whitespace_and_case():
    assert normalize_query("  What   is RAG?  ") == "what is rag?"


def test_generate_query_variants_expands_ai_terms_and_keywords():
    variants = generate_query_variants("What is RAG for AI")

    assert variants[0] == "what is rag for ai"
    assert any("artificial intelligence" in variant for variant in variants)
    assert "rag ai" in variants


def test_build_retrieval_plan_marks_hybrid_when_variants_exist():
    plan = build_retrieval_plan("How does RAG help AI?", top_k=8)

    assert plan.top_k == 8
    assert plan.use_hybrid is True
    assert plan.variants


def test_build_retrieval_plan_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        build_retrieval_plan("rag", top_k=0)
