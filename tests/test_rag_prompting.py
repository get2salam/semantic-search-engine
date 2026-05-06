import pytest

from rag_prompting import build_rag_prompt, build_refusal_prompt


def test_build_rag_prompt_includes_context_question_and_citation_instruction():
    prompt = build_rag_prompt("What is RAG?", "[S1] RAG combines retrieval and generation.")

    assert "Cite sources" in prompt
    assert "[S1] RAG combines" in prompt
    assert "Question:\nWhat is RAG?" in prompt
    assert prompt.endswith("Grounded answer:")


def test_build_rag_prompt_requires_question_and_context():
    with pytest.raises(ValueError, match="question"):
        build_rag_prompt("", "context")
    with pytest.raises(ValueError, match="context"):
        build_rag_prompt("question", "")


def test_build_refusal_prompt_names_missing_context_reason():
    prompt = build_refusal_prompt("What changed?", "No recent documents retrieved.")

    assert "not sufficient" in prompt
    assert "No recent documents retrieved" in prompt
