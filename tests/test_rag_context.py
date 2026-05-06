import pytest

from rag_context import RetrievedPassage, build_context_window


def test_build_context_window_labels_sources_and_scores():
    context = build_context_window(
        [RetrievedPassage("Alpha passage", "doc-a", 0.91, {"title": "Guide A"})],
        include_scores=True,
    )

    assert "[S1] Guide A score=0.910" in context
    assert "Alpha passage" in context


def test_build_context_window_respects_character_budget():
    context = build_context_window(
        [RetrievedPassage("x" * 200, "long-doc")],
        max_chars=60,
    )

    assert len(context) <= 60
    assert context.endswith("…")


def test_build_context_window_rejects_invalid_budget():
    with pytest.raises(ValueError, match="max_chars"):
        build_context_window([], max_chars=0)
