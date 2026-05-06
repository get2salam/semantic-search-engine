from rag_attribution import (
    SourceCitation,
    assign_citation_labels,
    extract_citation_labels,
    format_source_bibliography,
)


def test_assign_citation_labels_is_stable():
    labels = assign_citation_labels(["alpha.md", "beta.md"])

    assert [label.label for label in labels] == ["S1", "S2"]
    assert labels[0].source == "alpha.md"


def test_extract_citation_labels_deduplicates_answer_labels():
    labels = extract_citation_labels("Use retrieval [S1]. It is grounded [S1] [S2].")

    assert labels == {"S1", "S2"}


def test_format_source_bibliography_includes_optional_titles():
    text = format_source_bibliography(
        [
            SourceCitation("S1", "alpha.md", "Alpha"),
            SourceCitation("S2", "beta.md"),
        ]
    )

    assert "[S1] alpha.md — Alpha" in text
    assert "[S2] beta.md" in text
