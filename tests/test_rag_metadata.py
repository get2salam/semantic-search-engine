from rag_metadata import filter_records, get_nested, metadata_matches


def test_get_nested_reads_dotted_metadata():
    metadata = {"source": {"type": "guide", "year": 2026}}

    assert get_nested(metadata, "source.type") == "guide"
    assert get_nested(metadata, "source.missing", "fallback") == "fallback"


def test_metadata_matches_exact_any_range_and_contains_rules():
    metadata = {
        "collection": "ai-notes",
        "year": 2026,
        "tags": ["rag", "retrieval"],
    }

    assert metadata_matches(
        metadata,
        {
            "collection": {"any": ["ai-notes", "ml-notes"]},
            "year": {"min": 2025, "max": 2027},
            "tags": {"contains": "rag"},
        },
    )
    assert not metadata_matches(metadata, {"tags": {"contains": "vision"}})


def test_filter_records_keeps_matching_metadata():
    records = [
        {"id": "a", "metadata": {"topic": "rag"}},
        {"id": "b", "metadata": {"topic": "agents"}},
    ]

    filtered = filter_records(records, {"topic": "rag"})

    assert [record["id"] for record in filtered] == ["a"]
