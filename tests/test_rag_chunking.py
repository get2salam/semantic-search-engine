import pytest

from rag_chunking import TextChunk, chunk_text


def test_chunk_text_returns_stable_overlapping_windows():
    text = " ".join(f"w{i}" for i in range(12))

    chunks = chunk_text(text, max_words=5, overlap_words=2, source_id="guide")

    assert [chunk.chunk_id for chunk in chunks] == [
        "guide:0001",
        "guide:0002",
        "guide:0003",
        "guide:0004",
    ]
    assert [chunk.start_word for chunk in chunks] == [0, 3, 6, 9]
    assert chunks[1].text.startswith("w3 w4")
    assert chunks[-1].text == "w9 w10 w11"


def test_chunk_text_copies_metadata_per_chunk():
    chunks = chunk_text(
        "alpha beta gamma delta",
        max_words=2,
        overlap_words=0,
        source_id="doc",
        metadata={"collection": "demo"},
    )

    assert chunks[0].metadata == {"collection": "demo"}
    assert chunks[0].metadata is not chunks[1].metadata


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError, match="smaller"):
        chunk_text("alpha beta", max_words=3, overlap_words=3)


def test_text_chunk_token_estimate_is_positive():
    chunk = TextChunk("a", "hello world", 0, 2)

    assert chunk.token_estimate >= 1
