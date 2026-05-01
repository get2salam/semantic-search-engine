"""Tests for the corpus_profile module (length + vocabulary)."""

from __future__ import annotations

import json

import pytest

from corpus_profile import (
    LengthStats,
    VocabularyStats,
    length_stats,
    vocabulary_stats,
)

# ---------------------------------------------------------------------------
# length_stats
# ---------------------------------------------------------------------------


def test_length_stats_empty_corpus_returns_zeros():
    stats = length_stats([])
    assert isinstance(stats, LengthStats)
    assert stats.n == 0
    assert stats.char_mean == 0
    assert stats.empty_count == 0
    assert stats.longest_indices == []
    assert stats.shortest_indices == []


def test_length_stats_basic_distribution():
    docs = ["a" * 10, "b" * 50, "c" * 100, "d" * 200]
    stats = length_stats(docs)
    assert stats.n == 4
    # Median of [10, 50, 100, 200] (linear interp) = (50 + 100) / 2 = 75
    assert stats.char_median == pytest.approx(75.0)
    assert stats.char_mean == pytest.approx(90.0)
    assert stats.char_p99 > stats.char_p90 >= stats.char_median


def test_length_stats_flags_outliers():
    docs = [
        "x" * 5,  # very short
        "ok " * 10,
        "ok " * 100,
        "y" * 8000,  # very long
    ]
    stats = length_stats(docs, very_short_chars=20, very_long_chars=4000)
    assert stats.very_short_count == 1
    assert stats.very_long_count == 1
    assert 3 in stats.longest_indices  # the 8000-char doc
    assert 0 in stats.shortest_indices  # the 5-char doc


def test_length_stats_counts_empty_documents_separately():
    docs = ["hello world", "", "    ", "another"]
    stats = length_stats(docs)
    assert stats.empty_count == 2
    # Empty docs should not be in shortest_indices (they have char_len 0)
    assert all(i not in stats.shortest_indices for i in (1, 2))


def test_length_stats_serialises_to_dict_round_trips_via_json():
    docs = ["short", "medium length doc", "longer doc " * 5]
    stats = length_stats(docs)
    payload = json.dumps(stats.to_dict())
    parsed = json.loads(payload)
    assert parsed["n"] == 3
    assert parsed["empty_count"] == 0
    assert isinstance(parsed["longest_indices"], list)


def test_length_stats_handles_single_document():
    stats = length_stats(["only one document here"])
    assert stats.n == 1
    assert stats.char_median == 22
    assert stats.longest_indices == [0]


# ---------------------------------------------------------------------------
# vocabulary_stats
# ---------------------------------------------------------------------------


def test_vocabulary_stats_empty_returns_zeros():
    stats = vocabulary_stats([])
    assert isinstance(stats, VocabularyStats)
    assert stats.total_tokens == 0
    assert stats.unique_tokens == 0
    assert stats.top_tokens == []


def test_vocabulary_stats_counts_tokens_lowercased():
    stats = vocabulary_stats(["Hello WORLD", "hello there"])
    # tokens: hello, world, hello, there → 4 total, 3 unique
    assert stats.total_tokens == 4
    assert stats.unique_tokens == 3
    assert stats.type_token_ratio == pytest.approx(0.75)
    assert ("hello", 2) in stats.top_tokens


def test_vocabulary_stats_hapax_ratio_high_for_diverse_corpus():
    docs = ["alpha bravo charlie", "delta echo foxtrot"]
    stats = vocabulary_stats(docs)
    # Every token appears once → hapax ratio = 1.0
    assert stats.hapax_ratio == pytest.approx(1.0)


def test_vocabulary_stats_respects_stopwords():
    stats = vocabulary_stats(
        ["the quick brown fox", "the lazy dog the cat"],
        stopwords={"the"},
    )
    top_tokens = dict(stats.top_tokens)
    assert "the" not in top_tokens
    # Tokens before stopwords: 9 (the, quick, brown, fox, the, lazy, dog, the, cat).
    # Three "the" removed → 6 remain.
    assert stats.total_tokens == 6


def test_vocabulary_stats_respects_top_n_cap():
    docs = ["a b c d e f g h i j k l m n o p"]
    stats = vocabulary_stats(docs, top_n=3)
    assert len(stats.top_tokens) == 3


def test_vocabulary_stats_serialises_top_tokens_as_lists():
    stats = vocabulary_stats(["one two two"])
    payload = json.loads(json.dumps(stats.to_dict()))
    assert all(isinstance(item, list) and len(item) == 2 for item in payload["top_tokens"])
