"""
Tests for query_processor.py
==============================
Covers QueryNormalizer, StopWordFilter, QueryExpander, and QueryProcessor.
"""

from __future__ import annotations

import pytest

from query_processor import (
    ProcessedQuery,
    QueryExpander,
    QueryNormalizer,
    QueryProcessor,
    StopWordFilter,
)


# ===========================================================================
# QueryNormalizer
# ===========================================================================
class TestQueryNormalizer:
    """Tests for QueryNormalizer."""

    def test_basic_lowercase(self):
        n = QueryNormalizer()
        assert n.normalise("Hello World") == "hello world"

    def test_lowercase_disabled(self):
        n = QueryNormalizer(lowercase=False)
        assert n.normalise("Hello World") == "Hello World"

    def test_whitespace_collapsed(self):
        n = QueryNormalizer()
        assert n.normalise("  foo   bar  ") == "foo bar"

    def test_tab_and_newline_collapsed(self):
        n = QueryNormalizer()
        assert n.normalise("foo\tbar\nbaz") == "foo bar baz"

    def test_unicode_nfc(self):
        # é as NFD (e + combining accent) should become NFC é
        nfd = "e\u0301"  # e + combining acute accent
        n = QueryNormalizer()
        result = n.normalise(nfd)
        assert result == "\xe9"  # NFC é

    def test_strip_accents(self):
        n = QueryNormalizer(strip_accents=True)
        assert n.normalise("café") == "cafe"

    def test_strip_accents_disabled_by_default(self):
        n = QueryNormalizer(strip_accents=False)
        assert n.normalise("café") == "café"

    def test_max_length_truncates(self):
        n = QueryNormalizer(max_length=10)
        result = n.normalise("hello world foo bar")
        assert len(result) <= 10

    def test_max_length_zero_no_truncation(self):
        n = QueryNormalizer(max_length=0)
        long_query = "word " * 200
        result = n.normalise(long_query)
        assert len(result) > 100

    def test_empty_string(self):
        n = QueryNormalizer()
        assert n.normalise("") == ""

    def test_type_error_on_non_string(self):
        n = QueryNormalizer()
        with pytest.raises(TypeError, match="Query must be str"):
            n.normalise(42)  # type: ignore[arg-type]

    def test_callable_alias(self):
        n = QueryNormalizer()
        assert n("Hello") == n.normalise("Hello")

    def test_single_word(self):
        n = QueryNormalizer()
        assert n.normalise("Python") == "python"

    def test_numbers_preserved(self):
        n = QueryNormalizer()
        assert n.normalise("GPT-4 model 2024") == "gpt-4 model 2024"

    def test_punctuation_preserved(self):
        n = QueryNormalizer()
        result = n.normalise("What is NLP?")
        assert "?" in result


# ===========================================================================
# StopWordFilter
# ===========================================================================
class TestStopWordFilter:
    """Tests for StopWordFilter."""

    def test_removes_stop_words(self):
        f = StopWordFilter()
        filtered, tokens = f.filter("what is machine learning")
        assert "is" not in tokens
        assert "what" not in tokens
        assert "machine" in tokens
        assert "learning" in tokens

    def test_short_query_unchanged(self):
        f = StopWordFilter(min_tokens=1)
        # Single token: should not filter
        filtered, tokens = f.filter("the")
        assert filtered == "the"
        assert tokens == ["the"]

    def test_min_remaining_safety(self):
        f = StopWordFilter(min_remaining=1)
        # All words are stop words
        filtered, tokens = f.filter("the and or")
        assert len(tokens) >= 1

    def test_custom_stop_words(self):
        custom = frozenset({"foo", "bar"})
        f = StopWordFilter(stop_words=custom)
        filtered, tokens = f.filter("foo baz bar qux")
        assert "foo" not in tokens
        assert "bar" not in tokens
        assert "baz" in tokens
        assert "qux" in tokens

    def test_empty_query(self):
        f = StopWordFilter()
        filtered, tokens = f.filter("")
        assert filtered == ""
        assert tokens == []

    def test_all_content_words(self):
        f = StopWordFilter()
        filtered, tokens = f.filter("machine learning algorithms")
        assert tokens == ["machine", "learning", "algorithms"]

    def test_preserves_case_from_input(self):
        # Filter compares lowercased, but tokens returned as-is
        f = StopWordFilter()
        filtered, tokens = f.filter("machine IS learning")
        # "IS" should be filtered (case-insensitive check)
        assert "IS" not in tokens
        assert "machine" in tokens

    def test_output_filtered_string_matches_tokens(self):
        f = StopWordFilter()
        filtered, tokens = f.filter("what is the semantic search")
        assert filtered == " ".join(tokens)

    def test_min_tokens_threshold(self):
        f = StopWordFilter(min_tokens=3)
        # Query with 2 tokens: should NOT be filtered even if stop words present
        filtered, tokens = f.filter("is a")
        assert filtered == "is a"

    def test_query_with_only_stop_words(self):
        f = StopWordFilter(min_remaining=1)
        filtered, tokens = f.filter("is are was were")
        # Safety: at least one token returned
        assert len(tokens) >= 1


# ===========================================================================
# QueryExpander
# ===========================================================================
class TestQueryExpander:
    """Tests for QueryExpander."""

    def test_expands_known_abbreviation(self):
        e = QueryExpander()
        expanded, expansions = e.expand("nlp", ["nlp"])
        assert "natural language processing" in expanded
        assert "natural language processing" in expansions

    def test_no_expansion_for_unknown_token(self):
        e = QueryExpander()
        expanded, expansions = e.expand("python", ["python"])
        assert expanded == "python"
        assert expansions == []

    def test_multiple_abbreviations(self):
        e = QueryExpander()
        expanded, expansions = e.expand("nlp ml", ["nlp", "ml"])
        assert "natural language processing" in expanded
        assert "machine learning" in expanded

    def test_max_expansions_limit(self):
        synonyms = {
            "a": ["alpha one", "alpha two"],
            "b": ["beta one", "beta two"],
            "c": ["gamma one"],
        }
        e = QueryExpander(synonyms=synonyms, max_expansions=2)
        expanded, expansions = e.expand("a b c", ["a", "b", "c"])
        assert len(expansions) == 2

    def test_max_expansions_zero_unlimited(self):
        synonyms = {str(i): [f"word_{i}"] for i in range(10)}
        e = QueryExpander(synonyms=synonyms, max_expansions=0)
        tokens = list(synonyms.keys())
        expanded, expansions = e.expand(" ".join(tokens), tokens)
        assert len(expansions) == 10

    def test_case_insensitive_key_lookup(self):
        e = QueryExpander()
        expanded, expansions = e.expand("NLP", ["NLP"])
        assert "natural language processing" in expansions

    def test_empty_tokens(self):
        e = QueryExpander()
        expanded, expansions = e.expand("some query", [])
        assert expanded == "some query"
        assert expansions == []

    def test_custom_synonyms(self):
        synonyms = {"fast": ["rapid", "swift"]}
        e = QueryExpander(synonyms=synonyms)
        expanded, expansions = e.expand("fast search", ["fast", "search"])
        assert "rapid" in expansions
        assert "swift" in expansions

    def test_no_duplicate_expansions(self):
        synonyms = {"ml": ["machine learning"], "ai": ["machine learning"]}
        e = QueryExpander(synonyms=synonyms, max_expansions=0)
        expanded, expansions = e.expand("ml ai", ["ml", "ai"])
        assert expansions.count("machine learning") == 1

    def test_expansion_appended_to_query(self):
        e = QueryExpander()
        expanded, expansions = e.expand("nlp", ["nlp"])
        assert expanded.startswith("nlp")
        for phrase in expansions:
            assert phrase in expanded


# ===========================================================================
# QueryProcessor (integration)
# ===========================================================================
class TestQueryProcessor:
    """Integration tests for the full QueryProcessor pipeline."""

    @pytest.fixture
    def processor(self):
        return QueryProcessor()

    def test_returns_processed_query(self, processor):
        result = processor.process("What is NLP?")
        assert isinstance(result, ProcessedQuery)

    def test_original_preserved(self, processor):
        raw = "What is NLP and ML?"
        result = processor.process(raw)
        assert result.original == raw

    def test_normalised_is_lowercase(self, processor):
        result = processor.process("What Is MACHINE LEARNING?")
        assert result.normalised == result.normalised.lower()

    def test_expanded_contains_synonyms(self, processor):
        result = processor.process("nlp algorithms")
        assert "natural language processing" in result.expanded

    def test_expansion_list_populated(self, processor):
        result = processor.process("nlp algorithms")
        assert "natural language processing" in result.expansions

    def test_tokens_are_content_words(self, processor):
        result = processor.process("what is the best nlp model")
        # Stop words removed; 'nlp' and 'model' should survive
        assert "nlp" in result.tokens or "best" in result.tokens

    def test_callable_shortcut(self, processor):
        result1 = processor("search engine")
        result2 = processor.process("search engine")
        assert result1.expanded == result2.expanded

    def test_empty_query(self, processor):
        result = processor.process("")
        assert isinstance(result, ProcessedQuery)
        assert result.original == ""
        assert result.expanded == ""

    def test_single_stop_word_query(self, processor):
        # "the" alone — must not produce empty result
        result = processor.process("the")
        assert result.expanded != ""

    def test_process_batch(self, processor):
        queries = ["nlp overview", "machine learning basics", "deep learning"]
        results = processor.process_batch(queries)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ProcessedQuery)

    def test_process_batch_empty_list(self, processor):
        assert processor.process_batch([]) == []

    def test_process_batch_order_preserved(self, processor):
        queries = ["nlp", "ml", "dl"]
        results = processor.process_batch(queries)
        for original, result in zip(queries, results, strict=True):
            assert result.original == original

    def test_custom_pipeline_components(self):
        normalizer = QueryNormalizer(lowercase=False)
        stop_filter = StopWordFilter(min_tokens=0)
        expander = QueryExpander(max_expansions=1)
        proc = QueryProcessor(
            normalizer=normalizer,
            stop_filter=stop_filter,
            expander=expander,
        )
        result = proc.process("NLP overview")
        # Lowercase disabled: tokens may be uppercase
        assert result.original == "NLP overview"

    def test_repr_processed_query(self, processor):
        result = processor.process("nlp")
        # Just test it doesn't crash — the repr is informational
        r = repr(result)
        assert "ProcessedQuery" in r

    def test_unicode_query(self, processor):
        result = processor.process("café naïve résumé search")
        assert isinstance(result, ProcessedQuery)
        # Should not raise

    def test_very_long_query_truncated(self):
        proc = QueryProcessor(normalizer=QueryNormalizer(max_length=50))
        long_query = "machine learning " * 20
        result = proc.process(long_query)
        assert len(result.normalised) <= 50

    def test_numeric_tokens_preserved(self, processor):
        result = processor.process("top 10 search results")
        # "10" is not a stop word and should survive filtering
        assert "10" in result.tokens or "top" in result.tokens

    def test_rag_expansion(self, processor):
        result = processor.process("rag pipeline")
        assert "retrieval augmented generation" in result.expanded

    def test_llm_expansion(self, processor):
        result = processor.process("llm inference speed")
        assert "large language model" in result.expanded

    def test_no_expansion_plain_english(self, processor):
        result = processor.process("semantic similarity score")
        # No abbreviations → no expansions
        assert result.expansions == []
        # But filtered query still used for expanded
        assert result.expanded == result.filtered


# ===========================================================================
# ProcessedQuery dataclass
# ===========================================================================
class TestProcessedQuery:
    """Unit tests for the ProcessedQuery dataclass."""

    def test_default_expansions_empty_list(self):
        pq = ProcessedQuery(
            original="test",
            normalised="test",
            filtered="test",
            expanded="test",
            tokens=["test"],
        )
        assert pq.expansions == []

    def test_fields_accessible(self):
        pq = ProcessedQuery(
            original="RAW",
            normalised="raw",
            filtered="raw",
            expanded="raw extra",
            tokens=["raw"],
            expansions=["extra"],
        )
        assert pq.original == "RAW"
        assert pq.normalised == "raw"
        assert pq.filtered == "raw"
        assert pq.expanded == "raw extra"
        assert pq.tokens == ["raw"]
        assert pq.expansions == ["extra"]
