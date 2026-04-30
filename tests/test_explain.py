"""
Tests for the leave-one-out query explainer.
"""

from __future__ import annotations

import numpy as np
import pytest

from explain import (
    QueryExplanation,
    TokenAttribution,
    explain_query,
    tokenize,
)


def _bag_encoder(vocab: list[str]):
    """Build a deterministic bag-of-words encoder over ``vocab``.

    The synthetic encoder lets us test the explainer without depending on a
    real model — the leave-one-out arithmetic must hold for any encoder.
    """

    def _enc(texts):
        rows = []
        for text in texts:
            v = np.zeros(len(vocab), dtype=np.float32)
            for tok in text.lower().split():
                if tok in vocab:
                    v[vocab.index(tok)] += 1.0
            rows.append(v)
        return np.asarray(rows, dtype=np.float32)

    return _enc


class TestTokenize:
    def test_basic_words(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert tokenize("Machine, learning! is great.") == [
            "Machine",
            "learning",
            "is",
            "great",
        ]

    def test_keeps_apostrophes_and_hyphens(self):
        # Contractions and hyphenated terms are common in user queries; we
        # keep them as single tokens so attribution stays meaningful.
        assert tokenize("don't break state-of-the-art") == [
            "don't",
            "break",
            "state-of-the-art",
        ]

    def test_keeps_numbers(self):
        assert tokenize("BERT 2018 paper") == ["BERT", "2018", "paper"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestExplainQuery:
    def test_returns_attribution_per_token(self):
        encoder = _bag_encoder(["machine", "learning", "python", "great"])
        exp = explain_query("machine learning python", "machine learning", encoder)

        assert isinstance(exp, QueryExplanation)
        assert len(exp.tokens) == 3
        assert [t.token for t in exp.tokens] == ["machine", "learning", "python"]
        assert all(isinstance(t, TokenAttribution) for t in exp.tokens)

    def test_empty_query_returns_empty_explanation(self):
        encoder = _bag_encoder(["a", "b"])
        exp = explain_query("", "anything", encoder)
        assert exp.tokens == []
        assert exp.base_score == 0.0

    def test_token_with_overlap_has_positive_importance(self):
        # "machine" appears in both query and document — removing it should
        # drop the similarity, giving positive importance.
        encoder = _bag_encoder(["machine", "learning", "python"])
        exp = explain_query("machine python", "machine learning", encoder)
        machine_attr = next(t for t in exp.tokens if t.token == "machine")
        assert machine_attr.importance > 0
        assert machine_attr.score_with > machine_attr.score_without

    def test_noise_token_can_have_negative_importance(self):
        # "fun" is in the vocab but absent from the doc — it dilutes the query
        # vector with a dimension the doc has zero on, so removing it tightens
        # the cosine and importance comes out negative.
        encoder = _bag_encoder(["machine", "learning", "fun"])
        exp = explain_query("machine learning fun", "machine learning", encoder)
        fun_attr = next(t for t in exp.tokens if t.token == "fun")
        assert fun_attr.importance < 0

    def test_base_score_matches_cosine(self):
        encoder = _bag_encoder(["a", "b", "c"])
        exp = explain_query("a b", "a b c", encoder)
        # Manual cosine: query=[1,1,0], doc=[1,1,1] → 2 / (sqrt(2)*sqrt(3))
        expected = 2.0 / (np.sqrt(2) * np.sqrt(3))
        assert exp.base_score == pytest.approx(expected, abs=1e-6)

    def test_uses_precomputed_document_embedding(self):
        # When the doc embedding is supplied we should never re-encode it.
        # Sentinel encoder records every text it sees.
        seen: list[str] = []

        def encoder(texts):
            seen.extend(texts)
            return _bag_encoder(["x", "y"])(texts)

        doc_emb = _bag_encoder(["x", "y"])(["x y"])[0]
        explain_query("x y", "x y", encoder, document_embedding=doc_emb)
        # Only the perturbed queries + the original query should be encoded.
        assert "x y" in seen  # query side still encoded
        assert seen.count("x y") == 1  # doc was not re-encoded

    def test_uses_precomputed_base_score(self):
        # When base_score is supplied we save the original-query encode call.
        seen: list[str] = []

        def encoder(texts):
            seen.extend(texts)
            return _bag_encoder(["a", "b"])(texts)

        explain_query("a b", "a", encoder, base_score=0.7071)
        # Two perturbations + one document encode → 3 calls; original query
        # is *not* in the list.
        assert "a b" not in seen
        assert len(seen) == 3

    def test_top_positive_and_negative_sort(self):
        encoder = _bag_encoder(["good", "bad"])
        exp = explain_query("good bad", "good", encoder)
        pos = exp.top_positive(1)
        neg = exp.top_negative(1)
        assert pos[0].token == "good"
        assert neg[0].token == "bad"

    def test_to_dict_roundtrips_floats(self):
        encoder = _bag_encoder(["a", "b"])
        exp = explain_query("a b", "a", encoder)
        d = exp.to_dict()
        assert d["query"] == "a b"
        assert d["document"] == "a"
        assert "tokens" in d
        assert all(isinstance(t["importance"], float) for t in d["tokens"])
