"""
Tests for document-side contribution analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from contributions import (
    SpanContribution,
    TokenContribution,
    contributing_spans,
    contributing_tokens,
)


def _bag_encoder(vocab: list[str]):
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


class TestContributingTokens:
    def test_returns_one_per_token(self):
        encoder = _bag_encoder(["cat", "dog", "fish"])
        out = contributing_tokens("cat dog", "the cat sat", encoder)

        assert len(out) == 3
        assert all(isinstance(c, TokenContribution) for c in out)
        assert [c.token for c in out] == ["the", "cat", "sat"]

    def test_overlap_token_dominates_contribution(self):
        # "cat" is the only document word that overlaps with the query —
        # removing it should crater the score.
        encoder = _bag_encoder(["cat", "dog", "fish"])
        out = contributing_tokens("cat dog", "the cat fish", encoder)
        cat = next(c for c in out if c.token == "cat")
        the = next(c for c in out if c.token == "the")
        assert cat.contribution > the.contribution

    def test_empty_document_returns_empty(self):
        encoder = _bag_encoder(["a"])
        assert contributing_tokens("a", "", encoder) == []

    def test_to_dict_serialises(self):
        encoder = _bag_encoder(["a", "b"])
        out = contributing_tokens("a", "a b", encoder)
        d = out[0].to_dict()
        assert set(d.keys()) == {
            "token",
            "position",
            "contribution",
            "score_with",
            "score_without",
        }


class TestContributingSpans:
    def test_default_window_is_non_overlapping(self):
        encoder = _bag_encoder(["a", "b", "c", "d"])
        out = contributing_spans("a c", "a b c d", encoder, window=2)

        assert all(isinstance(s, SpanContribution) for s in out)
        # Two non-overlapping windows of width 2 cover positions [0,2) and [2,4)
        assert [(s.start, s.end) for s in out] == [(0, 2), (2, 4)]
        assert [s.span for s in out] == ["a b", "c d"]

    def test_overlapping_stride_yields_more_spans(self):
        encoder = _bag_encoder(["a", "b", "c", "d"])
        out = contributing_spans("a c", "a b c d", encoder, window=2, stride=1)
        assert len(out) >= 3
        # Spans must be ordered by start position
        starts = [s.start for s in out]
        assert starts == sorted(starts)

    def test_invalid_window_or_stride_raises(self):
        encoder = _bag_encoder(["a"])
        with pytest.raises(ValueError):
            contributing_spans("q", "doc", encoder, window=0)
        with pytest.raises(ValueError):
            contributing_spans("q", "doc", encoder, window=2, stride=0)

    def test_empty_document_returns_empty(self):
        encoder = _bag_encoder(["a"])
        assert contributing_spans("a", "", encoder, window=3) == []

    def test_window_larger_than_doc_drops_everything(self):
        # When window >= len(tokens), the only span covers the whole doc.
        encoder = _bag_encoder(["x"])
        out = contributing_spans("x", "x x x", encoder, window=10)
        assert len(out) == 1
        assert out[0].start == 0
        assert out[0].end == 3
