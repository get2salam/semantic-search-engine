"""
Tests for the ExplainedResult facade.
"""

from __future__ import annotations

import numpy as np
import pytest

from explain_result import ExplainedResult, explain_result, explain_results


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


class TestExplainResult:
    def test_returns_explained_result(self):
        encoder = _bag_encoder(["a", "b"])
        result = explain_result("a", "a b", encoder)
        assert isinstance(result, ExplainedResult)
        assert result.rank == 1
        assert result.query == "a"
        assert result.document == "a b"
        assert result.n_query_tokens == 1
        assert result.n_document_tokens == 2

    def test_short_doc_picks_token_mode(self):
        encoder = _bag_encoder(["a", "b"])
        result = explain_result("a", "a b", encoder)
        assert result.contribution_mode == "tokens"
        assert result.document_tokens
        assert not result.document_spans

    def test_long_doc_picks_span_mode_by_default(self):
        # 40-token document — auto mode should switch to spans.
        encoder = _bag_encoder(["a", "b"])
        long_doc = " ".join(["a"] * 40)
        result = explain_result("a", long_doc, encoder)
        assert result.contribution_mode == "spans"
        assert result.document_spans
        assert not result.document_tokens

    def test_explicit_token_mode_overrides(self):
        encoder = _bag_encoder(["a"])
        long_doc = " ".join(["a"] * 50)
        result = explain_result("a", long_doc, encoder, contribution_mode="tokens")
        assert result.contribution_mode == "tokens"
        # 50 tokens → 50 contributions
        assert len(result.document_tokens) == 50

    def test_invalid_mode_raises(self):
        encoder = _bag_encoder(["a"])
        with pytest.raises(ValueError):
            explain_result("a", "a", encoder, contribution_mode="bogus")

    def test_to_dict_is_json_clean(self):
        import json

        encoder = _bag_encoder(["a", "b"])
        result = explain_result("a b", "a b", encoder)
        # Round-trip through json — fails if any field isn't serialisable.
        s = json.dumps(result.to_dict())
        loaded = json.loads(s)
        assert loaded["query"] == "a b"
        assert loaded["contribution_mode"] in {"tokens", "spans"}

    def test_passes_score_through_when_provided(self):
        encoder = _bag_encoder(["a"])
        result = explain_result("a", "a", encoder, score=0.42)
        # base_score short-circuit means our supplied score becomes the
        # base for both LOO passes.
        assert result.score == pytest.approx(0.42, abs=1e-6)


class TestExplainResults:
    def test_handles_tuples(self):
        encoder = _bag_encoder(["a", "b"])
        out = explain_results("a", [("a b", 0.9), ("b a", 0.8)], encoder)
        assert len(out) == 2
        assert out[0].rank == 1
        assert out[1].rank == 2

    def test_handles_plain_strings(self):
        encoder = _bag_encoder(["a"])
        out = explain_results("a", ["a", "a a"], encoder)
        assert len(out) == 2
        assert all(isinstance(r, ExplainedResult) for r in out)

    def test_empty_results_returns_empty_list(self):
        encoder = _bag_encoder(["a"])
        assert explain_results("a", [], encoder) == []
