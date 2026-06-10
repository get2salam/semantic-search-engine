import pytest

from rag_fusion import reciprocal_rank_fusion, weighted_score_fusion


def test_reciprocal_rank_fusion_rewards_consensus():
    fused = reciprocal_rank_fusion(
        [
            [("a", 0.9), ("b", 0.8), ("c", 0.1)],
            [("b", 0.7), ("a", 0.6), ("d", 0.4)],
        ]
    )

    assert fused[0][0] in {"a", "b"}
    assert {doc_id for doc_id, _ in fused} == {"a", "b", "c", "d"}


def test_reciprocal_rank_fusion_supports_weights_and_top_k():
    fused = reciprocal_rank_fusion(
        [["dense-first", "shared"], ["lexical-first", "shared"]],
        weights=[2.0, 1.0],
        top_k=1,
    )

    assert fused == [("shared", pytest.approx((2.0 / 62) + (1.0 / 62)))]


def test_reciprocal_rank_fusion_rejects_bad_weight_count():
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


def test_reciprocal_rank_fusion_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, -0.5])


def test_reciprocal_rank_fusion_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion([["a", "b"]], top_k=-1)


def test_reciprocal_rank_fusion_zero_weight_drops_run():
    fused = reciprocal_rank_fusion(
        [["only-in-muted"], ["winner"]],
        weights=[0.0, 1.0],
    )

    assert fused == [("winner", pytest.approx(1.0 / 61))]


def test_weighted_score_fusion_uses_score_magnitudes():
    # Without normalization, raw scores accumulate: "b" wins on summed
    # confidence even though "a" leads on rank in run 1.
    fused = weighted_score_fusion(
        [
            [("a", 0.55), ("b", 0.50)],
            [("b", 0.95), ("a", 0.05)],
        ],
        normalize=False,
    )

    assert fused[0][0] == "b"
    assert dict(fused) == {"a": pytest.approx(0.60), "b": pytest.approx(1.45)}


def test_weighted_score_fusion_normalizes_heterogeneous_scales():
    # Run A is on a BM25-like unbounded scale; run B is cosine in [-1, 1].
    # Without normalization the BM25 run would dominate; with normalization
    # both contribute fairly.
    fused = weighted_score_fusion(
        [
            [("a", 12.0), ("b", 4.0)],
            [("b", 0.95), ("a", 0.10)],
        ],
        normalize=True,
    )
    scores = dict(fused)

    # After per-run min-max: A -> {a:1, b:0}; B -> {b:1, a:0}. Sum is 1.0 each,
    # so order is deterministic by doc_id ascending tiebreak.
    assert scores["a"] == pytest.approx(1.0)
    assert scores["b"] == pytest.approx(1.0)
    assert [doc for doc, _ in fused] == ["a", "b"]


def test_weighted_score_fusion_handles_equal_scores_within_run():
    # All scores equal in a run -> treat as pure inclusion (1.0 each) so the
    # run still casts a vote without collapsing to NaN via 0/0 normalization.
    fused = weighted_score_fusion(
        [
            [("a", 0.5), ("b", 0.5)],
            [("a", 0.9)],
        ]
    )

    assert dict(fused) == {"a": pytest.approx(2.0), "b": pytest.approx(1.0)}


def test_weighted_score_fusion_respects_weights_and_top_k():
    fused = weighted_score_fusion(
        [
            [("a", 1.0), ("b", 0.0)],
            [("b", 1.0), ("a", 0.0)],
        ],
        weights=[3.0, 1.0],
        top_k=1,
        normalize=False,
    )

    assert fused == [("a", pytest.approx(3.0))]


def test_weighted_score_fusion_default_score_for_id_only_hits():
    # The lexical run is ID-only; it should still contribute via default_score.
    fused = weighted_score_fusion(
        [
            [("a", 0.9), ("b", 0.1)],
            ["b", "c"],
        ],
        normalize=False,
        default_score=0.5,
    )

    assert dict(fused) == {
        "a": pytest.approx(0.9),
        "b": pytest.approx(0.1 + 0.5),
        "c": pytest.approx(0.5),
    }


def test_weighted_score_fusion_rejects_negative_weight():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_score_fusion([[("a", 1.0)]], weights=[-1.0])


def test_weighted_score_fusion_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        weighted_score_fusion([[("a", 1.0)]], top_k=-2)
