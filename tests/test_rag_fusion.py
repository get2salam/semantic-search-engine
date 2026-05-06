import pytest

from rag_fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_rewards_consensus():
    fused = reciprocal_rank_fusion([
        [("a", 0.9), ("b", 0.8), ("c", 0.1)],
        [("b", 0.7), ("a", 0.6), ("d", 0.4)],
    ])

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
