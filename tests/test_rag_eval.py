import pytest

from rag_eval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_and_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "c", "d"}

    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)


def test_reciprocal_rank_returns_first_relevant_position():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_mean_reciprocal_rank_validates_alignment():
    assert mean_reciprocal_rank([["a"], ["x", "b"]], [{"a"}, {"b"}]) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


def test_ndcg_at_k_rewards_higher_ranked_relevant_results():
    perfect = ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3)
    swapped = ndcg_at_k(["c", "a", "b"], {"a", "b"}, 3)

    assert perfect == pytest.approx(1.0)
    assert 0.0 < swapped < perfect


def test_ndcg_at_k_handles_empty_relevant_and_invalid_k():
    assert ndcg_at_k(["a"], set(), 5) == 0.0
    with pytest.raises(ValueError, match="k"):
        ndcg_at_k(["a"], {"a"}, 0)
