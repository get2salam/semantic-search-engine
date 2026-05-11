import pytest

from rag_eval import (
    average_precision_at_k,
    f1_at_k,
    hit_at_k,
    hit_rate_at_k,
    mean_average_precision,
    mean_f1_at_k,
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


def test_average_precision_rewards_early_relevant_hits():
    perfect = average_precision_at_k(["a", "b", "c"], {"a", "b"}, 3)
    delayed = average_precision_at_k(["x", "a", "b"], {"a", "b"}, 3)

    assert perfect == pytest.approx(1.0)
    assert delayed == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_hit_at_k_flags_any_relevant_within_topk():
    assert hit_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert hit_at_k(["a", "b", "c"], {"c"}, 2) == 0.0
    assert hit_at_k(["a"], set(), 5) == 0.0
    with pytest.raises(ValueError, match="k"):
        hit_at_k(["a"], {"a"}, 0)


def test_hit_rate_at_k_averages_across_queries():
    runs = [["a", "b"], ["x", "y"], ["m", "n"]]
    qrels = [{"a"}, {"z"}, {"n"}]

    assert hit_rate_at_k(runs, qrels, 2) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="same length"):
        hit_rate_at_k([["a"]], [{"a"}, {"b"}], 2)


def test_f1_at_k_balances_precision_and_recall():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "b"}

    # precision@4 = 0.5, recall@4 = 1.0 -> F1 = 2/3
    assert f1_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    # No relevant in top-k collapses to zero rather than NaN.
    assert f1_at_k(["x", "y"], {"a"}, 2) == 0.0
    assert f1_at_k(["a"], set(), 1) == 0.0
    with pytest.raises(ValueError, match="k"):
        f1_at_k(["a"], {"a"}, 0)


def test_mean_f1_at_k_averages_and_validates_alignment():
    runs = [["a", "b"], ["x", "y"]]
    qrels = [{"a", "b"}, {"a"}]

    assert mean_f1_at_k(runs, qrels, 2) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="same length"):
        mean_f1_at_k([["a"]], [{"a"}, {"b"}], 2)


def test_mean_average_precision_validates_alignment():
    runs = [["a", "b"], ["x", "b", "a"]]
    qrels = [{"a"}, {"a", "b"}]

    assert mean_average_precision(runs, qrels, 3) == pytest.approx((1.0 + (1 / 2 + 2 / 3) / 2) / 2)
    with pytest.raises(ValueError, match="same length"):
        mean_average_precision([["a"]], [{"a"}, {"b"}], 3)
