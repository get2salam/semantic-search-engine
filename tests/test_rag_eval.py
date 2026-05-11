import pytest

from rag_eval import (
    average_precision_at_k,
    expected_reciprocal_rank,
    f1_at_k,
    graded_ndcg_at_k,
    hit_at_k,
    hit_rate_at_k,
    mean_average_precision,
    mean_expected_reciprocal_rank,
    mean_f1_at_k,
    mean_graded_ndcg_at_k,
    mean_ndcg_at_k,
    mean_precision_at_k,
    mean_r_precision,
    mean_rank_biased_precision,
    mean_recall_at_k,
    mean_reciprocal_rank,
    mean_reciprocal_rank_at_k,
    ndcg_at_k,
    precision_at_k,
    r_precision,
    rank_biased_precision,
    recall_at_k,
    reciprocal_rank,
    reciprocal_rank_at_k,
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


def test_reciprocal_rank_at_k_bounds_first_relevant_to_cutoff():
    # Hit at rank 2 within k=3 -> 1/2; same hit excluded at k=1 -> 0.0.
    assert reciprocal_rank_at_k(["x", "a", "b"], {"a"}, 3) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x", "a", "b"], {"a"}, 1) == 0.0
    # Matches reciprocal_rank when the relevant doc is within the cutoff.
    assert reciprocal_rank_at_k(["a", "b"], {"b"}, 5) == pytest.approx(0.5)
    # Empty relevance collapses to zero rather than dividing by zero downstream.
    assert reciprocal_rank_at_k(["a"], set(), 3) == 0.0
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_at_k(["a"], {"a"}, 0)


def test_mean_reciprocal_rank_at_k_averages_and_validates_alignment():
    runs = [["a", "b"], ["x", "y", "b"]]
    qrels = [{"a"}, {"b"}]

    # Per-query RR@2 -> 1.0 and 0.0; mean = 0.5.
    assert mean_reciprocal_rank_at_k(runs, qrels, 2) == pytest.approx(0.5)
    # Raising k surfaces the deep hit: 1.0 and 1/3.
    assert mean_reciprocal_rank_at_k(runs, qrels, 3) == pytest.approx((1.0 + 1 / 3) / 2)
    assert mean_reciprocal_rank_at_k([], [], 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank_at_k([["a"]], [{"a"}, {"b"}], 2)


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


def test_r_precision_uses_relevance_set_size_as_cutoff():
    # |relevant|=2, so cutoff=2: top-2 has one hit out of two relevant.
    assert r_precision(["a", "x", "b"], {"a", "b"}) == pytest.approx(0.5)
    # Perfect ordering at the adaptive cutoff yields 1.0.
    assert r_precision(["a", "b", "c"], {"a", "b"}) == pytest.approx(1.0)
    # Empty relevance collapses to zero rather than dividing by zero.
    assert r_precision(["a"], set()) == 0.0


def test_mean_r_precision_averages_and_validates_alignment():
    runs = [["a", "x", "b"], ["c", "d"]]
    qrels = [{"a", "b"}, {"c", "d"}]

    assert mean_r_precision(runs, qrels) == pytest.approx((0.5 + 1.0) / 2)
    with pytest.raises(ValueError, match="same length"):
        mean_r_precision([["a"]], [{"a"}, {"b"}])


def test_mean_precision_and_recall_at_k_average_across_queries():
    runs = [["a", "b", "c"], ["x", "b", "y"]]
    qrels = [{"a", "b"}, {"a", "b"}]

    # Per-query precision@3 = 2/3 and 1/3 -> mean = 0.5.
    assert mean_precision_at_k(runs, qrels, 3) == pytest.approx(0.5)
    # Per-query recall@3 = 1.0 and 0.5 -> mean = 0.75.
    assert mean_recall_at_k(runs, qrels, 3) == pytest.approx(0.75)
    assert mean_precision_at_k([], [], 3) == 0.0
    assert mean_recall_at_k([], [], 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_precision_at_k([["a"]], [{"a"}, {"b"}], 2)
    with pytest.raises(ValueError, match="same length"):
        mean_recall_at_k([["a"]], [{"a"}, {"b"}], 2)


def test_mean_average_precision_validates_alignment():
    runs = [["a", "b"], ["x", "b", "a"]]
    qrels = [{"a"}, {"a", "b"}]

    assert mean_average_precision(runs, qrels, 3) == pytest.approx((1.0 + (1 / 2 + 2 / 3) / 2) / 2)
    with pytest.raises(ValueError, match="same length"):
        mean_average_precision([["a"]], [{"a"}, {"b"}], 3)


def test_mean_ndcg_at_k_averages_and_validates_alignment():
    runs = [["a", "b", "c"], ["c", "a", "b"]]
    qrels = [{"a", "b"}, {"a", "b"}]
    expected = (ndcg_at_k(runs[0], qrels[0], 3) + ndcg_at_k(runs[1], qrels[1], 3)) / 2

    assert mean_ndcg_at_k(runs, qrels, 3) == pytest.approx(expected)
    # Empty relevance sets contribute zero rather than dividing by zero.
    assert mean_ndcg_at_k([["a"], ["b"]], [set(), {"b"}], 2) == pytest.approx(0.5)
    assert mean_ndcg_at_k([], [], 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_ndcg_at_k([["a"]], [{"a"}, {"b"}], 3)


def test_graded_ndcg_at_k_rewards_higher_grades_first():
    grades = {"a": 3.0, "b": 1.0}

    assert graded_ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)
    swapped = graded_ndcg_at_k(["b", "a", "c"], grades, 3)
    assert 0.0 < swapped < 1.0
    # Non-positive grades are treated as irrelevant; collapses to zero.
    assert graded_ndcg_at_k(["a"], {"a": 0.0, "b": -1.0}, 1) == 0.0
    with pytest.raises(ValueError, match="k"):
        graded_ndcg_at_k(["a"], {"a": 1.0}, 0)


def test_mean_graded_ndcg_at_k_averages_and_validates_alignment():
    runs = [["a", "b"], ["b", "a"]]
    qrels = [{"a": 3.0, "b": 1.0}, {"a": 3.0, "b": 1.0}]
    expected = (
        graded_ndcg_at_k(runs[0], qrels[0], 2) + graded_ndcg_at_k(runs[1], qrels[1], 2)
    ) / 2

    assert mean_graded_ndcg_at_k(runs, qrels, 2) == pytest.approx(expected)
    assert mean_graded_ndcg_at_k([], [], 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_graded_ndcg_at_k([["a"]], [{"a": 1.0}, {"b": 1.0}], 2)


def test_rank_biased_precision_weights_early_hits_more_heavily():
    # p=0.5 with hits at every rank: (1-0.5)*(1 + 0.5 + 0.25) = 0.875.
    assert rank_biased_precision(["a", "b", "c"], {"a", "b", "c"}, 0.5, 3) == pytest.approx(0.875)
    # Same hit at rank 1 vs rank 3 -> 0.5 vs 0.125.
    assert rank_biased_precision(["a", "x", "y"], {"a"}, 0.5, 3) == pytest.approx(0.5)
    assert rank_biased_precision(["x", "y", "a"], {"a"}, 0.5, 3) == pytest.approx(0.125)
    # Empty relevance collapses to zero rather than dividing by zero downstream.
    assert rank_biased_precision(["a"], set(), 0.5, 3) == 0.0


def test_rank_biased_precision_validates_inputs():
    with pytest.raises(ValueError, match="k"):
        rank_biased_precision(["a"], {"a"}, 0.5, 0)
    with pytest.raises(ValueError, match="persistence"):
        rank_biased_precision(["a"], {"a"}, 0.0, 3)
    with pytest.raises(ValueError, match="persistence"):
        rank_biased_precision(["a"], {"a"}, 1.0, 3)


def test_mean_rank_biased_precision_averages_and_validates_alignment():
    runs = [["a", "b"], ["x", "a"]]
    qrels = [{"a"}, {"a"}]

    # Per-query RBP@2 with p=0.5 -> 0.5 and 0.25; mean = 0.375.
    assert mean_rank_biased_precision(runs, qrels, 0.5, 2) == pytest.approx(0.375)
    assert mean_rank_biased_precision([], [], 0.5, 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_rank_biased_precision([["a"]], [{"a"}, {"b"}], 0.5, 2)


def test_expected_reciprocal_rank_uses_graded_stopping_probabilities():
    # grade=2 at rank 1 then grade=1 at rank 2, g_max=2 -> stops 0.75 and 0.25.
    # Score = 0.75/1 + (1 - 0.75) * 0.25 / 2 = 0.78125.
    grades = {"a": 2.0, "b": 1.0}
    assert expected_reciprocal_rank(["a", "b"], grades, 2) == pytest.approx(0.78125)
    # Demoting the higher-grade doc strictly lowers ERR.
    swapped = expected_reciprocal_rank(["b", "a"], grades, 2)
    assert swapped < 0.78125
    # Uniform grades with single relevant doc: rank 1 -> 0.5, rank 3 -> 1/6.
    assert expected_reciprocal_rank(["a", "x", "y"], {"a": 1.0}, 3) == pytest.approx(0.5)
    assert expected_reciprocal_rank(["x", "y", "a"], {"a": 1.0}, 3) == pytest.approx(1 / 6)


def test_expected_reciprocal_rank_validates_inputs_and_empty_grades():
    assert expected_reciprocal_rank(["a"], {"a": 0.0, "b": -1.0}, 1) == 0.0
    with pytest.raises(ValueError, match="k"):
        expected_reciprocal_rank(["a"], {"a": 1.0}, 0)


def test_mean_expected_reciprocal_rank_averages_and_validates_alignment():
    runs = [["a", "x", "y"], ["x", "y", "a"]]
    qrels = [{"a": 1.0}, {"a": 1.0}]
    # Per-query ERR@3 -> 0.5 and 1/6; mean = (0.5 + 1/6) / 2.
    assert mean_expected_reciprocal_rank(runs, qrels, 3) == pytest.approx((0.5 + 1 / 6) / 2)
    assert mean_expected_reciprocal_rank([], [], 3) == 0.0
    with pytest.raises(ValueError, match="same length"):
        mean_expected_reciprocal_rank([["a"]], [{"a": 1.0}, {"b": 1.0}], 2)
