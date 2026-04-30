from __future__ import annotations

import pytest

from abtest import compare_systems


def test_compare_systems_builds_report_for_shared_metrics():
    report = compare_systems(
        {"ndcg@10": [0.4, 0.6, 0.5], "recall@10": [0.5, 0.5, 1.0]},
        {"ndcg@10": [0.5, 0.7, 0.5], "recall@10": [0.5, 1.0, 1.0]},
        name_a="dense",
        name_b="reranked",
        n_resamples=100,
        seed=11,
    )

    assert report.name_a == "dense"
    assert report.name_b == "reranked"
    assert report.n_queries == 3
    assert [m.metric for m in report.metrics] == ["ndcg@10", "recall@10"]
    assert report.metrics[0].delta == pytest.approx((0.1 + 0.1 + 0.0) / 3)
    assert report.metrics[0].b_wins == 2
    assert report.metrics[0].a_wins == 0
    assert report.metrics[0].ties == 1


def test_report_to_dict_is_json_ready():
    report = compare_systems(
        {"mrr": [0.2, 0.4]},
        {"mrr": [0.2, 0.6]},
        n_resamples=20,
        seed=1,
    )

    payload = report.to_dict()
    assert payload["name_a"] == "A"
    assert payload["name_b"] == "B"
    assert payload["metrics"][0]["metric"] == "mrr"
    assert "paired_test" in payload["metrics"][0]


def test_report_renders_markdown_and_summary_lines():
    report = compare_systems(
        {"mrr": [0.2, 0.4]},
        {"mrr": [0.3, 0.5]},
        name_a="baseline",
        name_b="candidate",
        n_resamples=20,
        seed=2,
    )

    markdown = report.to_markdown()
    assert "# A/B Comparison: baseline vs candidate" in markdown
    assert "| mrr |" in markdown
    assert "Queries: **2**" in markdown

    lines = report.summary_lines()
    assert lines[0] == "A/B Comparison: baseline vs candidate (2 queries)"
    assert lines[1].startswith("- mrr: Δ=")


def test_winner_returns_significant_direction_or_tie():
    report = compare_systems(
        {"mrr": [0.1, 0.1, 0.1, 0.1]},
        {"mrr": [0.9, 0.9, 0.9, 0.9]},
        name_a="old",
        name_b="new",
        n_resamples=50,
        seed=5,
    )

    assert report.winner("mrr") == "new"
    assert report.winner("mrr", alpha=0.0) == "tie"
    with pytest.raises(KeyError, match="No comparison"):
        report.winner("missing")


def test_compare_systems_validates_metric_keys():
    with pytest.raises(ValueError, match="metric keys differ"):
        compare_systems({"mrr": [0.1]}, {"ndcg": [0.1]})


def test_compare_systems_validates_metric_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        compare_systems({"mrr": [0.1, 0.2]}, {"mrr": [0.1]})


def test_compare_systems_validates_shared_query_count_across_metrics():
    with pytest.raises(ValueError, match="expected"):
        compare_systems(
            {"mrr": [0.1, 0.2], "ndcg": [0.1]},
            {"mrr": [0.2, 0.3], "ndcg": [0.2]},
        )
