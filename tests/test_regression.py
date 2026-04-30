from __future__ import annotations

import pytest

from regression import compare_ranked_snapshots


def test_compare_ranked_snapshots_summarises_ranking_movement():
    report = compare_ranked_snapshots(
        {"q1": ["d1", "d2", "d3"], "q2": ["a", "b"]},
        {"q1": ["d2", "d3", "d4"], "q2": ["a", "c"]},
        k=3,
    )

    assert report.n_queries == 2
    assert report.changed_top_count == 1
    assert report.queries[0].query_id == "q1"
    assert report.queries[0].baseline_top == "d1"
    assert report.queries[0].current_top == "d2"
    assert report.queries[0].overlap_at_k == 2
    assert report.queries[0].jaccard_at_k == pytest.approx(0.5)
    assert report.queries[0].dropped == ["d1"]
    assert report.queries[0].added == ["d4"]


def test_regression_report_to_dict_includes_aggregates():
    report = compare_ranked_snapshots({"q": ["a"]}, {"q": ["b"]}, k=1)

    payload = report.to_dict()
    assert payload["k"] == 1
    assert payload["n_queries"] == 1
    assert payload["changed_top_count"] == 1
    assert payload["mean_jaccard_at_k"] == 0.0
    assert payload["queries"][0]["changed_top"] is True


def test_regression_report_summary_lines_are_review_friendly():
    report = compare_ranked_snapshots({"q": ["a", "b"]}, {"q": ["a", "c"]}, k=2)

    lines = report.summary_lines()
    assert lines[0] == "Regression report (1 queries, top-2)"
    assert "changed top result" in lines[1]
    assert "mean jaccard@2" in lines[2]
    assert "q:" in lines[3]


def test_compare_ranked_snapshots_handles_empty_rankings():
    report = compare_ranked_snapshots({"q": []}, {"q": []}, k=5)

    assert report.queries[0].baseline_top is None
    assert report.queries[0].current_top is None
    assert report.queries[0].jaccard_at_k == 1.0


def test_compare_ranked_snapshots_validates_inputs():
    with pytest.raises(ValueError, match="k"):
        compare_ranked_snapshots({"q": ["a"]}, {"q": ["a"]}, k=0)
    with pytest.raises(ValueError, match="query ids differ"):
        compare_ranked_snapshots({"q1": ["a"]}, {"q2": ["a"]})
