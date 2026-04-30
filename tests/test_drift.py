from __future__ import annotations

import pytest

from drift import population_stability_index


def test_population_stability_index_reports_low_drift_for_same_distribution():
    report = population_stability_index([0.1, 0.2, 0.8, 0.9], [0.1, 0.2, 0.8, 0.9], n_bins=2)

    assert report.psi == pytest.approx(0.0)
    assert report.severity == "low"
    assert report.n_baseline == 4
    assert report.n_current == 4
    assert len(report.bins) == 2


def test_population_stability_index_detects_shifted_distribution():
    report = population_stability_index([0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0], n_bins=2)

    assert report.psi > 0.25
    assert report.severity == "high"
    assert report.bins[0].baseline_count > report.bins[0].current_count
    assert report.bins[1].current_count > report.bins[1].baseline_count


def test_drift_report_to_dict_and_summary_lines():
    report = population_stability_index([1.0, 1.0], [1.0, 2.0], n_bins=2)

    payload = report.to_dict()
    assert payload["severity"] in {"low", "moderate", "high"}
    assert payload["bins"][0]["baseline_count"] == 2

    lines = report.summary_lines()
    assert lines[0].startswith("Drift report: PSI=")
    assert any("current examples" in line for line in lines)


def test_population_stability_index_validates_arguments():
    with pytest.raises(ValueError, match="n_bins"):
        population_stability_index([0.1], [0.2], n_bins=0)
    with pytest.raises(ValueError, match="epsilon"):
        population_stability_index([0.1], [0.2], epsilon=0.0)
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([], [0.2])
