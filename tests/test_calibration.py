from __future__ import annotations

import pytest

from calibration import calibration_report


def test_calibration_report_computes_bins_and_errors():
    report = calibration_report([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], n_bins=2)

    assert report.n == 4
    assert report.positive_rate == pytest.approx(0.5)
    assert report.brier_score == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert report.expected_calibration_error == pytest.approx(0.15)
    assert report.max_calibration_error == pytest.approx(0.15)
    assert len(report.bins) == 2
    assert report.bins[0].count == 2
    assert report.bins[0].positive_rate == 0.0
    assert report.bins[1].positive_rate == 1.0


def test_calibration_report_handles_empty_input():
    report = calibration_report([], [], n_bins=3)

    assert report.to_dict() == {
        "n": 0,
        "positive_rate": 0.0,
        "brier_score": 0.0,
        "expected_calibration_error": 0.0,
        "max_calibration_error": 0.0,
        "bins": [],
    }


def test_calibration_bin_to_dict_includes_gap():
    report = calibration_report([0.8], [1], n_bins=2)

    payload = report.bins[0].to_dict()
    assert payload["avg_score"] == 0.8
    assert payload["positive_rate"] == 1.0
    assert payload["gap"] == pytest.approx(-0.2)


def test_calibration_report_summary_lines_are_terminal_friendly():
    report = calibration_report([0.25, 0.75], [0, 1], n_bins=2)

    lines = report.summary_lines()
    assert lines[0] == "Calibration report (2 examples)"
    assert any("expected_calibration_error" in line for line in lines)
    assert any("[0.50, 1.00]" in line for line in lines)


def test_calibration_report_validates_arguments():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_report([0.5], [1], n_bins=0)
    with pytest.raises(ValueError, match="shape mismatch"):
        calibration_report([0.5], [1, 0])
    with pytest.raises(ValueError, match="scores must be"):
        calibration_report([1.2], [1])
    with pytest.raises(ValueError, match="labels must be binary"):
        calibration_report([0.5], [2])
