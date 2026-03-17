"""Tests for embedding-space anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from anomaly_detector import (
    AnomalyDetector,
    AnomalyResult,
    DetectionMethod,
    DriftDetector,
    DriftReport,
    QueryDriftMonitor,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_embeddings() -> np.ndarray:
    """Generate 200 normal embeddings (clustered around origin)."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 1, (200, 64))


@pytest.fixture
def outlier_embeddings() -> np.ndarray:
    """Generate 10 outlier embeddings (far from origin)."""
    rng = np.random.default_rng(99)
    return rng.normal(10, 0.5, (10, 64))


@pytest.fixture
def mixed_embeddings(normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray) -> np.ndarray:
    """Normal + outlier embeddings mixed together."""
    return np.vstack([normal_embeddings, outlier_embeddings])


@pytest.fixture
def shifted_embeddings() -> np.ndarray:
    """Embeddings from a shifted distribution."""
    rng = np.random.default_rng(123)
    return rng.normal(3, 1, (200, 64))


# ---------------------------------------------------------------------------
# AnomalyDetector — Z-score
# ---------------------------------------------------------------------------


class TestZScoreDetector:
    def test_fit_and_detect(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="zscore", threshold=3.0)
        det.fit(normal_embeddings)
        assert det.is_fitted

        result = det.detect(normal_embeddings)
        assert isinstance(result, AnomalyResult)
        assert result.n_total == 200
        assert result.method == "zscore"

    def test_detects_outliers(
        self, normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray
    ) -> None:
        det = AnomalyDetector(method="zscore", threshold=3.0)
        det.fit(normal_embeddings)

        result = det.detect(outlier_embeddings)
        assert result.n_anomalies > 5  # Most outliers should be caught
        assert result.anomaly_rate > 0.5

    def test_few_false_positives(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="zscore", threshold=3.0)
        det.fit(normal_embeddings)

        result = det.detect(normal_embeddings)
        assert result.anomaly_rate < 0.1  # Less than 10% false positives

    def test_anomaly_indices(
        self, normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray
    ) -> None:
        det = AnomalyDetector(method="zscore", threshold=3.0)
        det.fit(normal_embeddings)
        result = det.detect(outlier_embeddings)
        assert len(result.anomaly_indices) == result.n_anomalies

    def test_timing(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="zscore")
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings)
        assert result.detection_time_ms >= 0


# ---------------------------------------------------------------------------
# AnomalyDetector — IQR
# ---------------------------------------------------------------------------


class TestIQRDetector:
    def test_detects_outliers(
        self, normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray
    ) -> None:
        det = AnomalyDetector(method="iqr", threshold=1.5)
        det.fit(normal_embeddings)
        result = det.detect(outlier_embeddings)
        assert result.n_anomalies > 5

    def test_normal_data_clean(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="iqr", threshold=1.5)
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings)
        assert result.anomaly_rate < 0.15


# ---------------------------------------------------------------------------
# AnomalyDetector — Mahalanobis
# ---------------------------------------------------------------------------


class TestMahalanobisDetector:
    def test_detects_outliers(
        self, normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray
    ) -> None:
        det = AnomalyDetector(method="mahalanobis", threshold=15.0)
        det.fit(normal_embeddings)
        result = det.detect(outlier_embeddings)
        assert result.n_anomalies > 5

    def test_normal_data(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="mahalanobis", threshold=15.0)
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings)
        assert result.anomaly_rate < 0.2


# ---------------------------------------------------------------------------
# AnomalyDetector — LOF
# ---------------------------------------------------------------------------


class TestLOFDetector:
    def test_detects_outliers(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="lof", threshold=3.0)
        det.fit(normal_embeddings[:50])  # Smaller for speed

        rng = np.random.default_rng(99)
        outliers = rng.normal(10, 0.5, (5, 64))
        result = det.detect(outliers)
        assert result.n_anomalies > 2


# ---------------------------------------------------------------------------
# AnomalyDetector — Isolation
# ---------------------------------------------------------------------------


class TestIsolationDetector:
    def test_detects_outliers(
        self, normal_embeddings: np.ndarray, outlier_embeddings: np.ndarray
    ) -> None:
        det = AnomalyDetector(method="isolation", threshold=2.0)
        det.fit(normal_embeddings)
        result = det.detect(outlier_embeddings)
        assert result.n_anomalies > 5

    def test_normal_data(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="isolation", threshold=2.0)
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings)
        assert result.anomaly_rate < 0.15


# ---------------------------------------------------------------------------
# AnomalyDetector — Edge cases
# ---------------------------------------------------------------------------


class TestAnomalyDetectorEdgeCases:
    def test_not_fitted_raises(self) -> None:
        det = AnomalyDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            det.detect(np.zeros((5, 10)))

    def test_1d_input_raises(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector()
        with pytest.raises(ValueError, match="2D"):
            det.fit(np.zeros(10))

    def test_detect_1d_raises(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector()
        det.fit(normal_embeddings)
        with pytest.raises(ValueError, match="2D"):
            det.detect(np.zeros(10))

    def test_single_sample(self, normal_embeddings: np.ndarray) -> None:
        det = AnomalyDetector(method="zscore")
        det.fit(normal_embeddings)
        result = det.detect(normal_embeddings[:1])
        assert result.n_total == 1

    def test_all_methods_valid(self) -> None:
        for method in DetectionMethod:
            det = AnomalyDetector(method=method.value)
            assert det.method == method

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError):
            AnomalyDetector(method="invalid")


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class TestDriftDetector:
    def test_no_drift(self, normal_embeddings: np.ndarray) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, (200, 64))
        cur = rng.normal(0, 1, (200, 64))

        dd = DriftDetector(threshold=0.5)
        dd.set_reference(ref)
        report = dd.check(cur)

        assert isinstance(report, DriftReport)
        assert report.drift_score < 0.5
        assert not report.is_drifted

    def test_detects_drift(
        self, normal_embeddings: np.ndarray, shifted_embeddings: np.ndarray
    ) -> None:
        dd = DriftDetector(threshold=0.5)
        dd.set_reference(normal_embeddings)
        report = dd.check(shifted_embeddings)

        assert report.is_drifted
        assert report.drift_score > 1.0

    def test_dimension_scores(self, normal_embeddings: np.ndarray) -> None:
        dd = DriftDetector()
        dd.set_reference(normal_embeddings)
        report = dd.check(normal_embeddings)
        assert report.dimension_scores is not None
        assert len(report.dimension_scores) == 64

    def test_not_set_raises(self) -> None:
        dd = DriftDetector()
        with pytest.raises(RuntimeError, match="Reference not set"):
            dd.check(np.zeros((10, 5)))

    def test_timing(self, normal_embeddings: np.ndarray) -> None:
        dd = DriftDetector()
        dd.set_reference(normal_embeddings)
        report = dd.check(normal_embeddings)
        assert report.detection_time_ms >= 0


# ---------------------------------------------------------------------------
# QueryDriftMonitor
# ---------------------------------------------------------------------------


class TestQueryDriftMonitor:
    def test_window_accumulation(self) -> None:
        monitor = QueryDriftMonitor(window_size=5)
        rng = np.random.default_rng(42)

        for _ in range(4):
            result = monitor.add(rng.normal(0, 1, 32))
            assert result is None  # Window not full yet

        assert monitor.total_queries == 4

    def test_window_triggers_on_boundary(self) -> None:
        monitor = QueryDriftMonitor(window_size=5)
        rng = np.random.default_rng(42)

        # Fill first window (no previous → no drift report)
        for _ in range(5):
            monitor.add(rng.normal(0, 1, 32))

        # Fill second window → should get drift report
        for i in range(5):
            result = monitor.add(rng.normal(0, 1, 32))
            if i == 4:
                assert result is not None
                assert isinstance(result, DriftReport)

    def test_drift_trend(self) -> None:
        monitor = QueryDriftMonitor(window_size=10)
        rng = np.random.default_rng(42)

        # 3 windows
        for _ in range(30):
            monitor.add(rng.normal(0, 1, 16))

        trend = monitor.get_drift_trend()
        assert len(trend) == 2  # 3 windows → 2 comparisons

    def test_save_state(self) -> None:
        monitor = QueryDriftMonitor(window_size=5)
        rng = np.random.default_rng(42)

        for _ in range(15):
            monitor.add(rng.normal(0, 1, 16))

        save_path = Path("test_monitor_state.json")
        try:
            monitor.save_state(save_path)
            assert save_path.exists()

            state = json.loads(save_path.read_text())
            assert state["window_size"] == 5
            assert state["total_queries"] == 15
            assert "drift_scores" in state
        finally:
            save_path.unlink(missing_ok=True)

    def test_detects_query_shift(self) -> None:
        monitor = QueryDriftMonitor(window_size=20, drift_threshold=0.5)
        rng = np.random.default_rng(42)

        # Window 1: normal queries
        for _ in range(20):
            monitor.add(rng.normal(0, 1, 32))

        # Window 2: shifted queries
        report = None
        for _ in range(20):
            result = monitor.add(rng.normal(5, 1, 32))
            if result is not None:
                report = result

        assert report is not None
        assert report.is_drifted
        assert report.drift_score > 1.0


# ---------------------------------------------------------------------------
# AnomalyResult dataclass
# ---------------------------------------------------------------------------


class TestAnomalyResult:
    def test_anomaly_rate(self) -> None:
        result = AnomalyResult(
            scores=np.array([1.0, 5.0, 2.0]),
            is_anomaly=np.array([False, True, False]),
            threshold=3.0,
            method="zscore",
            n_anomalies=1,
            n_total=3,
            detection_time_ms=1.0,
        )
        assert abs(result.anomaly_rate - 1 / 3) < 1e-6

    def test_anomaly_rate_empty(self) -> None:
        result = AnomalyResult(
            scores=np.array([]),
            is_anomaly=np.array([]),
            threshold=3.0,
            method="zscore",
            n_anomalies=0,
            n_total=0,
            detection_time_ms=0.0,
        )
        assert result.anomaly_rate == 0.0

    def test_anomaly_indices(self) -> None:
        result = AnomalyResult(
            scores=np.array([1.0, 5.0, 2.0, 4.0]),
            is_anomaly=np.array([False, True, False, True]),
            threshold=3.0,
            method="zscore",
            n_anomalies=2,
            n_total=4,
            detection_time_ms=0.5,
        )
        assert result.anomaly_indices == [1, 3]
