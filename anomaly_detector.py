"""Embedding-space anomaly detection for semantic search pipelines.

Detects outlier documents, query drift, and embedding distribution shifts
using statistical methods over vector spaces. Zero external ML dependencies —
pure NumPy implementation.

Use cases:
    - Detect low-quality or corrupted documents in your index
    - Monitor query drift over time (are users searching for new topics?)
    - Alert when embedding distributions shift (model degradation)
    - Find cluster outliers for data cleaning

Example::

    detector = AnomalyDetector(method="zscore", threshold=3.0)
    detector.fit(document_embeddings)
    outliers = detector.detect(new_embeddings)
    for idx, score in outliers:
        print(f"Document {idx}: anomaly score {score:.3f}")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class DetectionMethod(Enum):
    """Supported anomaly detection methods."""

    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION = "isolation"
    LOF = "lof"
    MAHALANOBIS = "mahalanobis"


@dataclass
class AnomalyResult:
    """Result of anomaly detection on a set of vectors.

    Attributes:
        scores: Anomaly score for each input vector (higher = more anomalous).
        is_anomaly: Boolean mask — True for detected anomalies.
        threshold: The threshold used for classification.
        method: Detection method used.
        n_anomalies: Count of detected anomalies.
        n_total: Total vectors evaluated.
        detection_time_ms: Wall-clock time in milliseconds.
        metadata: Additional method-specific metadata.
    """

    scores: np.ndarray
    is_anomaly: np.ndarray
    threshold: float
    method: str
    n_anomalies: int
    n_total: int
    detection_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def anomaly_rate(self) -> float:
        """Fraction of inputs classified as anomalies."""
        return self.n_anomalies / self.n_total if self.n_total > 0 else 0.0

    @property
    def anomaly_indices(self) -> list[int]:
        """Indices of detected anomalies."""
        return [int(i) for i in np.where(self.is_anomaly)[0]]


@dataclass
class DriftReport:
    """Report from distribution drift detection.

    Attributes:
        drift_score: Overall drift magnitude (0 = no drift, higher = more drift).
        is_drifted: Whether drift exceeds the threshold.
        threshold: Drift threshold used.
        dimension_scores: Per-dimension drift scores.
        reference_mean: Mean of the reference distribution.
        current_mean: Mean of the current distribution.
        detection_time_ms: Wall-clock time.
    """

    drift_score: float
    is_drifted: bool
    threshold: float
    dimension_scores: np.ndarray | None = None
    reference_mean: np.ndarray | None = None
    current_mean: np.ndarray | None = None
    detection_time_ms: float = 0.0


class AnomalyDetector:
    """Embedding-space anomaly detector.

    Fits a reference distribution from training embeddings, then scores
    new embeddings against it to find outliers.

    Args:
        method: Detection method to use.
        threshold: Anomaly threshold (meaning varies by method).
        contamination: Expected fraction of outliers in training data (for robust fitting).

    Example::

        detector = AnomalyDetector(method="zscore", threshold=3.0)
        detector.fit(train_embeddings)
        result = detector.detect(test_embeddings)
    """

    def __init__(
        self,
        method: str = "zscore",
        threshold: float = 3.0,
        contamination: float = 0.05,
    ) -> None:
        self.method = DetectionMethod(method)
        self.threshold = threshold
        self.contamination = contamination

        # Fitted parameters
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._q1: np.ndarray | None = None
        self._q3: np.ndarray | None = None
        self._iqr: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the detector has been fitted to reference data."""
        return self._fitted

    def fit(self, embeddings: np.ndarray) -> AnomalyDetector:
        """Fit the detector to reference embeddings.

        Args:
            embeddings: Reference embedding matrix (n_samples, n_dims).

        Returns:
            Self for chaining.
        """
        if embeddings.ndim != 2:
            msg = f"Expected 2D array, got {embeddings.ndim}D"
            raise ValueError(msg)

        if self.method == DetectionMethod.ZSCORE:
            self._mean = np.mean(embeddings, axis=0)
            self._std = np.std(embeddings, axis=0)
            # Avoid division by zero
            self._std = np.where(self._std < 1e-10, 1e-10, self._std)

        elif self.method == DetectionMethod.IQR:
            self._q1 = np.percentile(embeddings, 25, axis=0)
            self._q3 = np.percentile(embeddings, 75, axis=0)
            self._iqr = self._q3 - self._q1
            self._iqr = np.where(self._iqr < 1e-10, 1e-10, self._iqr)

        elif self.method == DetectionMethod.MAHALANOBIS:
            self._mean = np.mean(embeddings, axis=0)
            cov = np.cov(embeddings.T)
            # Regularise for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            self._cov_inv = np.linalg.inv(cov)

        elif self.method == DetectionMethod.LOF:
            # Store reference data for k-NN lookup
            self._reference = embeddings.copy()
            self._mean = np.mean(embeddings, axis=0)

        elif self.method == DetectionMethod.ISOLATION:
            # Simplified isolation: use distance from centroid + random projections
            self._mean = np.mean(embeddings, axis=0)
            self._std = np.std(embeddings, axis=0)
            self._std = np.where(self._std < 1e-10, 1e-10, self._std)
            # Store centroid distances for threshold calibration
            dists = np.linalg.norm(embeddings - self._mean, axis=1)
            self._dist_mean = np.mean(dists)
            self._dist_std = np.std(dists)

        self._fitted = True
        return self

    def detect(self, embeddings: np.ndarray) -> AnomalyResult:
        """Score embeddings and detect anomalies.

        Args:
            embeddings: Embedding matrix to evaluate (n_samples, n_dims).

        Returns:
            AnomalyResult with scores, classifications, and metadata.
        """
        if not self._fitted:
            msg = "Detector not fitted. Call fit() first."
            raise RuntimeError(msg)

        if embeddings.ndim != 2:
            msg = f"Expected 2D array, got {embeddings.ndim}D"
            raise ValueError(msg)

        start = time.perf_counter()

        if self.method == DetectionMethod.ZSCORE:
            scores = self._score_zscore(embeddings)
        elif self.method == DetectionMethod.IQR:
            scores = self._score_iqr(embeddings)
        elif self.method == DetectionMethod.MAHALANOBIS:
            scores = self._score_mahalanobis(embeddings)
        elif self.method == DetectionMethod.LOF:
            scores = self._score_lof(embeddings)
        elif self.method == DetectionMethod.ISOLATION:
            scores = self._score_isolation(embeddings)
        else:
            msg = f"Unknown method: {self.method}"
            raise ValueError(msg)

        is_anomaly = scores > self.threshold
        elapsed = (time.perf_counter() - start) * 1000

        return AnomalyResult(
            scores=scores,
            is_anomaly=is_anomaly,
            threshold=self.threshold,
            method=self.method.value,
            n_anomalies=int(np.sum(is_anomaly)),
            n_total=len(embeddings),
            detection_time_ms=elapsed,
        )

    def _score_zscore(self, embeddings: np.ndarray) -> np.ndarray:
        """Z-score: mean absolute deviation from reference mean per dimension."""
        z = np.abs((embeddings - self._mean) / self._std)
        return np.mean(z, axis=1)

    def _score_iqr(self, embeddings: np.ndarray) -> np.ndarray:
        """IQR: number of IQR widths outside the Q1-Q3 range."""
        below = np.maximum(0, self._q1 - embeddings) / self._iqr
        above = np.maximum(0, embeddings - self._q3) / self._iqr
        return np.mean(below + above, axis=1)

    def _score_mahalanobis(self, embeddings: np.ndarray) -> np.ndarray:
        """Mahalanobis distance from the reference centroid."""
        diff = embeddings - self._mean
        # Mahalanobis: sqrt(diff @ cov_inv @ diff.T) for each row
        left = diff @ self._cov_inv
        scores = np.sqrt(np.sum(left * diff, axis=1))
        return scores

    def _score_lof(self, embeddings: np.ndarray, k: int = 10) -> np.ndarray:
        """Simplified Local Outlier Factor using k-NN distances."""
        k = min(k, len(self._reference) - 1)
        scores = np.zeros(len(embeddings))

        for i, vec in enumerate(embeddings):
            # Compute distances to all reference points
            dists = np.linalg.norm(self._reference - vec, axis=1)
            # k-th nearest neighbour distance
            knn_dists = np.sort(dists)[:k]
            # Average k-NN distance as anomaly score
            scores[i] = np.mean(knn_dists)

        return scores

    def _score_isolation(self, embeddings: np.ndarray) -> np.ndarray:
        """Simplified isolation score based on normalised centroid distance."""
        dists = np.linalg.norm(embeddings - self._mean, axis=1)
        # Normalise by reference distribution
        if self._dist_std > 1e-10:
            scores = (dists - self._dist_mean) / self._dist_std
        else:
            scores = dists - self._dist_mean
        return np.abs(scores)


class DriftDetector:
    """Detect distribution drift between reference and current embeddings.

    Compares embedding distributions using statistical tests to detect
    when the data distribution has shifted (e.g., new topics, model degradation).

    Args:
        threshold: Drift score threshold to trigger an alert.
        window_size: Number of recent samples to consider.

    Example::

        drift = DriftDetector(threshold=0.5)
        drift.set_reference(baseline_embeddings)
        report = drift.check(current_embeddings)
        if report.is_drifted:
            print(f"Distribution drift detected! Score: {report.drift_score:.3f}")
    """

    def __init__(self, threshold: float = 0.5, window_size: int = 1000) -> None:
        self.threshold = threshold
        self.window_size = window_size
        self._ref_mean: np.ndarray | None = None
        self._ref_std: np.ndarray | None = None
        self._ref_n: int = 0

    def set_reference(self, embeddings: np.ndarray) -> DriftDetector:
        """Set the reference (baseline) distribution.

        Args:
            embeddings: Reference embedding matrix.

        Returns:
            Self for chaining.
        """
        self._ref_mean = np.mean(embeddings, axis=0)
        self._ref_std = np.std(embeddings, axis=0)
        self._ref_std = np.where(self._ref_std < 1e-10, 1e-10, self._ref_std)
        self._ref_n = len(embeddings)
        return self

    def check(self, embeddings: np.ndarray) -> DriftReport:
        """Check current embeddings against the reference distribution.

        Uses a simplified two-sample test comparing mean and variance
        of each embedding dimension.

        Args:
            embeddings: Current embedding matrix to check for drift.

        Returns:
            DriftReport with drift score and per-dimension analysis.
        """
        if self._ref_mean is None:
            msg = "Reference not set. Call set_reference() first."
            raise RuntimeError(msg)

        start = time.perf_counter()

        cur_mean = np.mean(embeddings, axis=0)
        cur_std = np.std(embeddings, axis=0)

        # Per-dimension: normalised mean shift (Cohen's d approximation)
        pooled_std = np.sqrt((self._ref_std**2 + cur_std**2) / 2)
        pooled_std = np.where(pooled_std < 1e-10, 1e-10, pooled_std)
        dimension_scores = np.abs(cur_mean - self._ref_mean) / pooled_std

        # Overall drift score: RMS of dimension scores
        drift_score = float(np.sqrt(np.mean(dimension_scores**2)))

        elapsed = (time.perf_counter() - start) * 1000

        return DriftReport(
            drift_score=drift_score,
            is_drifted=drift_score > self.threshold,
            threshold=self.threshold,
            dimension_scores=dimension_scores,
            reference_mean=self._ref_mean,
            current_mean=cur_mean,
            detection_time_ms=elapsed,
        )


class QueryDriftMonitor:
    """Monitor query embedding drift over time windows.

    Tracks query embeddings in rolling windows and detects when
    user search behaviour shifts significantly.

    Args:
        window_size: Number of queries per window.
        drift_threshold: Threshold to flag drift.
        history_size: Number of windows to retain.

    Example::

        monitor = QueryDriftMonitor(window_size=100)
        # Feed queries as they come in
        for query_embedding in query_stream:
            alert = monitor.add(query_embedding)
            if alert:
                print(f"Query drift detected: {alert.drift_score:.3f}")
    """

    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.5,
        history_size: int = 10,
    ) -> None:
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.history_size = history_size
        self._current_window: list[np.ndarray] = []
        self._previous_windows: list[np.ndarray] = []
        self._detector = DriftDetector(threshold=drift_threshold)
        self._drift_history: list[DriftReport] = []

    @property
    def total_queries(self) -> int:
        """Total queries processed across all windows."""
        return len(self._previous_windows) * self.window_size + len(self._current_window)

    def add(self, embedding: np.ndarray) -> DriftReport | None:
        """Add a query embedding. Returns DriftReport if window completed.

        Args:
            embedding: Single query embedding vector.

        Returns:
            DriftReport if a window boundary was crossed, else None.
        """
        self._current_window.append(embedding)

        if len(self._current_window) >= self.window_size:
            return self._process_window()

        return None

    def _process_window(self) -> DriftReport | None:
        """Process completed window and check for drift."""
        current = np.array(self._current_window)

        report = None
        if self._previous_windows:
            # Compare against the last window
            previous = self._previous_windows[-1]
            self._detector.set_reference(previous)
            report = self._detector.check(current)
            self._drift_history.append(report)

            # Trim history
            if len(self._drift_history) > self.history_size:
                self._drift_history = self._drift_history[-self.history_size :]

        # Store current as previous
        self._previous_windows.append(current)
        if len(self._previous_windows) > self.history_size:
            self._previous_windows = self._previous_windows[-self.history_size :]

        self._current_window = []
        return report

    def get_drift_trend(self) -> list[float]:
        """Get historical drift scores."""
        return [r.drift_score for r in self._drift_history]

    def save_state(self, path: str | Path) -> None:
        """Save monitor state to JSON for persistence."""
        state = {
            "window_size": self.window_size,
            "drift_threshold": self.drift_threshold,
            "total_queries": self.total_queries,
            "drift_scores": self.get_drift_trend(),
            "n_windows": len(self._previous_windows),
        }
        Path(path).write_text(json.dumps(state, indent=2), encoding="utf-8")
