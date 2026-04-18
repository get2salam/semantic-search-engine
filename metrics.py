"""
Zero-dependency Prometheus metrics
===================================
A minimal in-process metrics collector that emits the Prometheus text
exposition format. Supports counters, gauges, and histograms with
labels.

This is intentionally a small subset of the full Prometheus client
library: no multi-process aggregation, no pushgateway. The goal is to
give the semantic-search service a scrape-friendly /metrics endpoint
without pulling in prometheus_client as a dependency.

Example:
    from metrics import MetricsRegistry

    registry = MetricsRegistry()
    requests = registry.counter(
        "sse_requests_total", "Total API requests", labels=("path", "status")
    )
    requests.inc(path="/search", status="200")

    print(registry.render())
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


def _format_labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    parts = []
    for k in sorted(labels):
        v = str(labels[k]).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{k}="{v}"')
    return "{" + ",".join(parts) + "}"


@dataclass
class _Counter:
    name: str
    help: str
    label_names: tuple[str, ...]
    values: dict[tuple[str, ...], float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0, **labels) -> None:
        if amount < 0:
            raise ValueError("Counter.inc requires a non-negative amount")
        key = self._key(labels)
        with self.lock:
            self.values[key] = self.values.get(key, 0.0) + amount

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            raise ValueError(
                f"Counter {self.name} expects labels {self.label_names}, got {sorted(labels)}"
            )
        return tuple(str(labels[n]) for n in self.label_names)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        with self.lock:
            if not self.values:
                lines.append(f"{self.name} 0")
            else:
                for key, value in sorted(self.values.items()):
                    label_map = dict(zip(self.label_names, key, strict=False))
                    lines.append(f"{self.name}{_format_labels(label_map)} {value}")
        return "\n".join(lines)


@dataclass
class _Gauge:
    name: str
    help: str
    label_names: tuple[str, ...]
    values: dict[tuple[str, ...], float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float, **labels) -> None:
        key = self._key(labels)
        with self.lock:
            self.values[key] = float(value)

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            raise ValueError(
                f"Gauge {self.name} expects labels {self.label_names}, got {sorted(labels)}"
            )
        return tuple(str(labels[n]) for n in self.label_names)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        with self.lock:
            if not self.values:
                lines.append(f"{self.name} 0")
            else:
                for key, value in sorted(self.values.items()):
                    label_map = dict(zip(self.label_names, key, strict=False))
                    lines.append(f"{self.name}{_format_labels(label_map)} {value}")
        return "\n".join(lines)


# Default latency buckets (seconds). Chosen to span the realistic range
# for embedding search: sub-millisecond through several seconds.
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)


@dataclass
class _Histogram:
    name: str
    help: str
    label_names: tuple[str, ...]
    buckets: tuple[float, ...] = DEFAULT_BUCKETS
    counts: dict[tuple[str, ...], list[int]] = field(default_factory=dict)
    sums: dict[tuple[str, ...], float] = field(default_factory=dict)
    totals: dict[tuple[str, ...], int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, **labels) -> None:
        key = self._key(labels)
        with self.lock:
            if key not in self.counts:
                self.counts[key] = [0] * len(self.buckets)
                self.sums[key] = 0.0
                self.totals[key] = 0
            for i, boundary in enumerate(self.buckets):
                if value <= boundary:
                    self.counts[key][i] += 1
            self.sums[key] += float(value)
            self.totals[key] += 1

    def _key(self, labels: dict[str, str]) -> tuple[str, ...]:
        if set(labels) != set(self.label_names):
            raise ValueError(
                f"Histogram {self.name} expects labels {self.label_names}, got {sorted(labels)}"
            )
        return tuple(str(labels[n]) for n in self.label_names)

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        with self.lock:
            keys = sorted(self.counts) if self.counts else [tuple([""] * len(self.label_names))]
            for key in keys:
                label_map = dict(zip(self.label_names, key, strict=False))
                cumulative = 0
                for i, boundary in enumerate(self.buckets):
                    cumulative = self.counts.get(key, [0] * len(self.buckets))[i]
                    bucket_labels = {**label_map, "le": f"{boundary}"}
                    lines.append(f"{self.name}_bucket{_format_labels(bucket_labels)} {cumulative}")
                total = self.totals.get(key, 0)
                inf_labels = {**label_map, "le": "+Inf"}
                lines.append(f"{self.name}_bucket{_format_labels(inf_labels)} {total}")
                lines.append(
                    f"{self.name}_sum{_format_labels(label_map)} {self.sums.get(key, 0.0)}"
                )
                lines.append(f"{self.name}_count{_format_labels(label_map)} {total}")
        return "\n".join(lines)


class MetricsRegistry:
    """A collection of metrics that renders to Prometheus text format."""

    def __init__(self):
        self._metrics: dict[str, _Counter | _Gauge | _Histogram] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, help: str, labels: tuple[str, ...] = ()) -> _Counter:
        return self._register(name, _Counter(name=name, help=help, label_names=labels))

    def gauge(self, name: str, help: str, labels: tuple[str, ...] = ()) -> _Gauge:
        return self._register(name, _Gauge(name=name, help=help, label_names=labels))

    def histogram(
        self,
        name: str,
        help: str,
        labels: tuple[str, ...] = (),
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    ) -> _Histogram:
        return self._register(
            name, _Histogram(name=name, help=help, label_names=labels, buckets=buckets)
        )

    def _register(self, name, metric):
        with self._lock:
            if name in self._metrics:
                raise ValueError(f"Metric {name!r} is already registered")
            self._metrics[name] = metric
            return metric

    def get(self, name: str):
        return self._metrics.get(name)

    def render(self) -> str:
        """Render all registered metrics in Prometheus text format."""
        with self._lock:
            parts = [metric.render() for metric in self._metrics.values()]
        return "\n\n".join(parts) + "\n"


class Timer:
    """Context manager that observes elapsed seconds into a histogram."""

    def __init__(self, histogram: _Histogram, **labels):
        self._histogram = histogram
        self._labels = labels
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self._start
        self._histogram.observe(elapsed, **self._labels)
        return False
