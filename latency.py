"""
Latency Profiler
================
A small, allocation-light timing utility for instrumenting search pipelines.

Reasons this exists rather than reaching for ``cProfile``:

* cProfile is great for hotspot analysis but its per-call output isn't useful
  to log in production. We want a small, well-named breakdown ("encode",
  "search", "rerank", "post") that fits in a structured log line.
* The profiler's report format is JSON-serialisable so it slots cleanly into
  the structured logging already in this codebase.
* Stages can be timed across threads or async boundaries without context
  managers — useful when an API route times the encode step in one helper
  and the search step in another.

The implementation is intentionally tiny: a name → list[float] dict and a
context manager that appends a duration on exit. No global state, no
decorators that mutate functions. One profiler per request is the expected
usage.

Author: get2salam
License: MIT
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean, median
from typing import Any

__all__ = ["LatencyProfile", "LatencyProfiler"]


@dataclass(frozen=True)
class LatencyProfile:
    """Aggregated timing report for one request or pipeline run.

    The structure is a name → summary dict so the report is easy to emit as
    JSON or a Prometheus histogram.
    """

    total_ms: float
    stages: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {"total_ms": self.total_ms, "stages": self.stages}


class LatencyProfiler:
    """Collect named stage timings inside one request.

    Usage::

        prof = LatencyProfiler()
        with prof.stage("encode"):
            q = encode(query)
        with prof.stage("search"):
            res = engine.search(q)
        report = prof.report()

    The profiler accumulates multiple samples per stage when the same name is
    used more than once (e.g. timing each iteration of an inner loop).
    """

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a block of code and record it under ``name``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings.setdefault(name, []).append((time.perf_counter() - start) * 1000.0)

    def record(self, name: str, duration_ms: float) -> None:
        """Manually append a sample. Useful when timing crosses async boundaries
        and the stage() context manager isn't ergonomic."""
        if duration_ms < 0:
            raise ValueError(f"duration_ms must be non-negative, got {duration_ms}")
        self._timings.setdefault(name, []).append(float(duration_ms))

    def report(self) -> LatencyProfile:
        """Build the final aggregated report.

        Each stage's summary contains:

        * ``count``  — number of samples
        * ``total``  — sum of samples in milliseconds
        * ``mean``   — arithmetic mean
        * ``median`` — median
        * ``max``    — maximum
        """
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        stages: dict[str, dict[str, float]] = {}
        for name, samples in self._timings.items():
            if not samples:
                continue
            stages[name] = {
                "count": float(len(samples)),
                "total": float(sum(samples)),
                "mean": float(mean(samples)),
                "median": float(median(samples)),
                "max": float(max(samples)),
            }
        return LatencyProfile(total_ms=total_ms, stages=stages)
