"""
Tests for the latency profiler.
"""

from __future__ import annotations

import time

import pytest

from latency import LatencyProfile, LatencyProfiler


class TestLatencyProfiler:
    def test_basic_stage_timing(self):
        prof = LatencyProfiler()
        with prof.stage("encode"):
            time.sleep(0.005)
        report = prof.report()

        assert isinstance(report, LatencyProfile)
        assert "encode" in report.stages
        assert report.stages["encode"]["count"] == 1
        # Sleep is loose on Windows but should be > 1 ms.
        assert report.stages["encode"]["mean"] > 1

    def test_multiple_samples_aggregate(self):
        prof = LatencyProfiler()
        for _ in range(3):
            with prof.stage("loop"):
                time.sleep(0.002)
        report = prof.report()
        assert report.stages["loop"]["count"] == 3
        assert report.stages["loop"]["total"] >= report.stages["loop"]["mean"]

    def test_record_manual_sample(self):
        prof = LatencyProfiler()
        prof.record("rerank", 12.5)
        prof.record("rerank", 7.5)
        report = prof.report()
        assert report.stages["rerank"]["count"] == 2
        assert report.stages["rerank"]["total"] == pytest.approx(20.0, abs=1e-6)
        assert report.stages["rerank"]["max"] == 12.5

    def test_record_negative_raises(self):
        prof = LatencyProfiler()
        with pytest.raises(ValueError):
            prof.record("bad", -1.0)

    def test_total_ms_includes_all_stages(self):
        prof = LatencyProfiler()
        with prof.stage("a"):
            time.sleep(0.002)
        with prof.stage("b"):
            time.sleep(0.002)
        report = prof.report()
        # Total should be at least the sum of stage means.
        sum_means = report.stages["a"]["mean"] + report.stages["b"]["mean"]
        assert report.total_ms >= sum_means - 1.0  # tolerance for jitter

    def test_to_dict_is_serialisable(self):
        import json

        prof = LatencyProfiler()
        prof.record("x", 1.0)
        # Round-trip through json — dataclass values must be plain Python types.
        s = json.dumps(prof.report().to_dict())
        loaded = json.loads(s)
        assert "x" in loaded["stages"]

    def test_exception_in_block_still_records(self):
        prof = LatencyProfiler()
        with pytest.raises(RuntimeError), prof.stage("buggy"):
            time.sleep(0.001)
            raise RuntimeError("boom")
        # The finally branch must run even when the body raises.
        report = prof.report()
        assert "buggy" in report.stages
        assert report.stages["buggy"]["count"] == 1
