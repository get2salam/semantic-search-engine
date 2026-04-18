"""
Unit tests for the Prometheus metrics registry.
"""

from __future__ import annotations

import pytest

from metrics import MetricsRegistry, Timer


class TestCounter:
    def test_inc_with_no_labels(self):
        reg = MetricsRegistry()
        c = reg.counter("my_counter", "A counter")
        c.inc()
        c.inc(2)
        rendered = c.render()
        assert "# HELP my_counter A counter" in rendered
        assert "# TYPE my_counter counter" in rendered
        assert "my_counter 3" in rendered

    def test_inc_with_labels(self):
        reg = MetricsRegistry()
        c = reg.counter("requests_total", "Requests", labels=("method", "path"))
        c.inc(method="GET", path="/a")
        c.inc(method="GET", path="/a")
        c.inc(method="POST", path="/a")
        rendered = c.render()
        assert 'requests_total{method="GET",path="/a"} 2' in rendered
        assert 'requests_total{method="POST",path="/a"} 1' in rendered

    def test_missing_labels_raises(self):
        reg = MetricsRegistry()
        c = reg.counter("c", "help", labels=("a", "b"))
        with pytest.raises(ValueError):
            c.inc(a="1")

    def test_negative_inc_raises(self):
        reg = MetricsRegistry()
        c = reg.counter("c", "help")
        with pytest.raises(ValueError):
            c.inc(-1)

    def test_empty_counter_renders_zero(self):
        reg = MetricsRegistry()
        c = reg.counter("c", "help")
        assert "c 0" in c.render()


class TestGauge:
    def test_set_overwrites(self):
        reg = MetricsRegistry()
        g = reg.gauge("docs", "doc count")
        g.set(10)
        g.set(5)
        rendered = g.render()
        assert "docs 5" in rendered

    def test_set_with_labels(self):
        reg = MetricsRegistry()
        g = reg.gauge("temp", "temperature", labels=("zone",))
        g.set(21.5, zone="kitchen")
        g.set(19.0, zone="bedroom")
        out = g.render()
        assert 'temp{zone="kitchen"} 21.5' in out
        assert 'temp{zone="bedroom"} 19.0' in out


class TestHistogram:
    def test_observe_fills_buckets_cumulatively(self):
        reg = MetricsRegistry()
        h = reg.histogram("lat", "latency", buckets=(0.01, 0.1, 1.0))
        h.observe(0.005)  # in 0.01, 0.1, 1.0
        h.observe(0.05)  # in 0.1, 1.0
        h.observe(2.0)  # +Inf only
        out = h.render()
        assert 'lat_bucket{le="0.01"} 1' in out
        assert 'lat_bucket{le="0.1"} 2' in out
        assert 'lat_bucket{le="1.0"} 2' in out
        assert 'lat_bucket{le="+Inf"} 3' in out
        assert "lat_count 3" in out
        assert "lat_sum 2.055" in out

    def test_histogram_with_labels(self):
        reg = MetricsRegistry()
        h = reg.histogram("lat", "latency", labels=("endpoint",), buckets=(0.1,))
        h.observe(0.05, endpoint="search")
        out = h.render()
        assert 'lat_bucket{endpoint="search",le="0.1"} 1' in out
        assert 'lat_count{endpoint="search"} 1' in out


class TestTimer:
    def test_timer_observes_elapsed(self):
        reg = MetricsRegistry()
        h = reg.histogram("op_seconds", "op time", buckets=(1.0,))
        with Timer(h):
            pass
        out = h.render()
        # Single fast observation should fall in the 1.0s bucket
        assert 'op_seconds_bucket{le="1.0"} 1' in out
        assert "op_seconds_count 1" in out


class TestRegistry:
    def test_double_registration_raises(self):
        reg = MetricsRegistry()
        reg.counter("dup", "help")
        with pytest.raises(ValueError):
            reg.counter("dup", "help")

    def test_render_combines_all_metrics(self):
        reg = MetricsRegistry()
        c = reg.counter("a", "help a")
        g = reg.gauge("b", "help b")
        c.inc()
        g.set(42)
        out = reg.render()
        assert "a 1" in out
        assert "b 42" in out
        assert out.endswith("\n")

    def test_get_returns_registered_metric(self):
        reg = MetricsRegistry()
        c = reg.counter("x", "help")
        assert reg.get("x") is c
        assert reg.get("missing") is None
