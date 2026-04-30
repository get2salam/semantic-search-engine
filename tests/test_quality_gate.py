"""Unit tests for the retrieval quality gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quality_gate import (
    GateConfig,
    QualityGate,
    Threshold,
    load_report,
    write_baseline,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(
    *,
    mrr: float = 0.85,
    map_score: float = 0.80,
    ndcg5: float = 0.82,
    ndcg10: float = 0.84,
    p5: float = 0.60,
    r10: float = 0.70,
    model_name: str = "all-MiniLM-L6-v2",
    num_queries: int = 25,
    per_query: list | None = None,
) -> dict:
    """Build a synthetic EvalReport-shaped dict.

    Keeps the fixtures terse — only the fields the gate cares about are set.
    """
    return {
        "num_queries": num_queries,
        "k_values": [1, 5, 10],
        "mrr": mrr,
        "map": map_score,
        "ndcg": {"1": 0.75, "5": ndcg5, "10": ndcg10},
        "precision": {"1": 0.80, "5": p5, "10": 0.50},
        "recall": {"1": 0.30, "5": 0.60, "10": r10},
        "elapsed_seconds": 1.23,
        "model_name": model_name,
        "per_query": per_query or [],
    }


def _per_query_row(query: str, *, rr: float, ap: float, ndcg5: float = 0.8) -> dict:
    return {
        "query": query,
        "reciprocal_rank": rr,
        "average_precision": ap,
        "ndcg": {"5": ndcg5, "10": ndcg5},
        "precision": {"5": 0.6},
        "recall": {"5": 0.6},
    }


# ---------------------------------------------------------------------------
# Threshold logic
# ---------------------------------------------------------------------------


class TestThreshold:
    def test_no_regression_when_metric_unchanged(self):
        t = Threshold()
        regressed, reason = t.regression(baseline=0.85, current=0.85)
        assert not regressed
        assert reason is None

    def test_no_regression_on_improvement(self):
        t = Threshold()
        regressed, _ = t.regression(baseline=0.80, current=0.90)
        assert not regressed

    def test_absolute_drop_triggers(self):
        t = Threshold(abs_drop=0.02, rel_drop=1.0)  # disable rel_drop
        regressed, reason = t.regression(baseline=0.85, current=0.82)
        assert regressed
        assert "absolute" in reason

    def test_absolute_drop_within_tolerance(self):
        t = Threshold(abs_drop=0.02, rel_drop=1.0)
        regressed, _ = t.regression(baseline=0.85, current=0.84)
        assert not regressed

    def test_relative_drop_triggers(self):
        # On a low-magnitude metric, a small absolute drop is large relative.
        t = Threshold(abs_drop=1.0, rel_drop=0.10)  # disable abs_drop
        regressed, reason = t.regression(baseline=0.20, current=0.17)
        assert regressed
        assert "relative" in reason

    def test_min_value_floor(self):
        t = Threshold(abs_drop=1.0, rel_drop=1.0, min_value=0.5)
        regressed, reason = t.regression(baseline=0.6, current=0.49)
        assert regressed
        assert "floor" in reason


# ---------------------------------------------------------------------------
# QualityGate.compare
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_passes_when_metrics_unchanged(self):
        baseline = _make_report()
        current = _make_report()

        result = QualityGate().compare(baseline, current)

        assert result.passed
        assert result.regressed_metrics == []
        assert result.regressing_queries == []

    def test_passes_when_metrics_improve(self):
        baseline = _make_report(mrr=0.80, map_score=0.75)
        current = _make_report(mrr=0.85, map_score=0.82)

        result = QualityGate().compare(baseline, current)

        assert result.passed
        # Improvements should still appear as positive deltas in the table.
        mrr_change = next(c for c in result.changes if c.metric == "mrr")
        assert mrr_change.delta == pytest.approx(0.05, abs=1e-6)
        assert mrr_change.pct_delta > 0

    def test_fails_on_significant_mrr_drop(self):
        baseline = _make_report(mrr=0.85)
        current = _make_report(mrr=0.78)  # 0.07 drop, well over 0.02 default

        result = QualityGate().compare(baseline, current)

        assert not result.passed
        regressions = result.regressed_metrics
        assert any(c.metric == "mrr" for c in regressions)

    def test_fails_on_per_k_metric(self):
        baseline = _make_report(ndcg5=0.82)
        current = _make_report(ndcg5=0.78)

        result = QualityGate().compare(baseline, current)

        assert not result.passed
        ndcg5_change = next(c for c in result.changes if c.metric == "ndcg" and c.k == 5)
        assert ndcg5_change.regressed

    def test_per_metric_threshold_override(self):
        # Tighten just NDCG@5 to a 0.005 absolute drop while leaving the
        # default lenient — only NDCG@5 should fail.
        config = GateConfig(
            default_threshold=Threshold(abs_drop=0.10, rel_drop=0.50),
            thresholds={"ndcg@5": Threshold(abs_drop=0.005)},
        )
        baseline = _make_report(mrr=0.85, ndcg5=0.82)
        current = _make_report(mrr=0.81, ndcg5=0.81)  # mrr drop 0.04, ndcg5 drop 0.01

        result = QualityGate(config).compare(baseline, current)

        assert not result.passed
        regressed_keys = {c.key for c in result.regressed_metrics}
        assert regressed_keys == {"ndcg@5"}

    def test_strict_config(self):
        baseline = _make_report(mrr=0.85)
        current = _make_report(mrr=0.83)  # 0.02 drop is over strict 0.01

        assert QualityGate(GateConfig.default()).compare(baseline, current).passed
        assert not QualityGate(GateConfig.strict()).compare(baseline, current).passed

    def test_handles_string_keyed_k_dicts(self):
        # JSON round-trip turns int keys into strings; the gate must cope.
        baseline = _make_report()
        current = _make_report()
        # Re-serialize through JSON to mirror what comes off disk
        baseline = json.loads(json.dumps(baseline))
        current = json.loads(json.dumps(current))

        result = QualityGate().compare(baseline, current)
        assert result.passed
        # NDCG@5 should still be present in the comparison
        assert any(c.metric == "ndcg" and c.k == 5 for c in result.changes)

    def test_only_compares_shared_k_values(self):
        # Baseline reports only k=5; current reports k=5 and k=10.  The gate
        # should compare k=5 only, not synthesize a missing k=10 baseline.
        baseline = _make_report()
        baseline["ndcg"] = {"5": 0.82}
        current = _make_report(ndcg5=0.82, ndcg10=0.84)

        result = QualityGate().compare(baseline, current)
        ndcg_changes = [c for c in result.changes if c.metric == "ndcg"]
        assert {c.k for c in ndcg_changes} == {5}

    def test_metric_change_serialization_roundtrip(self):
        baseline = _make_report(mrr=0.85, ndcg5=0.82)
        current = _make_report(mrr=0.83, ndcg5=0.78)

        result = QualityGate().compare(baseline, current)
        payload = result.to_dict()

        reloaded = json.loads(json.dumps(payload))
        assert reloaded["passed"] == result.passed
        assert len(reloaded["changes"]) == len(result.changes)


# ---------------------------------------------------------------------------
# Per-query regression detection
# ---------------------------------------------------------------------------


class TestPerQueryRegressions:
    def test_detects_query_regression(self):
        baseline = _make_report(
            per_query=[
                _per_query_row("breach of contract", rr=1.0, ap=0.95),
                _per_query_row("property law", rr=1.0, ap=0.90),
            ]
        )
        current = _make_report(
            per_query=[
                _per_query_row("breach of contract", rr=0.5, ap=0.40),  # tanked
                _per_query_row("property law", rr=1.0, ap=0.90),  # stable
            ]
        )

        result = QualityGate().compare(baseline, current)

        assert any(q.query == "breach of contract" for q in result.regressing_queries)
        assert all(q.query != "property law" for q in result.regressing_queries)

    def test_per_query_can_be_disabled(self):
        baseline = _make_report(per_query=[_per_query_row("q", rr=1.0, ap=1.0)])
        current = _make_report(per_query=[_per_query_row("q", rr=0.0, ap=0.0)])

        config = GateConfig(per_query_drop=0)
        result = QualityGate(config).compare(baseline, current)
        assert result.regressing_queries == []

    def test_per_query_truncates_to_max_listed(self):
        per_query_baseline = [_per_query_row(f"q{i}", rr=1.0, ap=1.0) for i in range(10)]
        per_query_current = [_per_query_row(f"q{i}", rr=0.0, ap=0.0) for i in range(10)]
        baseline = _make_report(per_query=per_query_baseline)
        current = _make_report(per_query=per_query_current)

        config = GateConfig(per_query_max_listed=3)
        result = QualityGate(config).compare(baseline, current)
        assert len(result.regressing_queries) == 3

    def test_per_query_handles_missing_query_in_current(self):
        baseline = _make_report(per_query=[_per_query_row("q", rr=1.0, ap=1.0)])
        current = _make_report(per_query=[])  # current has no per-query rows

        result = QualityGate().compare(baseline, current)
        # No data → no per-query regressions, but headline metrics still pass
        assert result.regressing_queries == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_markdown_pass_includes_metrics_table(self):
        baseline = _make_report()
        current = _make_report()
        result = QualityGate().compare(baseline, current)

        md = result.render_markdown()
        assert "PASS" in md
        assert "| Metric |" in md
        assert "| `mrr` |" in md
        assert "| `ndcg@5` |" in md

    def test_markdown_fail_includes_remediation_hint(self):
        baseline = _make_report(mrr=0.85)
        current = _make_report(mrr=0.70)
        result = QualityGate().compare(baseline, current)

        md = result.render_markdown()
        assert "FAIL" in md
        assert "update the baseline" in md

    def test_markdown_per_query_section(self):
        baseline = _make_report(per_query=[_per_query_row("test query", rr=1.0, ap=1.0)])
        current = _make_report(per_query=[_per_query_row("test query", rr=0.0, ap=0.0)])
        result = QualityGate().compare(baseline, current)

        md = result.render_markdown()
        assert "Per-query regressions" in md
        assert "test query" in md

    def test_markdown_pipe_in_query_is_escaped(self):
        # A query containing a literal "|" must not break the markdown table.
        bad_query = "what about a|b syntax"
        baseline = _make_report(per_query=[_per_query_row(bad_query, rr=1.0, ap=1.0)])
        current = _make_report(per_query=[_per_query_row(bad_query, rr=0.0, ap=0.0)])
        result = QualityGate().compare(baseline, current)
        md = result.render_markdown()
        assert "a\\|b" in md

    def test_text_render(self):
        baseline = _make_report(mrr=0.85)
        current = _make_report(mrr=0.70)
        result = QualityGate().compare(baseline, current)

        text = result.render_text()
        assert "FAIL" in text
        assert "mrr" in text


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_reload_baseline(self, tmp_path: Path):
        report = _make_report()
        path = tmp_path / "baseline.json"
        write_baseline(report, path)

        assert path.exists()
        loaded = load_report(path)
        assert loaded["mrr"] == report["mrr"]

    def test_save_result_json_and_markdown(self, tmp_path: Path):
        baseline = _make_report()
        current = _make_report(mrr=0.70)
        result = QualityGate().compare(baseline, current)

        json_path = tmp_path / "out" / "gate.json"
        md_path = tmp_path / "out" / "gate.md"
        result.save_json(json_path)
        result.save_markdown(md_path)

        assert json_path.exists()
        assert md_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["passed"] is False

    def test_config_round_trip(self, tmp_path: Path):
        config = GateConfig(
            default_threshold=Threshold(abs_drop=0.03, rel_drop=0.10, min_value=0.5),
            thresholds={"mrr": Threshold(abs_drop=0.005)},
            per_query_drop=0.07,
            per_query_max_listed=10,
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

        loaded = GateConfig.load(path)
        assert loaded.default_threshold.abs_drop == 0.03
        assert loaded.thresholds["mrr"].abs_drop == 0.005
        assert loaded.per_query_drop == 0.07


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_eval_report_to_dict_is_compatible_with_gate():
    """Round-trip a real EvalReport through the gate to catch schema drift."""
    from evaluation import EvalQuery, EvalReport, EvalResult

    per_query = [
        EvalResult(
            query="q1",
            retrieved_docs=["d1", "d2"],
            relevant_docs=["d1"],
            reciprocal_rank=1.0,
            average_precision=1.0,
            ndcg={1: 1.0, 5: 1.0},
            precision={1: 1.0, 5: 0.2},
            recall={1: 1.0, 5: 1.0},
        ),
    ]
    report = EvalReport(
        num_queries=1,
        k_values=[1, 5],
        mrr=1.0,
        map_score=1.0,
        ndcg={1: 1.0, 5: 1.0},
        precision={1: 1.0, 5: 0.2},
        recall={1: 1.0, 5: 1.0},
        per_query=per_query,
        elapsed_seconds=0.1,
        model_name="test-model",
    )

    payload = report.to_dict()
    # EvalReport.to_dict() does not include per_query rows; the gate should
    # therefore tolerate their absence without crashing.
    payload.setdefault("per_query", [])
    assert isinstance(EvalQuery, type)  # smoke - import works

    result = QualityGate().compare(payload, payload)
    assert result.passed
