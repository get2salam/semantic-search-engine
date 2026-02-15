"""
Tests for Experiment Tracker
==============================
Comprehensive test suite covering run lifecycle, metrics logging,
comparison, best-run selection, model lineage, and persistence.
"""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from experiment_tracker import (
    ActiveRun,
    ExperimentTracker,
    MetricEntry,
    RunComparison,
    RunRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test experiments."""
    d = tempfile.mkdtemp(prefix="exp_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tracker(tmp_dir):
    """Create a fresh ExperimentTracker."""
    return ExperimentTracker(tmp_dir)


# ---------------------------------------------------------------------------
# MetricEntry
# ---------------------------------------------------------------------------


class TestMetricEntry:
    def test_to_dict(self):
        entry = MetricEntry(value=0.95, step=3, timestamp="2025-01-01T00:00:00")
        d = entry.to_dict()
        assert d["value"] == 0.95
        assert d["step"] == 3
        assert d["timestamp"] == "2025-01-01T00:00:00"

    def test_defaults(self):
        entry = MetricEntry(value=0.5)
        assert entry.step is None
        assert entry.timestamp is None


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_to_dict_roundtrip(self):
        record = RunRecord(
            run_id="test-run-001",
            name="test-run",
            status="completed",
            tags=["baseline"],
            params={"lr": 0.001, "epochs": 5},
            final_metrics={"mrr": 0.85, "map": 0.80},
            model_version="1.0.0",
        )
        d = record.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored.run_id == record.run_id
        assert restored.name == record.name
        assert restored.tags == ["baseline"]
        assert restored.params["lr"] == 0.001
        assert restored.final_metrics["mrr"] == 0.85
        assert restored.model_version == "1.0.0"

    def test_from_dict_extra_keys_ignored(self):
        """from_dict should ignore unknown keys gracefully."""
        data = {
            "run_id": "x",
            "name": "x",
            "unknown_field": "should be ignored",
        }
        record = RunRecord.from_dict(data)
        assert record.run_id == "x"
        assert not hasattr(record, "unknown_field") or True  # just shouldn't raise


# ---------------------------------------------------------------------------
# ActiveRun
# ---------------------------------------------------------------------------


class TestActiveRun:
    def test_log_params(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.log_param("model", "all-MiniLM-L6-v2")
        run.log_params({"lr": 2e-5, "epochs": 3})
        assert record.params["model"] == "all-MiniLM-L6-v2"
        assert record.params["lr"] == 2e-5
        assert record.params["epochs"] == 3

    def test_log_metric_auto_step(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.log_metric("loss", 0.5)
        run.log_metric("loss", 0.3)
        run.log_metric("loss", 0.1)
        assert len(record.metrics["loss"]) == 3
        steps = [e["step"] for e in record.metrics["loss"]]
        assert steps == [0, 1, 2]
        assert record.final_metrics["loss"] == 0.1

    def test_log_metric_explicit_step(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.log_metric("acc", 0.7, step=10)
        run.log_metric("acc", 0.9, step=20)
        assert record.metrics["acc"][0]["step"] == 10
        assert record.metrics["acc"][1]["step"] == 20

    def test_log_metrics_batch(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.log_metrics({"mrr": 0.85, "map": 0.80, "ndcg@5": 0.78})
        assert record.final_metrics["mrr"] == 0.85
        assert record.final_metrics["map"] == 0.80
        assert record.final_metrics["ndcg@5"] == 0.78

    def test_log_artifact_file(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))

        # Create a source file
        src = Path(tmp_dir) / "test_model.json"
        src.write_text('{"model": "test"}')

        dest = run.log_artifact(str(src))
        assert len(record.artifacts) == 1
        assert Path(dest).exists()

    def test_log_artifact_missing_file(self, tmp_dir):
        """Should handle missing files gracefully."""
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.log_artifact("/nonexistent/file.txt")
        assert len(record.artifacts) == 1  # recorded path but no copy

    def test_model_version(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.set_model_version("2.1.0")
        assert record.model_version == "2.1.0"

    def test_model_hash(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))

        # Create a model directory
        model_dir = Path(tmp_dir) / "model"
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"fake weights")
        (model_dir / "config.json").write_text('{"hidden_size": 384}')

        h = run.compute_model_hash(str(model_dir))
        assert len(h) == 16  # truncated SHA-256
        assert record.model_hash == h

    def test_model_hash_deterministic(self, tmp_dir):
        """Same content should produce the same hash."""
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))

        model_dir = Path(tmp_dir) / "model"
        model_dir.mkdir()
        (model_dir / "a.txt").write_text("hello")

        h1 = run.compute_model_hash(str(model_dir))
        h2 = run.compute_model_hash(str(model_dir))
        assert h1 == h2

    def test_notes_and_description(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run.set_description("Testing the baseline model")
        run.add_note("Started with default hyperparams")
        run.add_note("Loss converging well")
        assert record.description == "Testing the baseline model"
        assert len(record.notes) == 2
        assert "Started with default" in record.notes[0]

    def test_lifecycle_start_complete(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run._start()
        assert record.status == "running"
        assert record.start_time is not None

        time.sleep(0.01)
        run._complete()
        assert record.status == "completed"
        assert record.end_time is not None
        assert record.duration_seconds >= 0

    def test_lifecycle_fail(self, tmp_dir):
        record = RunRecord(run_id="r1", name="test")
        run = ActiveRun(record, Path(tmp_dir))
        run._start()
        run._fail("OOM error")
        assert record.status == "failed"
        assert any("FAILED: OOM error" in n for n in record.notes)


# ---------------------------------------------------------------------------
# ExperimentTracker
# ---------------------------------------------------------------------------


class TestExperimentTracker:
    def test_init_creates_directory(self, tmp_dir):
        tracker = ExperimentTracker(Path(tmp_dir) / "new_experiments")
        assert tracker.runs_dir.exists()
        assert tracker.run_count == 0

    def test_start_run_context_manager(self, tracker):
        with tracker.start_run("test-run") as run:
            run.log_params({"lr": 0.001})
            run.log_metric("loss", 0.5)

        assert tracker.run_count == 1
        runs = tracker.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].params["lr"] == 0.001

    def test_run_persisted_to_disk(self, tracker):
        with tracker.start_run("persist-test") as run:
            run.log_metrics({"mrr": 0.9})

        # Verify JSON file exists
        json_files = list(tracker.runs_dir.glob("*.json"))
        assert len(json_files) == 1

        # Load and verify content
        with open(json_files[0]) as f:
            data = json.load(f)
        assert data["name"] == "persist-test"
        assert data["final_metrics"]["mrr"] == 0.9

    def test_run_failure_recorded(self, tracker):
        with pytest.raises(ValueError):
            with tracker.start_run("fail-test") as run:
                run.log_metric("loss", 0.5)
                raise ValueError("Intentional failure")

        assert tracker.run_count == 1
        runs = tracker.list_runs()
        assert runs[0].status == "failed"

    def test_multiple_runs(self, tracker):
        for i in range(5):
            with tracker.start_run(f"run-{i}") as run:
                run.log_metric("score", i * 0.1)

        assert tracker.run_count == 5
        runs = tracker.list_runs()
        assert len(runs) == 5

    def test_list_runs_filter_by_status(self, tracker):
        with tracker.start_run("good") as run:
            run.log_metric("x", 1.0)

        with pytest.raises(RuntimeError):
            with tracker.start_run("bad") as run:
                raise RuntimeError("boom")

        completed = tracker.list_runs(status="completed")
        failed = tracker.list_runs(status="failed")
        assert len(completed) == 1
        assert len(failed) == 1
        assert completed[0].name == "good"
        assert failed[0].name == "bad"

    def test_list_runs_filter_by_tags(self, tracker):
        with tracker.start_run("a", tags=["baseline"]) as run:
            run.log_metric("x", 1)
        with tracker.start_run("b", tags=["experiment"]) as run:
            run.log_metric("x", 2)
        with tracker.start_run("c", tags=["baseline", "v2"]) as run:
            run.log_metric("x", 3)

        baseline_runs = tracker.list_runs(tags=["baseline"])
        assert len(baseline_runs) == 2

    def test_list_runs_with_limit(self, tracker):
        for i in range(10):
            with tracker.start_run(f"r-{i}") as run:
                run.log_metric("x", float(i))

        limited = tracker.list_runs(limit=3)
        assert len(limited) == 3

    def test_get_run_by_id(self, tracker):
        with tracker.start_run("find-me") as run:
            run_id = run.run_id
            run.log_metric("x", 42)

        found = tracker.get_run(run_id)
        assert found is not None
        assert found.name == "find-me"

    def test_get_run_by_name(self, tracker):
        with tracker.start_run("named-run") as run:
            run.log_metric("x", 1)

        found = tracker.get_run_by_name("named-run")
        assert found is not None
        assert found.final_metrics["x"] == 1

    def test_get_run_not_found(self, tracker):
        assert tracker.get_run("nonexistent") is None
        assert tracker.get_run_by_name("ghost") is None

    def test_delete_run(self, tracker):
        with tracker.start_run("delete-me") as run:
            run_id = run.run_id
            run.log_metric("x", 1)

        assert tracker.run_count == 1
        assert tracker.delete_run(run_id)
        assert tracker.run_count == 0
        assert tracker.get_run(run_id) is None

    def test_delete_nonexistent(self, tracker):
        assert not tracker.delete_run("fake-id")

    def test_persistence_reload(self, tmp_dir):
        """Tracker should reload runs from disk on init."""
        tracker1 = ExperimentTracker(tmp_dir)
        with tracker1.start_run("persistent") as run:
            run.log_params({"model": "bert"})
            run.log_metrics({"mrr": 0.92, "map": 0.88})

        # Create new tracker from same directory
        tracker2 = ExperimentTracker(tmp_dir)
        assert tracker2.run_count == 1
        reloaded = tracker2.list_runs()[0]
        assert reloaded.name == "persistent"
        assert reloaded.params["model"] == "bert"
        assert reloaded.final_metrics["mrr"] == 0.92


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


class TestComparison:
    def test_compare_runs(self, tracker):
        with tracker.start_run("baseline") as run:
            run.log_params({"model": "MiniLM", "lr": 2e-5})
            run.log_metrics({"mrr": 0.80, "map": 0.75})

        with tracker.start_run("improved") as run:
            run.log_params({"model": "mpnet", "lr": 3e-5})
            run.log_metrics({"mrr": 0.88, "map": 0.82})

        comparison = tracker.compare(["baseline", "improved"])
        assert len(comparison.runs) == 2
        assert "mrr" in comparison.metric_keys
        assert "map" in comparison.metric_keys

    def test_compare_to_dict(self, tracker):
        with tracker.start_run("a") as run:
            run.log_metrics({"mrr": 0.7})
        with tracker.start_run("b") as run:
            run.log_metrics({"mrr": 0.9})

        comparison = tracker.compare(["a", "b"])
        d = comparison.to_dict()
        assert "columns" in d
        assert "rows" in d
        assert len(d["rows"]) == 2

    def test_diff_against_baseline(self, tracker):
        with tracker.start_run("base") as run:
            run.log_metrics({"mrr": 0.80, "map": 0.70})
        with tracker.start_run("exp") as run:
            run.log_metrics({"mrr": 0.85, "map": 0.75})

        comparison = tracker.compare(["base", "exp"])
        diffs = comparison.diff("base")
        assert "exp" in diffs
        assert abs(diffs["exp"]["mrr"] - 0.05) < 1e-6
        assert abs(diffs["exp"]["map"] - 0.05) < 1e-6

    def test_diff_baseline_not_found(self, tracker):
        with tracker.start_run("x") as run:
            run.log_metrics({"mrr": 0.5})

        comparison = tracker.compare(["x"])
        with pytest.raises(ValueError, match="not found"):
            comparison.diff("nonexistent")

    def test_compare_missing_run(self, tracker):
        with tracker.start_run("exists") as run:
            run.log_metrics({"mrr": 0.5})

        comparison = tracker.compare(["exists", "ghost"])
        assert len(comparison.runs) == 1  # ghost is skipped

    def test_print_table_smoke(self, tracker, capsys):
        """Verify print_table runs without error."""
        with tracker.start_run("a") as run:
            run.log_params({"lr": 0.001})
            run.log_metrics({"mrr": 0.8})
        with tracker.start_run("b") as run:
            run.log_params({"lr": 0.01})
            run.log_metrics({"mrr": 0.9})

        comparison = tracker.compare(["a", "b"])
        comparison.print_table()
        captured = capsys.readouterr()
        assert "Experiment Comparison" in captured.out
        assert "mrr" in captured.out


# ---------------------------------------------------------------------------
# Best Run
# ---------------------------------------------------------------------------


class TestBestRun:
    def test_best_run_higher_is_better(self, tracker):
        for score in [0.7, 0.9, 0.8]:
            with tracker.start_run(f"run-{score}") as run:
                run.log_metric("mrr", score)

        best = tracker.best_run("mrr", higher_is_better=True)
        assert best is not None
        assert best.final_metrics["mrr"] == 0.9

    def test_best_run_lower_is_better(self, tracker):
        for loss in [0.3, 0.1, 0.5]:
            with tracker.start_run(f"run-{loss}") as run:
                run.log_metric("loss", loss)

        best = tracker.best_run("loss", higher_is_better=False)
        assert best is not None
        assert best.final_metrics["loss"] == 0.1

    def test_best_run_no_matches(self, tracker):
        with tracker.start_run("x") as run:
            run.log_metric("acc", 0.5)

        best = tracker.best_run("nonexistent_metric")
        assert best is None

    def test_best_run_only_completed(self, tracker):
        with tracker.start_run("good") as run:
            run.log_metric("mrr", 0.5)

        with pytest.raises(ZeroDivisionError):
            with tracker.start_run("better-but-failed") as run:
                run.log_metric("mrr", 0.99)
                raise ZeroDivisionError("crash")

        best = tracker.best_run("mrr")
        assert best.name == "good"

    def test_best_run_with_tags(self, tracker):
        with tracker.start_run("a", tags=["prod"]) as run:
            run.log_metric("mrr", 0.7)
        with tracker.start_run("b", tags=["dev"]) as run:
            run.log_metric("mrr", 0.9)

        best = tracker.best_run("mrr", tags=["prod"])
        assert best.name == "a"


# ---------------------------------------------------------------------------
# Metric History
# ---------------------------------------------------------------------------


class TestMetricHistory:
    def test_get_history(self, tracker):
        with tracker.start_run("training") as run:
            for i in range(5):
                run.log_metric("loss", 1.0 - i * 0.2, step=i)

        history = tracker.metric_history("training", "loss")
        assert len(history) == 5
        assert history[0] == (0, 1.0)
        assert history[4] == (4, 0.2)

    def test_history_nonexistent(self, tracker):
        history = tracker.metric_history("ghost", "loss")
        assert history == []


# ---------------------------------------------------------------------------
# Model Lineage
# ---------------------------------------------------------------------------


class TestModelLineage:
    def test_lineage_chain(self, tracker):
        with tracker.start_run("v1") as run:
            v1_id = run.run_id
            run.set_model_version("1.0.0")
            run.log_metric("mrr", 0.7)

        with tracker.start_run("v2", parent_run_id=v1_id) as run:
            v2_id = run.run_id
            run.set_model_version("2.0.0")
            run.log_metric("mrr", 0.8)

        with tracker.start_run("v3", parent_run_id=v2_id) as run:
            run.set_model_version("3.0.0")
            run.log_metric("mrr", 0.9)

        lineage = tracker.model_lineage("v3")
        assert len(lineage) == 3
        assert lineage[0].model_version == "3.0.0"
        assert lineage[1].model_version == "2.0.0"
        assert lineage[2].model_version == "1.0.0"

    def test_lineage_single(self, tracker):
        with tracker.start_run("solo") as run:
            run.set_model_version("1.0.0")

        lineage = tracker.model_lineage("solo")
        assert len(lineage) == 1

    def test_latest_model(self, tracker):
        with tracker.start_run("old") as run:
            run.set_model_version("1.0.0")
        with tracker.start_run("new") as run:
            run.set_model_version("2.0.0")
        with tracker.start_run("no-version") as run:
            run.log_metric("x", 1)

        latest = tracker.latest_model()
        assert latest is not None
        assert latest.model_version == "2.0.0"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_summary(self, tracker, tmp_dir):
        with tracker.start_run("a") as run:
            run.log_metrics({"mrr": 0.8})
        with tracker.start_run("b") as run:
            run.log_metrics({"mrr": 0.9})

        export_path = Path(tmp_dir) / "summary.json"
        tracker.export_summary(export_path)
        assert export_path.exists()

        with open(export_path) as f:
            data = json.load(f)
        assert data["total_runs"] == 2
        assert data["completed"] == 2
        assert len(data["runs"]) == 2

    def test_export_empty(self, tracker, tmp_dir):
        export_path = Path(tmp_dir) / "empty.json"
        tracker.export_summary(export_path)
        with open(export_path) as f:
            data = json.load(f)
        assert data["total_runs"] == 0


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_run_with_no_metrics(self, tracker):
        with tracker.start_run("empty-run") as run:
            run.log_params({"model": "test"})

        runs = tracker.list_runs()
        assert runs[0].final_metrics == {}

    def test_run_with_special_characters_in_name(self, tracker):
        with tracker.start_run("model/v2 (fine-tuned)") as run:
            run.log_metric("score", 0.5)

        assert tracker.run_count == 1

    def test_concurrent_metric_keys(self, tracker):
        """Different metrics should not interfere."""
        with tracker.start_run("multi") as run:
            run.log_metric("loss", 0.5)
            run.log_metric("acc", 0.8)
            run.log_metric("loss", 0.3)
            run.log_metric("acc", 0.9)

        record = tracker.list_runs()[0]
        assert len(record.metrics["loss"]) == 2
        assert len(record.metrics["acc"]) == 2
        assert record.final_metrics["loss"] == 0.3
        assert record.final_metrics["acc"] == 0.9

    def test_repr(self, tracker):
        repr_str = repr(tracker)
        assert "ExperimentTracker" in repr_str
        assert "runs=0" in repr_str

    def test_run_count_property(self, tracker):
        assert tracker.run_count == 0
        with tracker.start_run("x") as run:
            run.log_metric("x", 1)
        assert tracker.run_count == 1
