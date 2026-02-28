"""
Experiment Tracker
==================
Lightweight MLOps experiment tracking for model training runs.
Logs parameters, metrics, artifacts, and model versions without
external dependencies (MLflow, W&B, etc.).

Designed to integrate with the training and evaluation pipelines
to provide reproducibility, comparison, and model lineage.

Usage:
    from experiment_tracker import ExperimentTracker, Experiment

    tracker = ExperimentTracker("experiments/")

    with tracker.start_run("fine-tune-v1", tags=["baseline"]) as run:
        run.log_params({"model": "all-MiniLM-L6-v2", "epochs": 5, "lr": 2e-5})
        run.log_metric("train_loss", 0.342, step=1)
        run.log_metric("train_loss", 0.198, step=2)
        run.log_metrics({"mrr": 0.87, "map": 0.82, "ndcg@5": 0.79})
        run.log_artifact("models/fine-tuned/config.json")
        run.set_model_version("1.0.0")

    # Compare experiments
    comparison = tracker.compare(["fine-tune-v1", "fine-tune-v2"])
    comparison.print_table()

    # Find best run
    best = tracker.best_run(metric="mrr", higher_is_better=True)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


@dataclass
class MetricEntry:
    """A single metric observation at a given step."""

    value: float
    step: int | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return {"value": self.value, "step": self.step, "timestamp": self.timestamp}


@dataclass
class RunRecord:
    """Complete record of an experiment run."""

    run_id: str
    name: str
    status: str = "created"  # created | running | completed | failed
    tags: list[str] = field(default_factory=list)

    # Timing
    start_time: str | None = None
    end_time: str | None = None
    duration_seconds: float | None = None

    # Data
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, list[dict]] = field(default_factory=dict)
    final_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    # Model versioning
    model_version: str | None = None
    model_hash: str | None = None
    parent_run_id: str | None = None

    # Notes
    description: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunRecord:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Active Run (context manager interface)
# ---------------------------------------------------------------------------


class ActiveRun:
    """
    An active experiment run that collects params, metrics, and artifacts.

    Use as a context manager via ``ExperimentTracker.start_run()``.
    Automatically records timing and finalizes on exit.
    """

    def __init__(self, record: RunRecord, storage_dir: Path):
        self._record = record
        self._storage_dir = storage_dir
        self._step_counter: dict[str, int] = {}

    @property
    def run_id(self) -> str:
        return self._record.run_id

    @property
    def name(self) -> str:
        return self._record.name

    @property
    def record(self) -> RunRecord:
        return self._record

    # -- Parameters -----------------------------------------------------------

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._record.params[key] = value
        logger.debug("Param: %s = %s", key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        self._record.params.update(params)
        logger.debug("Logged %d params", len(params))

    # -- Metrics --------------------------------------------------------------

    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a metric value, optionally at a specific step.

        If step is None, auto-increments from the last step for this key.
        The final (most recent) value is stored in ``final_metrics``.
        """
        if step is None:
            step = self._step_counter.get(key, 0)
            self._step_counter[key] = step + 1

        entry = MetricEntry(
            value=value,
            step=step,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if key not in self._record.metrics:
            self._record.metrics[key] = []
        self._record.metrics[key].append(entry.to_dict())
        self._record.final_metrics[key] = value

        logger.debug("Metric: %s = %.6f (step %d)", key, value, step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step=step)

    # -- Artifacts ------------------------------------------------------------

    def log_artifact(self, source_path: str, dest_name: str | None = None) -> str:
        """
        Copy an artifact file into the run's artifact directory.

        Args:
            source_path: Path to the source file.
            dest_name: Optional name for the artifact (defaults to filename).

        Returns:
            Path to the stored artifact.
        """
        source = Path(source_path)
        if not source.exists():
            logger.warning("Artifact not found: %s (recording path only)", source)
            self._record.artifacts.append(str(source))
            return str(source)

        artifact_dir = self._storage_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        dest = artifact_dir / (dest_name or source.name)
        shutil.copy2(source, dest)
        self._record.artifacts.append(str(dest))

        logger.debug("Artifact: %s -> %s", source, dest)
        return str(dest)

    # -- Model Versioning -----------------------------------------------------

    def set_model_version(self, version: str) -> None:
        """Set the model version for this run."""
        self._record.model_version = version

    def compute_model_hash(self, model_path: str) -> str:
        """
        Compute SHA-256 hash of a model directory for integrity tracking.

        Hashes all files in the directory to produce a single fingerprint.
        """
        model_dir = Path(model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        hasher = hashlib.sha256()

        if model_dir.is_file():
            files = [model_dir]
        else:
            files = sorted(model_dir.rglob("*"))
            files = [f for f in files if f.is_file()]

        for file_path in files:
            # Include relative path in hash for structure sensitivity
            rel_path = file_path.relative_to(model_dir) if model_dir.is_dir() else file_path.name
            hasher.update(str(rel_path).encode())
            hasher.update(file_path.read_bytes())

        model_hash = hasher.hexdigest()[:16]
        self._record.model_hash = model_hash
        logger.debug("Model hash: %s", model_hash)
        return model_hash

    # -- Notes ----------------------------------------------------------------

    def set_description(self, description: str) -> None:
        """Set a description for this run."""
        self._record.description = description

    def add_note(self, note: str) -> None:
        """Add a timestamped note to this run."""
        timestamped = f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {note}"
        self._record.notes.append(timestamped)

    # -- Lifecycle ------------------------------------------------------------

    def _start(self) -> None:
        """Mark the run as started."""
        self._record.status = "running"
        self._record.start_time = datetime.now(timezone.utc).isoformat()

    def _complete(self) -> None:
        """Mark the run as completed."""
        self._record.status = "completed"
        self._record.end_time = datetime.now(timezone.utc).isoformat()
        if self._record.start_time:
            start = datetime.fromisoformat(self._record.start_time)
            end = datetime.fromisoformat(self._record.end_time)
            self._record.duration_seconds = round((end - start).total_seconds(), 2)

    def _fail(self, error: str) -> None:
        """Mark the run as failed."""
        self._record.status = "failed"
        self._record.end_time = datetime.now(timezone.utc).isoformat()
        self.add_note(f"FAILED: {error}")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@dataclass
class RunComparison:
    """Side-by-side comparison of multiple experiment runs."""

    runs: list[RunRecord]
    metric_keys: list[str]
    param_keys: list[str]

    def to_dict(self) -> dict:
        rows = []
        for run in self.runs:
            row = {
                "name": run.name,
                "status": run.status,
                "model_version": run.model_version,
                "duration_s": run.duration_seconds,
            }
            for pk in self.param_keys:
                row[f"param:{pk}"] = run.params.get(pk, "—")
            for mk in self.metric_keys:
                row[f"metric:{mk}"] = run.final_metrics.get(mk, None)
            rows.append(row)
        return {"columns": list(rows[0].keys()) if rows else [], "rows": rows}

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        if not self.runs:
            print("No runs to compare.")
            return

        # Column widths
        name_w = max(len(r.name) for r in self.runs)
        name_w = max(name_w, 6)

        print(f"\n{'=' * 80}")
        print("  Experiment Comparison")
        print(f"{'=' * 80}")

        # Header
        header = f"  {'Run':<{name_w}}  {'Status':<10}"
        for pk in self.param_keys:
            header += f"  {pk:<12}"
        for mk in self.metric_keys:
            header += f"  {mk:>10}"
        print(header)

        sep = f"  {'—' * name_w}  {'—' * 10}"
        for _pk in self.param_keys:
            sep += f"  {'—' * 12}"
        for _mk in self.metric_keys:
            sep += f"  {'—' * 10}"
        print(sep)

        # Rows
        for run in self.runs:
            row = f"  {run.name:<{name_w}}  {run.status:<10}"
            for pk in self.param_keys:
                val = run.params.get(pk, "—")
                row += f"  {str(val):<12}"
            for mk in self.metric_keys:
                val = run.final_metrics.get(mk)
                if val is not None:
                    row += f"  {val:>10.4f}"
                else:
                    row += f"  {'—':>10}"
            print(row)

        print(f"{'=' * 80}\n")

    def diff(self, baseline_name: str) -> dict[str, dict[str, float | None]]:
        """
        Compute metric deltas relative to a baseline run.

        Returns:
            Dict mapping run name -> {metric_key: delta}.
            Positive delta means the run outperformed the baseline.
        """
        baseline = None
        for r in self.runs:
            if r.name == baseline_name:
                baseline = r
                break

        if baseline is None:
            raise ValueError(f"Baseline run '{baseline_name}' not found")

        diffs: dict[str, dict[str, float | None]] = {}
        for run in self.runs:
            if run.name == baseline_name:
                continue
            run_diff: dict[str, float | None] = {}
            for mk in self.metric_keys:
                base_val = baseline.final_metrics.get(mk)
                run_val = run.final_metrics.get(mk)
                if base_val is not None and run_val is not None:
                    run_diff[mk] = round(run_val - base_val, 6)
                else:
                    run_diff[mk] = None
            diffs[run.name] = run_diff

        return diffs


# ---------------------------------------------------------------------------
# Experiment Tracker
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """
    Lightweight experiment tracker persisted to the local filesystem.

    Stores each run as a JSON file under ``{base_dir}/runs/{run_id}.json``.
    Supports tagging, comparison, best-run selection, and model lineage.

    Design goals:
    - Zero external dependencies (no MLflow/W&B server)
    - Git-friendly (JSON files can be committed)
    - Integrates with training.py and evaluation.py

    Example::

        tracker = ExperimentTracker("experiments/")

        with tracker.start_run("baseline-v1") as run:
            run.log_params({"model": "all-MiniLM-L6-v2", "epochs": 5})
            for epoch in range(5):
                loss = train_one_epoch()
                run.log_metric("loss", loss, step=epoch)
            run.log_metrics({"mrr": 0.87, "map": 0.82})

        tracker.list_runs()
    """

    def __init__(self, base_dir: str | Path = "experiments"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs_cache: dict[str, RunRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        """Load all existing run records from disk."""
        for json_file in sorted(self.runs_dir.glob("*.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                record = RunRecord.from_dict(data)
                self._runs_cache[record.run_id] = record
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Skipping corrupted run file %s: %s", json_file, e)

        logger.info("Loaded %d existing runs from %s", len(self._runs_cache), self.runs_dir)

    def _save_run(self, record: RunRecord) -> None:
        """Persist a run record to disk."""
        path = self.runs_dir / f"{record.run_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, default=str)

    def _generate_run_id(self, name: str) -> str:
        """Generate a unique run ID from name + timestamp."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name_slug = name.lower().replace(" ", "-").replace("/", "-")[:30]
        return f"{name_slug}_{ts}"

    # -- Run Lifecycle --------------------------------------------------------

    @contextmanager
    def start_run(
        self,
        name: str,
        tags: list[str] | None = None,
        description: str | None = None,
        parent_run_id: str | None = None,
    ):
        """
        Start a new experiment run as a context manager.

        Args:
            name: Human-readable name for this run.
            tags: Optional tags for filtering/grouping.
            description: Optional description.
            parent_run_id: Link to a parent run (for nested experiments).

        Yields:
            ActiveRun instance for logging params/metrics/artifacts.
        """
        run_id = self._generate_run_id(name)
        record = RunRecord(
            run_id=run_id,
            name=name,
            tags=tags or [],
            description=description,
            parent_run_id=parent_run_id,
        )

        storage_dir = self.runs_dir / run_id
        storage_dir.mkdir(parents=True, exist_ok=True)

        active = ActiveRun(record, storage_dir)
        active._start()

        logger.info("Started run: %s (%s)", name, run_id)

        try:
            yield active
            active._complete()
            logger.info(
                "Completed run: %s (%.1fs, %d metrics)",
                name,
                record.duration_seconds or 0,
                len(record.final_metrics),
            )
        except Exception as e:
            active._fail(str(e))
            logger.error("Run failed: %s — %s", name, e)
            raise
        finally:
            self._runs_cache[run_id] = record
            self._save_run(record)

    # -- Querying -------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a run by ID."""
        return self._runs_cache.get(run_id)

    def get_run_by_name(self, name: str) -> RunRecord | None:
        """Get the most recent run with the given name."""
        matches = [r for r in self._runs_cache.values() if r.name == name]
        if not matches:
            return None
        return max(matches, key=lambda r: r.start_time or "")

    def list_runs(
        self,
        status: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        """
        List runs with optional filters.

        Args:
            status: Filter by status (completed, failed, running).
            tags: Filter by tags (any match).
            limit: Maximum number of runs to return.

        Returns:
            List of RunRecords sorted by start time (newest first).
        """
        runs = list(self._runs_cache.values())

        if status:
            runs = [r for r in runs if r.status == status]

        if tags:
            tag_set = set(tags)
            runs = [r for r in runs if tag_set & set(r.tags)]

        # Sort newest first
        runs.sort(key=lambda r: r.start_time or "", reverse=True)

        if limit:
            runs = runs[:limit]

        return runs

    def delete_run(self, run_id: str) -> bool:
        """Delete a run record and its artifacts."""
        if run_id not in self._runs_cache:
            return False

        # Remove from cache
        del self._runs_cache[run_id]

        # Remove files
        run_file = self.runs_dir / f"{run_id}.json"
        if run_file.exists():
            run_file.unlink()

        artifacts_dir = self.runs_dir / run_id
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)

        logger.info("Deleted run: %s", run_id)
        return True

    # -- Comparison -----------------------------------------------------------

    def compare(
        self,
        names_or_ids: list[str],
        metric_keys: list[str] | None = None,
        param_keys: list[str] | None = None,
    ) -> RunComparison:
        """
        Compare multiple runs side-by-side.

        Args:
            names_or_ids: List of run names or IDs to compare.
            metric_keys: Specific metrics to include (None = all).
            param_keys: Specific params to include (None = all).

        Returns:
            RunComparison object with tabular output support.
        """
        runs: list[RunRecord] = []
        for identifier in names_or_ids:
            run = self.get_run(identifier) or self.get_run_by_name(identifier)
            if run:
                runs.append(run)
            else:
                logger.warning("Run not found: %s", identifier)

        if not runs:
            return RunComparison(runs=[], metric_keys=[], param_keys=[])

        # Auto-discover keys if not specified
        if metric_keys is None:
            all_mk: set = set()
            for r in runs:
                all_mk.update(r.final_metrics.keys())
            metric_keys = sorted(all_mk)

        if param_keys is None:
            all_pk: set = set()
            for r in runs:
                all_pk.update(r.params.keys())
            param_keys = sorted(all_pk)

        return RunComparison(runs=runs, metric_keys=metric_keys, param_keys=param_keys)

    # -- Best Run Selection ---------------------------------------------------

    def best_run(
        self,
        metric: str,
        higher_is_better: bool = True,
        status: str = "completed",
        tags: list[str] | None = None,
    ) -> RunRecord | None:
        """
        Find the best run by a specific metric.

        Args:
            metric: Metric key to optimize.
            higher_is_better: Whether higher values are better.
            status: Only consider runs with this status.
            tags: Optional tag filter.

        Returns:
            The best RunRecord, or None if no matching runs.
        """
        candidates = self.list_runs(status=status, tags=tags)
        candidates = [r for r in candidates if metric in r.final_metrics]

        if not candidates:
            return None

        return (max if higher_is_better else min)(candidates, key=lambda r: r.final_metrics[metric])

    # -- Metric History -------------------------------------------------------

    def metric_history(
        self,
        run_name_or_id: str,
        metric_key: str,
    ) -> list[tuple[int, float]]:
        """
        Get the step-by-step history of a metric for a run.

        Returns:
            List of (step, value) tuples.
        """
        run = self.get_run(run_name_or_id) or self.get_run_by_name(run_name_or_id)
        if not run or metric_key not in run.metrics:
            return []

        return [
            (entry.get("step", i), entry["value"])
            for i, entry in enumerate(run.metrics[metric_key])
        ]

    # -- Model Registry -------------------------------------------------------

    def model_lineage(self, run_name_or_id: str) -> list[RunRecord]:
        """
        Trace the model lineage from a run back through its parents.

        Returns:
            List of RunRecords from newest to oldest in the lineage chain.
        """
        chain: list[RunRecord] = []
        current = self.get_run(run_name_or_id) or self.get_run_by_name(run_name_or_id)

        visited: set = set()
        while current and current.run_id not in visited:
            chain.append(current)
            visited.add(current.run_id)
            if current.parent_run_id:
                current = self.get_run(current.parent_run_id)
            else:
                break

        return chain

    def latest_model(
        self,
        tags: list[str] | None = None,
    ) -> RunRecord | None:
        """
        Get the most recent completed run with a model version.

        Useful for deployment pipelines that need the latest trained model.
        """
        runs = self.list_runs(status="completed", tags=tags)
        for run in runs:
            if run.model_version:
                return run
        return None

    # -- Export / Import ------------------------------------------------------

    def export_summary(self, path: str | Path) -> None:
        """
        Export a summary of all runs as a single JSON file.

        Useful for dashboards, CI reports, or sharing results.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        runs = self.list_runs()
        summary = {
            "total_runs": len(runs),
            "completed": sum(1 for r in runs if r.status == "completed"),
            "failed": sum(1 for r in runs if r.status == "failed"),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "runs": [
                {
                    "run_id": r.run_id,
                    "name": r.name,
                    "status": r.status,
                    "tags": r.tags,
                    "start_time": r.start_time,
                    "duration_seconds": r.duration_seconds,
                    "final_metrics": r.final_metrics,
                    "model_version": r.model_version,
                    "params": r.params,
                }
                for r in runs
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Exported summary of %d runs to %s", len(runs), path)

    # -- Utilities ------------------------------------------------------------

    @property
    def run_count(self) -> int:
        """Total number of tracked runs."""
        return len(self._runs_cache)

    def __repr__(self) -> str:
        completed = sum(1 for r in self._runs_cache.values() if r.status == "completed")
        return (
            f"ExperimentTracker(dir='{self.base_dir}', "
            f"runs={self.run_count}, completed={completed})"
        )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Experiment tracker CLI")
    sub = parser.add_subparsers(dest="command")

    # List runs
    list_p = sub.add_parser("list", help="List experiment runs")
    list_p.add_argument("--status", choices=["completed", "failed", "running"])
    list_p.add_argument("--tags", nargs="+")
    list_p.add_argument("--limit", type=int, default=20)
    list_p.add_argument("--dir", default="experiments")

    # Compare runs
    cmp_p = sub.add_parser("compare", help="Compare experiment runs")
    cmp_p.add_argument("names", nargs="+", help="Run names or IDs")
    cmp_p.add_argument("--dir", default="experiments")

    # Best run
    best_p = sub.add_parser("best", help="Find the best run by metric")
    best_p.add_argument("metric", help="Metric key")
    best_p.add_argument("--lower", action="store_true", help="Lower is better")
    best_p.add_argument("--dir", default="experiments")

    # Export
    exp_p = sub.add_parser("export", help="Export run summary")
    exp_p.add_argument("--output", default="experiment_summary.json")
    exp_p.add_argument("--dir", default="experiments")

    args = parser.parse_args()

    if args.command == "list":
        tracker = ExperimentTracker(args.dir)
        runs = tracker.list_runs(status=args.status, tags=args.tags, limit=args.limit)
        print(f"\n{'=' * 70}")
        print(f"  Experiment Runs ({len(runs)} shown)")
        print(f"{'=' * 70}")
        for r in runs:
            tags = f" [{', '.join(r.tags)}]" if r.tags else ""
            dur = f" ({r.duration_seconds:.1f}s)" if r.duration_seconds else ""
            ver = f" v{r.model_version}" if r.model_version else ""
            print(f"  [{r.status:>9}] {r.name}{tags}{ver}{dur}")
            if r.final_metrics:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(r.final_metrics.items()))
                print(f"             {metrics_str}")
        print()

    elif args.command == "compare":
        tracker = ExperimentTracker(args.dir)
        comparison = tracker.compare(args.names)
        comparison.print_table()

    elif args.command == "best":
        tracker = ExperimentTracker(args.dir)
        best = tracker.best_run(args.metric, higher_is_better=not args.lower)
        if best:
            val = best.final_metrics.get(args.metric, "?")
            print(f"\nBest run for {args.metric}: {best.name} = {val}")
            print(f"  Run ID: {best.run_id}")
            if best.model_version:
                print(f"  Model version: {best.model_version}")
        else:
            print(f"No completed runs with metric '{args.metric}'")

    elif args.command == "export":
        tracker = ExperimentTracker(args.dir)
        tracker.export_summary(args.output)
        print(f"Exported to {args.output}")

    else:
        parser.print_help()
