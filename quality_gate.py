"""
Retrieval Quality Gate
======================
A CI-friendly regression guardrail for semantic search.  Given a *baseline*
evaluation report (committed alongside the code) and a *current* evaluation
report produced by the latest pipeline, this module decides whether retrieval
quality has regressed enough to fail the build.

Design goals
------------

* **Model-agnostic.**  The gate operates on serialized
  :class:`evaluation.EvalReport` payloads (or anything dict-shaped with the
  same keys).  It does not load embedding models, indices, or torch — making
  it cheap to run in CI and trivially testable without network access.
* **Multi-signal threshold logic.**  A regression is flagged when *either*
  the absolute drop *or* the relative drop crosses its limit, *or* the metric
  falls below a hard floor.  This catches both proportionally large
  regressions on small metrics and small-but-material drops on already-high
  metrics.
* **Per-query diagnostics.**  Aggregate metrics hide *which* queries got
  worse.  When per-query data is present, the gate identifies queries whose
  reciprocal-rank, average-precision, or NDCG@k dropped meaningfully and
  surfaces them in the report.
* **PR-friendly Markdown output.**  ``render_markdown()`` emits a compact
  summary table suitable for posting as a pull-request comment via GitHub
  Actions.

Example::

    from evaluation import EvalReport
    from quality_gate import QualityGate, GateConfig

    gate = QualityGate(GateConfig.default())
    result = gate.compare(baseline_report, current_report)
    print(result.render_markdown())
    if not result.passed:
        sys.exit(1)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# Default per-metric tolerances.  Values are tuned for typical small/medium
# eval sets; override per-project via :class:`GateConfig`.
_DEFAULT_ABS_DROP = 0.02  # absolute points (e.g. 0.85 → 0.83)
_DEFAULT_REL_DROP = 0.05  # 5% of baseline value
_DEFAULT_MIN_VALUE = 0.0  # disabled by default

#: The metrics the gate watches.  Headline metrics get top-level entries; per-k
#: metrics use ``ndcg@5``, ``precision@10`` style keys to stay flat.
HEADLINE_METRICS = ("mrr", "map")
PER_K_METRICS = ("ndcg", "precision", "recall")


@dataclass(frozen=True)
class Threshold:
    """Per-metric tolerance.

    A regression is flagged on any of these conditions::

        current < baseline - abs_drop
        current < baseline * (1 - rel_drop)
        current < min_value
    """

    abs_drop: float = _DEFAULT_ABS_DROP
    rel_drop: float = _DEFAULT_REL_DROP
    min_value: float = _DEFAULT_MIN_VALUE

    def regression(self, baseline: float, current: float) -> tuple[bool, str | None]:
        """Return ``(is_regression, reason_or_none)``."""
        if current < self.min_value:
            return True, f"below floor {self.min_value:.4f}"
        if current < baseline - self.abs_drop:
            return True, f"absolute drop > {self.abs_drop:.4f}"
        if baseline > 0 and current < baseline * (1 - self.rel_drop):
            pct = self.rel_drop * 100
            return True, f"relative drop > {pct:.1f}%"
        return False, None


@dataclass
class GateConfig:
    """Per-metric thresholds plus per-query regression sensitivity."""

    thresholds: dict[str, Threshold] = field(default_factory=dict)
    default_threshold: Threshold = field(default_factory=Threshold)

    #: When >0, any individual query whose reciprocal-rank / AP / NDCG@k drops
    #: by more than this absolute amount is reported as a per-query
    #: regression.  Set to 0 to disable per-query checks.
    per_query_drop: float = 0.10

    #: Maximum number of regressing queries to list in the markdown summary.
    per_query_max_listed: int = 5

    @classmethod
    def default(cls) -> GateConfig:
        return cls()

    @classmethod
    def strict(cls) -> GateConfig:
        """Tighter defaults for high-stakes pipelines."""
        return cls(
            default_threshold=Threshold(abs_drop=0.01, rel_drop=0.02),
            per_query_drop=0.05,
        )

    def threshold_for(self, metric_key: str) -> Threshold:
        return self.thresholds.get(metric_key, self.default_threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": {k: asdict(v) for k, v in self.thresholds.items()},
            "default_threshold": asdict(self.default_threshold),
            "per_query_drop": self.per_query_drop,
            "per_query_max_listed": self.per_query_max_listed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GateConfig:
        thresholds = {k: Threshold(**v) for k, v in (payload.get("thresholds") or {}).items()}
        # Accept both the new "default_threshold" key and the legacy "default"
        # alias for backwards compatibility with hand-edited config files.
        default_payload = payload.get("default_threshold") or payload.get("default")
        default = Threshold(**default_payload) if default_payload else Threshold()
        return cls(
            thresholds=thresholds,
            default_threshold=default,
            per_query_drop=float(payload.get("per_query_drop", 0.10)),
            per_query_max_listed=int(payload.get("per_query_max_listed", 5)),
        )

    @classmethod
    def load(cls, path: str | Path) -> GateConfig:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class MetricChange:
    """Change for a single (metric, optional k) pair."""

    metric: str
    k: int | None
    baseline: float
    current: float
    delta: float
    pct_delta: float
    threshold: Threshold
    regressed: bool
    reason: str | None

    @property
    def key(self) -> str:
        return f"{self.metric}@{self.k}" if self.k is not None else self.metric

    @property
    def status_glyph(self) -> str:
        if self.regressed:
            return "FAIL"
        if self.delta > 1e-6:
            return "UP"
        if self.delta < -1e-6:
            return "DOWN"
        return "OK"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "k": self.k,
            "baseline": self.baseline,
            "current": self.current,
            "delta": self.delta,
            "pct_delta": self.pct_delta,
            "regressed": self.regressed,
            "reason": self.reason,
        }


@dataclass
class QueryRegression:
    """A single query whose retrieval quality dropped beyond the per-query
    threshold."""

    query: str
    metric: str
    baseline: float
    current: float
    delta: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    """Outcome of comparing two evaluation reports."""

    passed: bool
    changes: list[MetricChange]
    regressing_queries: list[QueryRegression]
    baseline_model: str | None
    current_model: str | None
    num_queries_baseline: int
    num_queries_current: int
    generated_at: str

    @property
    def regressed_metrics(self) -> list[MetricChange]:
        return [c for c in self.changes if c.regressed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "baseline_model": self.baseline_model,
            "current_model": self.current_model,
            "num_queries_baseline": self.num_queries_baseline,
            "num_queries_current": self.num_queries_current,
            "generated_at": self.generated_at,
            "changes": [c.to_dict() for c in self.changes],
            "regressing_queries": [q.to_dict() for q in self.regressing_queries],
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_markdown(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown())

    # ------------------------------------------------------------------
    # Renderers
    # ------------------------------------------------------------------

    def render_markdown(self) -> str:
        """Render a PR-friendly Markdown summary."""
        title = "PASS" if self.passed else "FAIL"
        lines: list[str] = []
        lines.append(f"## Retrieval quality gate — **{title}**")
        lines.append("")
        meta_bits = []
        if self.baseline_model and self.current_model:
            if self.baseline_model == self.current_model:
                meta_bits.append(f"model `{self.current_model}`")
            else:
                meta_bits.append(f"model `{self.baseline_model}` → `{self.current_model}`")
        meta_bits.append(f"queries {self.num_queries_baseline} → {self.num_queries_current}")
        meta_bits.append(self.generated_at)
        lines.append("_" + " · ".join(meta_bits) + "_")
        lines.append("")

        lines.append("| Metric | Baseline | Current | Δ | Δ% | Status |")
        lines.append("| --- | ---: | ---: | ---: | ---: | :---: |")
        for change in self.changes:
            delta_str = f"{change.delta:+.4f}"
            pct_str = f"{change.pct_delta:+.1f}%" if change.baseline > 0 else "n/a"
            status = change.status_glyph
            if change.regressed and change.reason:
                status = f"FAIL ({change.reason})"
            lines.append(
                f"| `{change.key}` | {change.baseline:.4f} | {change.current:.4f}"
                f" | {delta_str} | {pct_str} | {status} |"
            )

        if self.regressing_queries:
            lines.append("")
            n = len(self.regressing_queries)
            lines.append(f"### Per-query regressions ({n})")
            lines.append("")
            lines.append("| Query | Metric | Baseline | Current | Δ |")
            lines.append("| --- | --- | ---: | ---: | ---: |")
            for q in self.regressing_queries:
                preview = q.query if len(q.query) <= 80 else q.query[:77] + "..."
                preview = preview.replace("|", "\\|")
                lines.append(
                    f"| {preview} | `{q.metric}` | {q.baseline:.4f}"
                    f" | {q.current:.4f} | {q.delta:+.4f} |"
                )

        if not self.passed:
            lines.append("")
            lines.append(
                "> Quality gate failed: resolve the regressions above or "
                "intentionally update the baseline with `search-cli "
                "quality-gate --update-baseline`."
            )

        return "\n".join(lines) + "\n"

    def render_text(self) -> str:
        """Plain-text summary for stdout/CI logs."""
        verdict = "PASS" if self.passed else "FAIL"
        lines = [f"Retrieval quality gate: {verdict}"]
        for change in self.changes:
            arrow = "↓" if change.delta < 0 else ("↑" if change.delta > 0 else "=")
            tag = " [REGRESSION]" if change.regressed else ""
            lines.append(
                f"  {change.key:<14} {change.baseline:.4f} -> "
                f"{change.current:.4f} ({change.delta:+.4f}, {arrow}){tag}"
                + (f"  // {change.reason}" if change.reason else "")
            )
        if self.regressing_queries:
            lines.append("")
            lines.append("Per-query regressions:")
            for q in self.regressing_queries:
                preview = q.query if len(q.query) <= 60 else q.query[:57] + "..."
                lines.append(
                    f"  - [{q.metric}] {q.baseline:.4f} -> {q.current:.4f}"
                    f" ({q.delta:+.4f})  {preview!r}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def _coerce_k_dict(payload: Any) -> dict[int, float]:
    """JSON int keys are serialised as strings; coerce them back."""
    if not payload:
        return {}
    out: dict[int, float] = {}
    for k, v in payload.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return out


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class QualityGate:
    """Compare two evaluation reports and produce a pass/fail verdict."""

    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig.default()

    # -- public API ----------------------------------------------------

    def compare(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> GateResult:
        """Compare two ``EvalReport.to_dict()`` payloads."""
        changes: list[MetricChange] = []

        for metric in HEADLINE_METRICS:
            base_v = float(baseline.get(metric, 0.0))
            curr_v = float(current.get(metric, 0.0))
            changes.append(self._build_change(metric, None, base_v, curr_v))

        # Honour the union of k-values seen in either report so a wider
        # current report still gets compared against the baseline at shared k.
        for metric in PER_K_METRICS:
            base_k = _coerce_k_dict(baseline.get(metric))
            curr_k = _coerce_k_dict(current.get(metric))
            shared_k = sorted(set(base_k) & set(curr_k))
            for k in shared_k:
                changes.append(self._build_change(metric, k, base_k[k], curr_k[k]))

        regressing_queries = self._per_query_regressions(baseline, current)

        passed = not any(c.regressed for c in changes)

        return GateResult(
            passed=passed,
            changes=changes,
            regressing_queries=regressing_queries,
            baseline_model=baseline.get("model_name"),
            current_model=current.get("model_name"),
            num_queries_baseline=int(baseline.get("num_queries", 0)),
            num_queries_current=int(current.get("num_queries", 0)),
            generated_at=_utcnow_iso(),
        )

    # -- helpers -------------------------------------------------------

    def _build_change(
        self,
        metric: str,
        k: int | None,
        baseline: float,
        current: float,
    ) -> MetricChange:
        key = f"{metric}@{k}" if k is not None else metric
        threshold = self.config.threshold_for(key)
        delta = current - baseline
        pct = (delta / baseline) * 100.0 if baseline > 0 else 0.0
        regressed, reason = threshold.regression(baseline, current)
        return MetricChange(
            metric=metric,
            k=k,
            baseline=round(baseline, 6),
            current=round(current, 6),
            delta=round(delta, 6),
            pct_delta=round(pct, 4),
            threshold=threshold,
            regressed=regressed,
            reason=reason,
        )

    def _per_query_regressions(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
    ) -> list[QueryRegression]:
        if self.config.per_query_drop <= 0:
            return []
        base_per = baseline.get("per_query") or []
        curr_per = current.get("per_query") or []
        if not base_per or not curr_per:
            return []

        # Index baseline rows by their query string (the natural identifier
        # for our EvalReport payloads).  Duplicate queries are unusual; if
        # they appear we keep the first occurrence and skip the rest.
        base_by_query: dict[str, dict[str, Any]] = {}
        for row in base_per:
            q = row.get("query")
            if isinstance(q, str) and q not in base_by_query:
                base_by_query[q] = row

        regressions: list[QueryRegression] = []
        threshold_drop = self.config.per_query_drop
        for row in curr_per:
            q = row.get("query")
            base_row = base_by_query.get(q) if isinstance(q, str) else None
            if base_row is None:
                continue
            for metric in ("reciprocal_rank", "average_precision"):
                base_v = float(base_row.get(metric, 0.0))
                curr_v = float(row.get(metric, 0.0))
                if base_v - curr_v > threshold_drop:
                    regressions.append(
                        QueryRegression(
                            query=q,
                            metric=metric,
                            baseline=round(base_v, 4),
                            current=round(curr_v, 4),
                            delta=round(curr_v - base_v, 4),
                        )
                    )
            base_ndcg = _coerce_k_dict(base_row.get("ndcg"))
            curr_ndcg = _coerce_k_dict(row.get("ndcg"))
            for k in sorted(set(base_ndcg) & set(curr_ndcg)):
                if base_ndcg[k] - curr_ndcg[k] > threshold_drop:
                    regressions.append(
                        QueryRegression(
                            query=q,
                            metric=f"ndcg@{k}",
                            baseline=round(base_ndcg[k], 4),
                            current=round(curr_ndcg[k], 4),
                            delta=round(curr_ndcg[k] - base_ndcg[k], 4),
                        )
                    )

        # Sort by largest drop first, then truncate to the configured limit
        # so PR comments stay readable even on noisy regressions.
        regressions.sort(key=lambda r: r.delta)
        return regressions[: self.config.per_query_max_listed]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a serialized evaluation report (or a baseline report) from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_baseline(report: dict[str, Any], path: str | Path) -> None:
    """Persist a report as a baseline under ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
