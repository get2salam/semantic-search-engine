#!/usr/bin/env python3
"""
Command-line interface for the semantic search engine
======================================================
A thin, scriptable wrapper around ``SemanticSearchEngine`` suitable for
shell pipelines and ad-hoc exploration. Ships four subcommands:

    index     Build a persistent index from a text/JSONL file
    search    Run one or more queries against a saved index
    info      Print stats about a saved index
    version   Print the engine version

Examples:
    # Build an index from a newline-delimited file
    python cli.py index --input docs.txt --output ./my-index

    # Query it
    python cli.py search --index ./my-index --query "machine learning" --top-k 5

    # Pipe multiple queries (one per line) for batch mode
    echo -e "AI models\\nweb development" | python cli.py search --index ./my-index --stdin

    # JSON output for shell scripting
    python cli.py search --index ./my-index --query "python" --json | jq .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from importlib import metadata as _metadata
except ImportError:  # pragma: no cover - py<3.8 guard (project is 3.10+)
    _metadata = None  # type: ignore

from config import get_settings


def _get_version() -> str:
    try:
        return _metadata.version("semantic-search-engine") if _metadata else "unknown"
    except Exception:
        return get_settings().app_version


def _load_documents(path: Path) -> list[str]:
    """
    Load documents from either a plain text file (one per line) or a
    JSONL file where each line is either a string or an object with a
    ``"text"`` field.
    """
    docs: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
                if isinstance(obj, str):
                    docs.append(obj)
                elif isinstance(obj, dict) and "text" in obj:
                    docs.append(str(obj["text"]))
                else:
                    raise ValueError(
                        f"{path}:{lineno}: JSONL entry must be a string or object with 'text'"
                    )
            else:
                docs.append(line)
    return docs


def _format_results(query: str, results, as_json: bool) -> str:
    if as_json:
        payload = {
            "query": query,
            "results": [
                {"rank": i + 1, "score": round(score, 4), "document": doc}
                for i, (doc, score) in enumerate(results)
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    if not results:
        return f"No results for {query!r}"

    lines = [f"Query: {query!r}", "-" * 60]
    for i, (doc, score) in enumerate(results, 1):
        lines.append(f"  {i}. [{score:.3f}] {doc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    from semantic_search import SemanticSearchEngine

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 2

    docs = _load_documents(input_path)
    if not docs:
        print(f"error: no documents found in {input_path}", file=sys.stderr)
        return 2

    engine = SemanticSearchEngine(model_name=args.model, use_faiss=not args.no_faiss)
    engine.add_documents(docs, batch_size=args.batch_size, show_progress=not args.quiet)

    output = Path(args.output)
    engine.save(output)
    print(f"Indexed {len(docs)} documents → {output}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from semantic_search import SemanticSearchEngine

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"error: index directory not found: {index_path}", file=sys.stderr)
        return 2

    engine = SemanticSearchEngine.load(index_path)

    queries: list[str] = []
    if args.query:
        queries.append(args.query)
    if args.stdin:
        for raw in sys.stdin:
            line = raw.strip()
            if line:
                queries.append(line)

    if not queries:
        print("error: provide --query or --stdin", file=sys.stderr)
        return 2

    separator_needed = len(queries) > 1 and not args.json
    for i, q in enumerate(queries):
        results = engine.search(
            q,
            top_k=args.top_k,
            threshold=args.threshold,
            mmr_lambda=args.mmr_lambda,
        )
        if separator_needed and i > 0:
            print()
        print(_format_results(q, results, as_json=args.json))

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from semantic_search import SemanticSearchEngine

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"error: index directory not found: {index_path}", file=sys.stderr)
        return 2

    engine = SemanticSearchEngine.load(index_path)
    payload = {
        "path": str(index_path),
        "documents": len(engine),
        "model": engine.model_name,
        "embedding_dim": engine.embedding_dim,
        "faiss_enabled": engine.use_faiss,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Index: {payload['path']}")
        print(f"  Documents:     {payload['documents']}")
        print(f"  Model:         {payload['model']}")
        print(f"  Embedding dim: {payload['embedding_dim']}")
        print(f"  FAISS:         {'yes' if payload['faiss_enabled'] else 'no'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Measure retrieval quality against a BEIR-style dataset."""
    from eval_data import load_beir_like
    from evaluation import RetrievalEvaluator
    from semantic_search import SemanticSearchEngine

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"error: dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    ds = load_beir_like(dataset_dir)
    if not ds.queries:
        print(f"error: no queries with relevance judgments in {dataset_dir}", file=sys.stderr)
        return 2

    engine = SemanticSearchEngine(model_name=args.model, use_faiss=not args.no_faiss)
    engine.add_documents(ds.corpus_texts(), show_progress=not args.quiet)

    # Map back retrieved texts to their doc ids
    text_to_id = {text: doc_id for doc_id, text in ds.corpus.items()}

    def search_fn(query: str, k: int) -> list[str]:
        results = engine.search(query, top_k=k)
        return [text_to_id.get(doc, "") for doc, _ in results]

    k_values = [int(k) for k in args.k.split(",")]
    evaluator = RetrievalEvaluator()
    evaluator.add_queries(ds.queries)
    report = evaluator.evaluate(search_fn=search_fn, k_values=k_values, model_name=args.model)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    if args.output:
        report.save(args.output)
        print(f"Report saved to {args.output}")

    return 0


def cmd_quality_gate(args: argparse.Namespace) -> int:
    """Compare a current evaluation report against a committed baseline.

    Exit codes:
        0  baseline updated, or comparison passed.
        1  comparison detected a regression beyond configured thresholds.
        2  usage / I/O error (missing files, malformed JSON, ...).
    """
    from quality_gate import GateConfig, QualityGate, load_report, write_baseline

    baseline_path = Path(args.baseline)
    report_path = Path(args.report)

    if not report_path.exists():
        print(f"error: report file not found: {report_path}", file=sys.stderr)
        return 2

    try:
        current = load_report(report_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: failed to read report {report_path}: {exc}", file=sys.stderr)
        return 2

    # Bootstrap path: if a baseline does not exist yet, --update-baseline
    # creates it from the current report. Without that flag we can't
    # compare, so fail loudly rather than silently passing.
    if args.update_baseline:
        write_baseline(current, baseline_path)
        print(f"Baseline updated → {baseline_path}")
        return 0

    if not baseline_path.exists():
        print(
            f"error: baseline not found: {baseline_path}\n"
            "  hint: bootstrap one with --update-baseline",
            file=sys.stderr,
        )
        return 2

    try:
        baseline = load_report(baseline_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"error: failed to read baseline {baseline_path}: {exc}",
            file=sys.stderr,
        )
        return 2

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"error: config file not found: {config_path}", file=sys.stderr)
            return 2
        config = GateConfig.load(config_path)
    elif args.strict:
        config = GateConfig.strict()
    else:
        config = GateConfig.default()

    result = QualityGate(config).compare(baseline, current)

    if args.markdown:
        result.save_markdown(args.markdown)
    if args.output:
        result.save_json(args.output)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.render_text())

    return 0 if result.passed else 1


def _load_metric_vectors(path: Path) -> dict[str, list[float]]:
    """Load ``metric -> per-query scores`` from a compact JSON file."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("metric file must be a JSON object")

    vectors: dict[str, list[float]] = {}
    for metric, values in payload.items():
        if not isinstance(metric, str):
            raise ValueError("metric names must be strings")
        if not isinstance(values, list):
            raise ValueError(f"metric {metric!r} must be a list of numbers")
        try:
            vectors[metric] = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"metric {metric!r} contains a non-numeric value") from exc
    return vectors


def cmd_ab_compare(args: argparse.Namespace) -> int:
    """Compare two systems from per-query metric vectors."""
    from abtest import compare_systems

    try:
        metrics_a = _load_metric_vectors(Path(args.a))
        metrics_b = _load_metric_vectors(Path(args.b))
        report = compare_systems(
            metrics_a,
            metrics_b,
            name_a=args.name_a,
            name_b=args.name_b,
            confidence=args.confidence,
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: failed to build A/B report: {exc}", file=sys.stderr)
        return 2

    if args.markdown:
        Path(args.markdown).write_text(report.to_markdown(alpha=args.alpha), encoding="utf-8")
    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    elif args.markdown_stdout:
        print(report.to_markdown(alpha=args.alpha))
    else:
        print("\n".join(report.summary_lines(alpha=args.alpha)))

    return 0


def _load_score_label_json(path: Path) -> tuple[list[float], list[int]]:
    """Load calibration inputs from JSON.

    Accepted shapes:
    - ``{"scores": [...], "labels": [...]}``
    - ``[{"score": 0.7, "label": 1}, ...]``
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "scores" in payload and "labels" in payload:
        scores_raw = payload["scores"]
        labels_raw = payload["labels"]
    elif isinstance(payload, list):
        scores_raw = [row["score"] for row in payload]
        labels_raw = [row["label"] for row in payload]
    else:
        raise ValueError("expected {'scores': [...], 'labels': [...]} or rows with score/label")

    try:
        return [float(v) for v in scores_raw], [int(v) for v in labels_raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("scores and labels must be numeric") from exc


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Build score calibration diagnostics from labelled retrieval examples."""
    from calibration import calibration_report

    try:
        scores, labels = _load_score_label_json(Path(args.input))
        report = calibration_report(scores, labels, n_bins=args.bins)
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"error: failed to build calibration report: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n".join(report.summary_lines()))

    return 0


def _load_number_vector(path: Path) -> list[float]:
    """Load a numeric vector from JSON list or ``{"values": [...]}``."""
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "values" in payload:
        payload = payload["values"]
    if not isinstance(payload, list):
        raise ValueError("expected a JSON list or {'values': [...]} object")
    try:
        return [float(v) for v in payload]
    except (TypeError, ValueError) as exc:
        raise ValueError("all values must be numeric") from exc


def cmd_drift_report(args: argparse.Namespace) -> int:
    """Compare baseline/current numeric distributions for drift."""
    from drift import population_stability_index

    try:
        baseline = _load_number_vector(Path(args.baseline))
        current = _load_number_vector(Path(args.current))
        report = population_stability_index(baseline, current, n_bins=args.bins)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: failed to build drift report: {exc}", file=sys.stderr)
        return 2

    if args.output:
        Path(args.output).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n".join(report.summary_lines()))

    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    print(_get_version())
    return 0


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="search-cli",
        description="Command-line interface for the semantic search engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # index
    p_index = sub.add_parser("index", help="Build and save an index from a file")
    p_index.add_argument("--input", "-i", required=True, help="Text or JSONL input file")
    p_index.add_argument("--output", "-o", required=True, help="Directory to save the index to")
    p_index.add_argument(
        "--model", "-m", default=get_settings().model_name, help="Sentence-transformer model"
    )
    p_index.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    p_index.add_argument("--no-faiss", action="store_true", help="Disable FAISS backend")
    p_index.add_argument("--quiet", "-q", action="store_true", help="Suppress progress bars")
    p_index.set_defaults(func=cmd_index)

    # search
    p_search = sub.add_parser("search", help="Run one or more queries against an index")
    p_search.add_argument("--index", "-i", required=True, help="Path to a saved index directory")
    p_search.add_argument("--query", "-q", help="A single query string")
    p_search.add_argument(
        "--stdin", action="store_true", help="Read additional queries from stdin (one per line)"
    )
    p_search.add_argument("--top-k", "-k", type=int, default=5, help="Results per query")
    p_search.add_argument(
        "--threshold", type=float, default=None, help="Minimum similarity score (0-1)"
    )
    p_search.add_argument(
        "--mmr-lambda",
        type=float,
        default=None,
        help="Enable MMR diversification (0=diverse, 1=relevance)",
    )
    p_search.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p_search.set_defaults(func=cmd_search)

    # info
    p_info = sub.add_parser("info", help="Inspect a saved index")
    p_info.add_argument("--index", "-i", required=True, help="Path to a saved index directory")
    p_info.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p_info.set_defaults(func=cmd_info)

    # evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Measure retrieval quality against a BEIR-style TSV dataset",
    )
    p_eval.add_argument(
        "--dataset", "-d", required=True, help="Directory containing corpus/queries/qrels TSVs"
    )
    p_eval.add_argument(
        "--model", "-m", default=get_settings().model_name, help="Sentence-transformer model"
    )
    p_eval.add_argument("--k", default="1,3,5,10", help="Comma-separated list of k values")
    p_eval.add_argument("--output", "-o", default=None, help="Write JSON report to this path")
    p_eval.add_argument("--no-faiss", action="store_true", help="Disable FAISS backend")
    p_eval.add_argument("--quiet", "-q", action="store_true", help="Suppress progress bars")
    p_eval.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    p_eval.set_defaults(func=cmd_evaluate)

    # quality-gate
    p_gate = sub.add_parser(
        "quality-gate",
        help="Compare a retrieval report against a committed baseline",
    )
    p_gate.add_argument(
        "--baseline",
        "-b",
        required=True,
        help="Path to the committed baseline JSON (produced by `evaluate`)",
    )
    p_gate.add_argument(
        "--report",
        "-r",
        required=True,
        help="Path to the current evaluation report JSON",
    )
    p_gate.add_argument(
        "--config",
        "-c",
        default=None,
        help="Optional GateConfig JSON (per-metric thresholds)",
    )
    p_gate.add_argument(
        "--strict",
        action="store_true",
        help="Use the built-in strict threshold preset (overridden by --config)",
    )
    p_gate.add_argument(
        "--update-baseline",
        action="store_true",
        help="Replace the baseline with the current report and exit 0",
    )
    p_gate.add_argument(
        "--markdown",
        default=None,
        help="Write a Markdown summary (suitable for PR comments) to this path",
    )
    p_gate.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write the JSON gate result to this path",
    )
    p_gate.add_argument("--json", action="store_true", help="Print JSON instead of text to stdout")
    p_gate.set_defaults(func=cmd_quality_gate)

    # ab-compare
    p_ab = sub.add_parser(
        "ab-compare",
        help="Compare two retrieval systems from per-query metric JSON files",
    )
    p_ab.add_argument("--a", required=True, help="Baseline metric JSON: metric -> values")
    p_ab.add_argument("--b", required=True, help="Candidate metric JSON: metric -> values")
    p_ab.add_argument("--name-a", default="A", help="Display name for system A")
    p_ab.add_argument("--name-b", default="B", help="Display name for system B")
    p_ab.add_argument("--confidence", type=float, default=0.95, help="Bootstrap CI confidence")
    p_ab.add_argument("--n-resamples", type=int, default=2000, help="Bootstrap resamples")
    p_ab.add_argument("--seed", type=int, default=0, help="Random seed")
    p_ab.add_argument("--alpha", type=float, default=0.05, help="Winner significance level")
    p_ab.add_argument("--output", "-o", default=None, help="Write full JSON report to this path")
    p_ab.add_argument("--markdown", default=None, help="Write Markdown report to this path")
    p_ab.add_argument("--markdown-stdout", action="store_true", help="Print Markdown to stdout")
    p_ab.add_argument("--json", action="store_true", help="Print JSON instead of text to stdout")
    p_ab.set_defaults(func=cmd_ab_compare)

    # calibrate
    p_cal = sub.add_parser(
        "calibrate",
        help="Summarise score calibration from labelled retrieval examples",
    )
    p_cal.add_argument("--input", "-i", required=True, help="JSON with scores/labels")
    p_cal.add_argument("--bins", type=int, default=10, help="Number of reliability bins")
    p_cal.add_argument("--output", "-o", default=None, help="Write JSON report to this path")
    p_cal.add_argument("--json", action="store_true", help="Print JSON instead of text to stdout")
    p_cal.set_defaults(func=cmd_calibrate)

    # drift-report
    p_drift = sub.add_parser(
        "drift-report",
        help="Compare baseline/current score or feature distributions with PSI",
    )
    p_drift.add_argument("--baseline", required=True, help="Baseline numeric vector JSON")
    p_drift.add_argument("--current", required=True, help="Current numeric vector JSON")
    p_drift.add_argument("--bins", type=int, default=10, help="Number of PSI bins")
    p_drift.add_argument("--output", "-o", default=None, help="Write JSON report to this path")
    p_drift.add_argument("--json", action="store_true", help="Print JSON instead of text to stdout")
    p_drift.set_defaults(func=cmd_drift_report)

    # version
    p_version = sub.add_parser("version", help="Print the engine version")
    p_version.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
