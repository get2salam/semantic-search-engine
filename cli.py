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
