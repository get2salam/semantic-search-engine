"""Regenerate the committed retrieval-quality baseline.

This is a thin wrapper around the public CLI:

    python -m examples.quality_gate.regenerate_baseline

Equivalent to::

    python cli.py evaluate \
        --dataset examples/quality_gate/dataset \
        --k 1,3,5,10 \
        --output examples/quality_gate/baseline.json \
        --quiet --json > /dev/null

The baseline JSON is committed to the repo so the gate can run in CI
without re-encoding the corpus on every build.  Re-run this script
whenever you intentionally upgrade the model or expand the dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from cli import main  # noqa: E402  (import after sys.path mutation)


def regenerate() -> int:
    dataset = REPO_ROOT / "examples" / "quality_gate" / "dataset"
    output = REPO_ROOT / "examples" / "quality_gate" / "baseline.json"
    return main(
        [
            "evaluate",
            "--dataset",
            str(dataset),
            "--k",
            "1,3,5,10",
            "--output",
            str(output),
            "--quiet",
            "--json",
        ]
    )


if __name__ == "__main__":  # pragma: no cover - thin wrapper
    sys.exit(regenerate())
