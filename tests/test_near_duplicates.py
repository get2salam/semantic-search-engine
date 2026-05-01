"""Tests for the near_duplicates module."""

from __future__ import annotations

import json

import numpy as np
import pytest

from near_duplicates import (
    NearDuplicateGroup,
    NearDuplicateReport,
    near_duplicate_report,
)


def _normed(rows: list[list[float]]) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


def test_near_duplicate_report_empty_corpus():
    report = near_duplicate_report(np.zeros((0, 3)))
    assert isinstance(report, NearDuplicateReport)
    assert report.n_documents == 0
    assert report.n_groups == 0
    assert report.duplication_ratio == 0.0


def test_near_duplicate_report_no_duplicates_at_high_threshold():
    # Three orthogonal unit vectors → no pairwise cosine ≥ 0.5
    embs = _normed([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    report = near_duplicate_report(embs, threshold=0.5)
    assert report.n_groups == 0
    assert report.n_duplicate_documents == 0


def test_near_duplicate_report_clusters_paraphrases():
    # Two tight clusters + one isolated.
    embs = _normed(
        [
            [1.0, 0.05, 0.0],
            [1.0, 0.04, 0.0],  # near-dup of 0
            [1.0, 0.06, 0.0],  # near-dup of 0
            [0.0, 1.0, 0.05],
            [0.0, 1.0, 0.04],  # near-dup of 3
            [0.0, 0.0, 1.0],  # isolated
        ]
    )
    report = near_duplicate_report(embs, threshold=0.95)
    assert report.n_groups == 2
    assert report.n_duplicate_documents == 3  # 6 docs - 2 cluster representatives - 1 single
    sizes = sorted(g.size for g in report.groups)
    assert sizes == [2, 3]


def test_near_duplicate_report_chains_via_transitive_closure():
    # A — B — C chained by overlapping similarities; threshold-passing
    # adjacency should pull all three into one cluster.
    embs = _normed(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],
            [0.97, 0.24, 0.0],
        ]
    )
    report = near_duplicate_report(embs, threshold=0.97)
    assert report.n_groups == 1
    assert report.groups[0].size == 3
    assert report.groups[0].mean_similarity >= 0.97


def test_near_duplicate_report_picks_medoid_as_representative():
    # Doc 1 is between docs 0 and 2 — it should be the medoid.
    embs = _normed(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],
            [0.96, 0.28, 0.0],
        ]
    )
    report = near_duplicate_report(embs, threshold=0.95)
    assert report.n_groups == 1
    assert report.groups[0].representative == 1


def test_near_duplicate_report_renormalises_inputs():
    # Same vectors as the cluster test, but un-normalised. Should still
    # cluster identically because the function renormalises in place.
    raw = np.asarray(
        [
            [10.0, 0.5, 0.0],
            [10.0, 0.4, 0.0],
            [0.0, 10.0, 0.5],
            [0.0, 10.0, 0.4],
        ]
    )
    report = near_duplicate_report(raw, threshold=0.95)
    assert report.n_groups == 2


def test_near_duplicate_report_caps_groups_keeps_aggregates_honest():
    # 30 distinct duplicate pairs.
    rows: list[list[float]] = []
    for i in range(30):
        v = [0.0] * 30
        v[i] = 1.0
        rows.append(v)
        rows.append(v)
    embs = _normed(rows)
    report = near_duplicate_report(embs, threshold=0.95, max_groups=5)
    assert report.n_groups == 30
    assert len(report.groups) == 5
    assert report.n_duplicate_documents == 30  # one per pair


def test_near_duplicate_report_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        near_duplicate_report(np.zeros((2, 3)), threshold=1.5)


def test_near_duplicate_report_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        near_duplicate_report(np.zeros((5,)))


def test_near_duplicate_report_serialises():
    embs = _normed([[1.0, 0.05, 0.0], [1.0, 0.04, 0.0], [0.0, 0.0, 1.0]])
    payload = json.loads(json.dumps(near_duplicate_report(embs, threshold=0.95).to_dict()))
    assert payload["n_documents"] == 3
    assert payload["n_groups"] == 1
    assert isinstance(payload["groups"], list)
    g = payload["groups"][0]
    assert isinstance(g["indices"], list)
    assert g["size"] == 2


def test_near_duplicate_group_dataclass_round_trip():
    g = NearDuplicateGroup(representative=0, indices=[0, 1], mean_similarity=0.99, size=2)
    d = g.to_dict()
    assert d["representative"] == 0
    assert d["size"] == 2
