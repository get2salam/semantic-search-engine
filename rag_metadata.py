"""Metadata filtering helpers for RAG retrieval pipelines."""

from collections.abc import Mapping, Sequence
from typing import Any


def get_nested(metadata: Mapping[str, Any], dotted_path: str, default: Any = None) -> Any:
    """Read a dotted key from nested metadata dictionaries."""

    current: Any = metadata
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        return [value]
    return list(value)


def _match_rule(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        if "any" in expected:
            return actual in set(_as_list(expected["any"]))
        if "all" in expected:
            actual_values = set(_as_list(actual))
            return set(_as_list(expected["all"])).issubset(actual_values)
        if "contains" in expected:
            return expected["contains"] in _as_list(actual)
        if "min" in expected and actual < expected["min"]:
            return False
        return not ("max" in expected and actual > expected["max"])
    if isinstance(expected, set | list | tuple):
        return actual in expected
    return actual == expected


def metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Return whether metadata satisfies all provided filter rules."""

    if not filters:
        return True
    for key, expected in filters.items():
        actual = get_nested(metadata, key)
        if actual is None or not _match_rule(actual, expected):
            return False
    return True


def filter_records(
    records: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any] | None,
    *,
    metadata_key: str = "metadata",
) -> list[Mapping[str, Any]]:
    """Filter retriever records by nested metadata rules."""

    return [record for record in records if metadata_matches(record.get(metadata_key, {}), filters)]
