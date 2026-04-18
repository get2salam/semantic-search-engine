"""
Structured logging
==================
Configures the root logger to emit one JSON object per line. Each log
record includes:

- ``ts``: ISO-8601 UTC timestamp
- ``level``: log level name
- ``logger``: logger name
- ``msg``: the formatted message
- ``request_id``: correlation ID (when set via ``set_request_id()``)
- plus any extra fields passed via ``extra={"key": ...}``

Request IDs are propagated across asyncio tasks via ``contextvars``,
so a single request's logs can be collated end-to-end regardless of
which coroutine emitted them.
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import sys
import uuid

# Standard LogRecord attributes we don't want to spill into the JSON body.
_LOG_RECORD_BUILTIN_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "sse_request_id", default=None
)


def set_request_id(request_id: str | None) -> contextvars.Token:
    """Bind a request ID to the current async context."""
    return _request_id.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Undo a prior ``set_request_id`` call."""
    _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the request ID bound to the current context, if any."""
    return _request_id.get()


def new_request_id() -> str:
    """Generate a fresh request ID (UUIDv4, no dashes for brevity)."""
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """Format each LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": _dt.datetime.fromtimestamp(record.created, tz=_dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        req_id = get_request_id()
        if req_id:
            payload["request_id"] = req_id

        # Propagate any caller-supplied `extra=` fields
        for key, value in record.__dict__.items():
            if key in _LOG_RECORD_BUILTIN_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)  # must be serialisable
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO", json_format: bool = True) -> None:
    """
    Reconfigure the root logger.

    Args:
        level: logging level name (``"INFO"``, ``"DEBUG"``, ...)
        json_format: if True, emit one JSON object per line. If False,
            fall back to a human-readable single-line format.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove any pre-existing handlers so this function is idempotent
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)
