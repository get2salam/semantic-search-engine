"""
Unit tests for structured JSON logging and request-id propagation.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging

import pytest

from logging_config import (
    JsonFormatter,
    configure_logging,
    get_request_id,
    new_request_id,
    reset_request_id,
    set_request_id,
)


def _emit(msg: str, **extra):
    """Emit a log record through JsonFormatter and return the parsed dict."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("sse.test")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.info(msg, extra=extra)
    return json.loads(stream.getvalue().strip())


class TestJsonFormatter:
    def test_emits_valid_json(self):
        payload = _emit("hello world")
        assert payload["msg"] == "hello world"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "sse.test"

    def test_includes_iso_timestamp(self):
        payload = _emit("ts check")
        # Must end with UTC offset (+00:00 or Z)
        assert payload["ts"].endswith("+00:00")

    def test_propagates_extra_fields(self):
        payload = _emit("with extras", user_id=42, path="/search")
        assert payload["user_id"] == 42
        assert payload["path"] == "/search"

    def test_non_serialisable_extras_are_reprd(self):
        class Weird:
            def __repr__(self):
                return "<weird obj>"

        payload = _emit("weirdness", obj=Weird())
        assert payload["obj"] == "<weird obj>"

    def test_exc_info_is_captured(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("sse.exc")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("explosion")

        payload = json.loads(stream.getvalue().strip())
        assert payload["msg"] == "explosion"
        assert "exc_info" in payload
        assert "RuntimeError" in payload["exc_info"]


class TestRequestIdContext:
    def test_default_is_none(self):
        # Reset any state from another test
        assert get_request_id() is None or isinstance(get_request_id(), str)

    def test_set_and_reset(self):
        token = set_request_id("abc123")
        try:
            assert get_request_id() == "abc123"
        finally:
            reset_request_id(token)
        assert get_request_id() is None

    def test_new_request_id_is_unique(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_request_id_flows_into_log_payload(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger("sse.reqid")
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        token = set_request_id("req-xyz")
        try:
            logger.info("inside context")
        finally:
            reset_request_id(token)
        logger.info("outside context")

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        inside, outside = json.loads(lines[0]), json.loads(lines[1])
        assert inside["request_id"] == "req-xyz"
        assert "request_id" not in outside

    def test_request_id_survives_across_async_await(self):
        # The whole point of contextvars: each task sees its own binding,
        # but awaits inside the same task preserve the binding.
        async def _work():
            token = set_request_id("async-id")
            try:
                await asyncio.sleep(0)
                return get_request_id()
            finally:
                reset_request_id(token)

        result = asyncio.run(_work())
        assert result == "async-id"


class TestConfigureLogging:
    def test_configure_idempotent(self):
        configure_logging(level="INFO", json_format=True)
        handlers_a = list(logging.getLogger().handlers)
        configure_logging(level="INFO", json_format=True)
        handlers_b = list(logging.getLogger().handlers)
        # Should replace, not accumulate
        assert len(handlers_a) == len(handlers_b) == 1

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING"])
    def test_respects_level(self, level):
        configure_logging(level=level, json_format=False)
        assert logging.getLogger().level == getattr(logging, level)
