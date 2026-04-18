"""
Unit tests for the token-bucket rate limiter.
"""

from __future__ import annotations

import time

import pytest

from rate_limit import TokenBucketRateLimiter


class TestBasicBehaviour:
    def test_allows_up_to_capacity(self):
        limiter = TokenBucketRateLimiter(capacity=3, refill_rate_per_sec=1.0)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True
        assert limiter.allow("a") is True

    def test_rejects_when_bucket_empty(self):
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate_per_sec=0.01)
        assert limiter.allow("x") is True
        assert limiter.allow("x") is True
        assert limiter.allow("x") is False

    def test_independent_buckets_per_key(self):
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=0.01)
        assert limiter.allow("a") is True
        assert limiter.allow("a") is False
        # Different key gets its own bucket
        assert limiter.allow("b") is True

    def test_refills_over_time(self):
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=100.0)
        assert limiter.allow("k") is True
        assert limiter.allow("k") is False
        time.sleep(0.05)  # >= 1/100s of refill
        assert limiter.allow("k") is True

    def test_retry_after_when_allowed_is_zero(self):
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_sec=1.0)
        limiter.allow("k")
        assert limiter.retry_after_seconds("k") == 0.0

    def test_retry_after_when_empty_is_positive(self):
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=2.0)
        limiter.allow("k")  # drain
        wait = limiter.retry_after_seconds("k")
        assert 0 < wait <= 0.5 + 1e-6


class TestValidation:
    def test_zero_capacity_rejected(self):
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(capacity=0, refill_rate_per_sec=1.0)

    def test_zero_refill_rate_rejected(self):
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=0.0)


class TestResetAndSnapshot:
    def test_reset_single_key(self):
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=0.01)
        limiter.allow("k")
        assert limiter.allow("k") is False
        limiter.reset("k")
        assert limiter.allow("k") is True

    def test_reset_all(self):
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate_per_sec=0.01)
        limiter.allow("a")
        limiter.allow("b")
        limiter.reset()
        assert limiter.allow("a") is True
        assert limiter.allow("b") is True

    def test_snapshot_reports_tokens(self):
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate_per_sec=0.01)
        limiter.allow("z")
        snap = limiter.snapshot()
        assert "z" in snap
        assert snap["z"] <= 4.0


class TestConvenienceConstructor:
    def test_per_minute_capacity(self):
        limiter = TokenBucketRateLimiter.per_minute(60)
        # Burst of 60 allowed, 61st denied
        for _ in range(60):
            assert limiter.allow("k") is True
        assert limiter.allow("k") is False
