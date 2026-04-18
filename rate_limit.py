"""
Token-bucket rate limiter
=========================
A lightweight, in-process rate limiter with no external dependencies.
Each client (identified by a key, typically their IP) gets an independent
token bucket that refills at a steady rate. When the bucket is empty, the
request is rejected with HTTP 429.

This implementation is deliberately simple: one dict lookup + a float
arithmetic per request. For production multi-instance deployments you
should swap this for a shared store (Redis, etc.), but for single-node
FastAPI apps it is more than sufficient and has zero runtime cost when
disabled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class _Bucket:
    """State for a single client's token bucket."""

    tokens: float
    last_refill: float
    lock: threading.Lock = field(default_factory=threading.Lock)


class TokenBucketRateLimiter:
    """
    Classic token-bucket rate limiter.

    Each client gets ``capacity`` tokens and earns a new token every
    ``1 / refill_rate`` seconds. A request costs 1 token. When the bucket
    is empty, ``allow()`` returns False and the caller can return 429.

    Thread-safe: a per-bucket lock protects the refill + consume step.

    Example:
        limiter = TokenBucketRateLimiter(capacity=60, refill_rate_per_sec=1.0)
        if not limiter.allow("192.168.1.1"):
            raise HTTPException(429, "Rate limit exceeded")
    """

    def __init__(self, capacity: int, refill_rate_per_sec: float):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate_per_sec <= 0:
            raise ValueError("refill_rate_per_sec must be > 0")

        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate_per_sec)
        self._buckets: dict[str, _Bucket] = {}
        self._global_lock = threading.Lock()

    @classmethod
    def per_minute(cls, requests_per_minute: int) -> TokenBucketRateLimiter:
        """Convenience constructor: N requests per 60s, bucket size = N."""
        return cls(capacity=requests_per_minute, refill_rate_per_sec=requests_per_minute / 60.0)

    def _get_bucket(self, key: str) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is not None:
            return bucket
        with self._global_lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=self.capacity, last_refill=time.monotonic())
                self._buckets[key] = bucket
            return bucket

    def allow(self, key: str, cost: float = 1.0) -> bool:
        """
        Attempt to consume ``cost`` tokens from ``key``'s bucket.

        Returns True if the request is allowed, False otherwise.
        """
        bucket = self._get_bucket(key)
        with bucket.lock:
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
            bucket.last_refill = now
            if bucket.tokens >= cost:
                bucket.tokens -= cost
                return True
            return False

    def retry_after_seconds(self, key: str, cost: float = 1.0) -> float:
        """
        How many seconds until the given key would next be allowed.

        Useful for populating the ``Retry-After`` header on 429 responses.
        """
        bucket = self._get_bucket(key)
        with bucket.lock:
            missing = max(0.0, cost - bucket.tokens)
            return missing / self.refill_rate if missing > 0 else 0.0

    def reset(self, key: str | None = None) -> None:
        """Reset a single client's bucket, or all buckets if key is None."""
        with self._global_lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)

    def snapshot(self) -> dict[str, float]:
        """Return a snapshot of current token counts (for observability)."""
        with self._global_lock:
            return {k: b.tokens for k, b in self._buckets.items()}
