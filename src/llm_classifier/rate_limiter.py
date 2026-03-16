"""Centralized, thread-safe rate limiter with global 429 cooldown.

Design:
- Token-bucket controls steady-state request rate (target RPM / 60).
- Semaphore caps in-flight concurrent requests.
- On any 429, a global cooldown-until timestamp is set so ALL workers
  pause — preventing retry storms.
- Retry-After header is respected when present.

Usage:
    limiter = APIRateLimiter(target_rpm=60, max_concurrency=4)
    with limiter.acquire():
        try:
            response = client.chat.completions.create(...)
        except openai.RateLimitError as exc:
            limiter.report_rate_limit(exc)
            raise
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiterStats:
    """Counters exposed for instrumentation / logging."""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limit_hits: int = 0
    retries: int = 0
    total_retry_delay_s: float = 0.0
    total_limiter_wait_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def record_request(self, *, success: bool):
        with self._lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1

    def record_rate_limit(self):
        with self._lock:
            self.rate_limit_hits += 1

    def record_retry(self, delay_s: float):
        with self._lock:
            self.retries += 1
            self.total_retry_delay_s += delay_s

    def record_limiter_wait(self, wait_s: float):
        with self._lock:
            self.total_limiter_wait_s += wait_s

    def record_cache(self, *, hit: bool):
        with self._lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def to_dict(self) -> dict:
        with self._lock:
            total = self.total_requests
            successful = self.successful_requests
            rate_limits = self.rate_limit_hits
            retries = self.retries
            retry_delay = self.total_retry_delay_s
            limiter_wait = self.total_limiter_wait_s
            cache_hits = self.cache_hits
            cache_misses = self.cache_misses

        cache_total = cache_hits + cache_misses
        return {
            "total_requests": total,
            "successful_requests": successful,
            "rate_limit_429s": rate_limits,
            "retries": retries,
            "avg_retry_delay_s": round(retry_delay / max(retries, 1), 2),
            "total_limiter_wait_s": round(limiter_wait, 2),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": round(cache_hits / max(cache_total, 1), 3),
            "effective_api_calls": cache_misses,
        }


class APIRateLimiter:
    """Thread-safe rate limiter with global 429 cooldown.

    Args:
        target_rpm: Target requests per minute. A 10% safety margin is
            applied internally (actual rate = target_rpm * 0.9).
        max_concurrency: Maximum in-flight concurrent requests.
        cooldown_on_429: Default cooldown seconds when a 429 has no
            Retry-After header.
    """

    def __init__(
        self,
        target_rpm: float = 60,
        max_concurrency: int = 4,
        cooldown_on_429: float = 15.0,
    ):
        effective_rpm = target_rpm * 0.9  # 10% safety margin
        self._min_interval = 60.0 / max(effective_rpm, 1)
        self._max_concurrency = max(max_concurrency, 1)
        self._default_cooldown = cooldown_on_429

        self._lock = threading.Lock()
        self._last_dispatch = 0.0
        self._cooldown_until = 0.0
        self._semaphore = threading.Semaphore(self._max_concurrency)

        self.stats = RateLimiterStats()

    # -- public API --------------------------------------------------------

    def acquire(self):
        """Return a context-manager that gates one API request.

        Blocks until:
          1. Any global cooldown has elapsed.
          2. The token-bucket interval has passed.
          3. A concurrency slot is available.
        """
        return _AcquireContext(self)

    def report_rate_limit(self, exc: Exception | None = None):
        """Called by any worker that receives a 429.

        Sets a global cooldown so ALL workers pause.
        """
        retry_after = self._parse_retry_after(exc)
        cooldown = retry_after if retry_after is not None else self._default_cooldown
        # Add jitter to prevent thundering-herd on resume
        cooldown += random.uniform(0, min(cooldown * 0.25, 5.0))

        with self._lock:
            new_until = time.monotonic() + cooldown
            # Only extend, never shorten an existing cooldown
            if new_until > self._cooldown_until:
                self._cooldown_until = new_until
        self.stats.record_rate_limit()

    def compute_retry_delay(self, attempt: int, exc: Exception | None = None) -> float:
        """Exponential backoff with jitter, respecting Retry-After."""
        retry_after = self._parse_retry_after(exc)
        if retry_after is not None:
            return retry_after + random.uniform(0, 2)
        base = min(2 ** attempt * 5, 60)
        return base + random.uniform(0, min(base * 0.5, 5))

    # -- internals ---------------------------------------------------------

    def _wait_for_slot(self):
        """Block until rate-limit window + cooldown have passed, then claim a semaphore slot."""
        t0 = time.monotonic()

        # 1. Wait for global cooldown
        with self._lock:
            remaining = self._cooldown_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

        # 2. Token-bucket: space out requests
        with self._lock:
            now = time.monotonic()
            wait = self._last_dispatch + self._min_interval - now
            if wait > 0:
                time.sleep(wait)
            self._last_dispatch = time.monotonic()

        # 3. Concurrency gate
        self._semaphore.acquire()

        waited = time.monotonic() - t0
        if waited > 0.01:
            self.stats.record_limiter_wait(waited)

    def _release_slot(self):
        self._semaphore.release()

    @staticmethod
    def _parse_retry_after(exc: Exception | None) -> float | None:
        """Extract Retry-After seconds from an OpenAI RateLimitError."""
        if exc is None:
            return None
        # openai.RateLimitError stores headers on the response
        response = getattr(exc, "response", None)
        if response is None:
            return None
        headers = getattr(response, "headers", None) or {}
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return float(retry_after)
            except (TypeError, ValueError):
                pass
        return None


class _AcquireContext:
    """Context manager returned by APIRateLimiter.acquire()."""

    def __init__(self, limiter: APIRateLimiter):
        self._limiter = limiter

    def __enter__(self):
        self._limiter._wait_for_slot()
        return self._limiter

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._limiter._release_slot()
        return False  # do not suppress exceptions
