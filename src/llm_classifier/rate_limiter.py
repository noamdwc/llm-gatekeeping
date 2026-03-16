"""Centralized, thread-safe rate limiter with adaptive throttling.

Design:
- Token-bucket controls steady-state request rate (target RPM / 60).
- Semaphore caps in-flight concurrent requests.
- On any 429, a global cooldown-until timestamp is set so ALL workers
  pause — preventing retry storms.
- Cooldown escalates on repeated 429s: each 429 within a window
  doubles the cooldown duration, up to a cap.
- After a cooldown with no 429s, the rate gradually recovers.
- Retry-After header is respected when present.

Usage:
    limiter = APIRateLimiter(target_rpm=30, max_concurrency=1)
    with limiter.acquire():
        try:
            response = client.chat.completions.create(...)
        except openai.RateLimitError as exc:
            limiter.report_rate_limit(exc)
            raise
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    """Thread-safe rate limiter with adaptive throttling on 429s.

    Args:
        target_rpm: Target requests per minute. A 10% safety margin is
            applied internally (actual rate = target_rpm * 0.9).
        max_concurrency: Maximum in-flight concurrent requests.
        cooldown_on_429: Base cooldown seconds when a 429 has no
            Retry-After header. Escalates on repeated 429s.
    """

    def __init__(
        self,
        target_rpm: float = 30,
        max_concurrency: int = 1,
        cooldown_on_429: float = 15.0,
    ):
        self._target_rpm = target_rpm
        self._base_cooldown = cooldown_on_429
        self._max_concurrency = max(max_concurrency, 1)

        # Adaptive state
        effective_rpm = target_rpm * 0.9  # 10% safety margin
        self._min_interval = 60.0 / max(effective_rpm, 1)
        self._initial_min_interval = self._min_interval
        self._consecutive_429s = 0
        self._last_429_time = 0.0
        # Window: if no 429 for this many seconds, reset escalation
        self._escalation_reset_window = 120.0
        self._max_cooldown = 120.0  # cap escalated cooldown

        self._lock = threading.Lock()
        self._last_dispatch = 0.0
        self._cooldown_until = 0.0
        self._semaphore = threading.Semaphore(self._max_concurrency)

        self.stats = RateLimiterStats()

    @property
    def effective_rpm(self) -> float:
        """Current effective requests per minute after any throttling."""
        with self._lock:
            return 60.0 / self._min_interval

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

        Sets a global cooldown so ALL workers pause. Escalates on
        repeated 429s and slows down the token-bucket rate.
        """
        retry_after = self._parse_retry_after(exc)

        with self._lock:
            now = time.monotonic()

            # Track consecutive 429s (reset if it's been quiet)
            if now - self._last_429_time > self._escalation_reset_window:
                self._consecutive_429s = 0
            self._consecutive_429s += 1
            self._last_429_time = now

            # Escalating cooldown: base * 2^(consecutive-1), capped
            if retry_after is not None:
                cooldown = retry_after
            else:
                cooldown = min(
                    self._base_cooldown * (2 ** (self._consecutive_429s - 1)),
                    self._max_cooldown,
                )

            # Add jitter (10-25% of cooldown)
            cooldown += random.uniform(cooldown * 0.1, cooldown * 0.25)

            # Set global cooldown (only extend, never shorten)
            new_until = now + cooldown
            if new_until > self._cooldown_until:
                self._cooldown_until = new_until

            # Adaptive slowdown: halve the effective RPM (double the interval)
            # Floor at 6 RPM (10s between requests)
            new_interval = min(self._min_interval * 1.5, 10.0)
            if new_interval > self._min_interval:
                self._min_interval = new_interval
                effective = 60.0 / self._min_interval
                logger.info(f"Rate throttled to {effective:.1f} RPM after {self._consecutive_429s} consecutive 429s")

        self.stats.record_rate_limit()

    def report_success(self):
        """Called after a successful API request.

        Gradually recovers the rate toward the original target.
        """
        with self._lock:
            now = time.monotonic()
            # Only recover if no recent 429 (at least 30s since last)
            if now - self._last_429_time < 30.0:
                return
            # Recover 10% toward the original rate
            if self._min_interval > self._initial_min_interval * 1.01:
                self._min_interval = max(
                    self._initial_min_interval,
                    self._min_interval * 0.9,
                )
                # Reset consecutive counter on sustained success
                self._consecutive_429s = 0

    def compute_retry_delay(self, attempt: int, exc: Exception | None = None) -> float:
        """Exponential backoff with jitter, respecting Retry-After."""
        retry_after = self._parse_retry_after(exc)
        if retry_after is not None:
            return retry_after + random.uniform(0, 2)
        base = min(2 ** attempt * 8, 90)
        return base + random.uniform(0, min(base * 0.5, 10))

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
            # Re-check cooldown (might have been extended while we slept)
            cooldown_remaining = self._cooldown_until - now
            if cooldown_remaining > 0:
                time.sleep(cooldown_remaining)
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
