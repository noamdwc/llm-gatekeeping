"""Tests for the centralized API rate limiter."""

import threading
import time

import pytest

from src.llm_classifier.rate_limiter import APIRateLimiter, RateLimiterStats


class TestRateLimiterStats:
    def test_initial_state(self):
        stats = RateLimiterStats()
        d = stats.to_dict()
        assert d["total_requests"] == 0
        assert d["rate_limit_429s"] == 0
        assert d["cache_hit_rate"] == 0

    def test_record_request(self):
        stats = RateLimiterStats()
        stats.record_request(success=True)
        stats.record_request(success=False)
        d = stats.to_dict()
        assert d["total_requests"] == 2
        assert d["successful_requests"] == 1

    def test_cache_hit_rate(self):
        stats = RateLimiterStats()
        stats.record_cache(hit=True)
        stats.record_cache(hit=True)
        stats.record_cache(hit=False)
        d = stats.to_dict()
        assert d["cache_hits"] == 2
        assert d["cache_misses"] == 1
        assert abs(d["cache_hit_rate"] - 0.667) < 0.01

    def test_thread_safety(self):
        stats = RateLimiterStats()
        n_threads = 10
        n_ops = 100

        def worker():
            for _ in range(n_ops):
                stats.record_request(success=True)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert stats.to_dict()["total_requests"] == n_threads * n_ops


class TestAPIRateLimiter:
    def test_token_bucket_spacing(self):
        """Requests should be spaced at least min_interval apart."""
        # 600 RPM = 10/s → ~100ms apart (after 10% margin → ~110ms)
        limiter = APIRateLimiter(target_rpm=600, max_concurrency=1)

        timestamps = []
        for _ in range(5):
            with limiter.acquire():
                timestamps.append(time.monotonic())

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i - 1]
            # Should be at least ~100ms (allowing some slack for test env)
            assert gap >= 0.08, f"Gap {gap:.3f}s too short between requests {i-1} and {i}"

    def test_concurrency_limit(self):
        """No more than max_concurrency requests should be in-flight."""
        limiter = APIRateLimiter(target_rpm=6000, max_concurrency=2)
        in_flight = [0]
        max_seen = [0]
        lock = threading.Lock()

        def worker():
            with limiter.acquire():
                with lock:
                    in_flight[0] += 1
                    max_seen[0] = max(max_seen[0], in_flight[0])
                time.sleep(0.05)
                with lock:
                    in_flight[0] -= 1

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_seen[0] <= 2, f"max in-flight {max_seen[0]} exceeded concurrency limit 2"

    def test_global_cooldown_on_429(self):
        """After report_rate_limit(), all workers should pause."""
        limiter = APIRateLimiter(target_rpm=6000, max_concurrency=4, cooldown_on_429=0.5)

        # Trigger a cooldown
        limiter.report_rate_limit()
        t0 = time.monotonic()

        with limiter.acquire():
            elapsed = time.monotonic() - t0

        # Should have waited at least ~0.5s (the cooldown + jitter)
        assert elapsed >= 0.4, f"Cooldown only waited {elapsed:.2f}s, expected >= 0.4s"

    def test_cooldown_escalates_on_repeated_429s(self):
        """Consecutive 429s should produce longer cooldowns."""
        limiter = APIRateLimiter(target_rpm=6000, max_concurrency=1, cooldown_on_429=1.0)

        # First 429
        limiter.report_rate_limit()
        with limiter._lock:
            cooldown_1 = limiter._cooldown_until - time.monotonic()

        # Immediately hit another 429
        limiter.report_rate_limit()
        with limiter._lock:
            cooldown_2 = limiter._cooldown_until - time.monotonic()

        # Second cooldown should be longer (base * 2^1 = 2s vs base * 2^0 = 1s)
        assert cooldown_2 > cooldown_1 * 1.3, (
            f"Second cooldown {cooldown_2:.2f}s should be notably longer than first {cooldown_1:.2f}s"
        )

    def test_adaptive_slowdown_on_429(self):
        """429s should slow down the token-bucket rate."""
        limiter = APIRateLimiter(target_rpm=600, max_concurrency=1)
        initial_interval = limiter._min_interval

        limiter.report_rate_limit()
        assert limiter._min_interval > initial_interval, (
            "Rate should slow down after 429"
        )

    def test_rate_recovers_after_success(self):
        """Sustained success should gradually recover the rate."""
        limiter = APIRateLimiter(target_rpm=600, max_concurrency=1)
        initial_interval = limiter._min_interval

        # Trigger a slowdown
        limiter.report_rate_limit()
        slowed_interval = limiter._min_interval
        assert slowed_interval > initial_interval

        # Simulate time passing (no recent 429)
        with limiter._lock:
            limiter._last_429_time = time.monotonic() - 60

        # Report success
        limiter.report_success()
        assert limiter._min_interval < slowed_interval, (
            "Rate should recover after successful requests with no recent 429"
        )

    def test_no_recovery_during_active_429s(self):
        """report_success should NOT speed up if 429 was recent."""
        limiter = APIRateLimiter(target_rpm=600, max_concurrency=1)

        limiter.report_rate_limit()
        slowed_interval = limiter._min_interval

        # Success immediately after 429 — should NOT recover
        limiter.report_success()
        assert limiter._min_interval == slowed_interval

    def test_escalation_resets_after_quiet_period(self):
        """Consecutive 429 counter should reset after the escalation window."""
        limiter = APIRateLimiter(target_rpm=6000, max_concurrency=1, cooldown_on_429=1.0)

        # Build up consecutive 429s
        limiter.report_rate_limit()
        limiter.report_rate_limit()
        assert limiter._consecutive_429s == 2

        # Simulate quiet period
        with limiter._lock:
            limiter._last_429_time = time.monotonic() - 200

        # Next 429 should reset the counter
        limiter.report_rate_limit()
        assert limiter._consecutive_429s == 1

    def test_retry_delay_exponential_backoff(self):
        limiter = APIRateLimiter(target_rpm=30)
        d0 = limiter.compute_retry_delay(0)
        d1 = limiter.compute_retry_delay(1)
        d2 = limiter.compute_retry_delay(2)
        # Base increases exponentially: 8, 16, 32 (plus jitter)
        assert d1 > d0 * 0.5  # account for jitter
        assert d2 > d1 * 0.5

    def test_retry_delay_respects_retry_after_header(self):
        """When exc has Retry-After header, use that instead of backoff."""

        class FakeResponse:
            headers = {"retry-after": "42"}

        class FakeExc(Exception):
            response = FakeResponse()

        limiter = APIRateLimiter(target_rpm=30)
        delay = limiter.compute_retry_delay(0, FakeExc())
        # Should be ~42 + small jitter
        assert 42 <= delay <= 45

    def test_stats_integration(self):
        limiter = APIRateLimiter(target_rpm=6000, max_concurrency=4)
        limiter.stats.record_request(success=True)
        limiter.stats.record_request(success=True)
        limiter.report_rate_limit()
        limiter.stats.record_retry(5.0)

        d = limiter.stats.to_dict()
        assert d["total_requests"] == 2
        assert d["rate_limit_429s"] == 1
        assert d["retries"] == 1
        assert d["avg_retry_delay_s"] == 5.0

    def test_effective_rpm_property(self):
        limiter = APIRateLimiter(target_rpm=60, max_concurrency=1)
        rpm = limiter.effective_rpm
        # 60 * 0.9 safety margin = 54
        assert 53 < rpm < 55
