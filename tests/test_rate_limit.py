"""Groq rate-limit pacing (src/llm/rate_limit.py), driven by a fake clock — no real waiting."""

from __future__ import annotations

import math
import threading

import pytest

from src.config import load_settings
from src.llm.rate_limit import DAY, MINUTE, RateLimiter, RateLimitExceeded, RateLimits, limiter_for

ACCOUNT = RateLimits(requests_per_minute=30, requests_per_day=1000, tokens_per_minute=8000, tokens_per_day=200_000)
CALL = 6300  # one measured gpt-oss-120b ranking call


class FakeClock:
    def __init__(self) -> None:
        self.now = 10_000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def limiter(clock: FakeClock, limits: RateLimits = ACCOUNT) -> RateLimiter:
    return RateLimiter(limits, clock=clock, sleep=clock.sleep)


def test_tokens_per_minute_allow_one_ranking_call_a_minute(clock):
    rl = limiter(clock)
    rl.acquire(CALL)
    with pytest.raises(RateLimitExceeded, match="8,000 tokens per minute") as exc:
        rl.acquire(CALL)
    assert exc.value.retry_after_s == pytest.approx(MINUTE)
    clock.now += MINUTE + 0.1
    rl.acquire(CALL)
    assert clock.slept == []


def test_short_waits_are_waited_out_instead_of_degrading(clock):
    rl = limiter(clock)
    rl.acquire(CALL)
    clock.now += 55
    reservation = rl.acquire(CALL, max_wait_s=10)
    assert sum(clock.slept) == pytest.approx(5, abs=0.1)
    assert reservation.waited_s == pytest.approx(sum(clock.slept))  # reported so M-20 can exclude queueing
    assert rl.usage()["requests_last_minute"] == 1  # the first call has left the window


def test_long_waits_degrade_immediately(clock):
    rl = limiter(clock)
    rl.acquire(CALL)
    clock.now += 20
    with pytest.raises(RateLimitExceeded, match="capacity frees in ~40 s"):
        rl.acquire(CALL, max_wait_s=10)
    assert clock.slept == []


def test_settling_to_real_usage_frees_the_overestimate(clock):
    rl = limiter(clock)
    reservation = rl.acquire(7000)
    reservation.settle(3000)
    rl.acquire(5000)
    assert rl.usage()["tokens_last_minute"] == 8000


def test_only_calls_that_must_expire_are_waited_for(clock):
    rl = limiter(clock)
    rl.acquire(2000)
    clock.now += 10
    rl.acquire(2000)
    clock.now += 10
    rl.acquire(2000)
    with pytest.raises(RateLimitExceeded) as exc:
        rl.acquire(4000)  # 6,000 used: only the first call (t=0) must leave the window
    assert exc.value.retry_after_s == pytest.approx(MINUTE - 20)


def test_requests_per_minute(clock):
    rl = limiter(clock)
    for _ in range(30):
        rl.acquire(10)
    with pytest.raises(RateLimitExceeded, match="30 requests per minute"):
        rl.acquire(10)


def test_requests_per_day(clock):
    rl = limiter(clock, RateLimits(requests_per_minute=30, requests_per_day=3, tokens_per_minute=8000, tokens_per_day=200_000))
    for _ in range(3):
        rl.acquire(10)
        clock.now += MINUTE
    with pytest.raises(RateLimitExceeded, match="3 requests per day") as exc:
        rl.acquire(10)
    assert exc.value.retry_after_s == pytest.approx(DAY - 3 * MINUTE)


def test_tokens_per_day_allow_about_thirty_ranking_calls(clock):
    rl = limiter(clock)
    made = 0
    while True:
        try:
            rl.acquire(CALL)
        except RateLimitExceeded as exc:
            assert "200,000 tokens per day" in str(exc)
            break
        made += 1
        clock.now += MINUTE + 1
    assert made == 200_000 // CALL == 31
    clock.now += DAY
    rl.acquire(CALL)


def test_a_call_bigger_than_a_limit_never_waits(clock):
    rl = limiter(clock)
    with pytest.raises(RateLimitExceeded, match="more than the 8,000 tokens per minute") as exc:
        rl.acquire(9000, max_wait_s=3600)
    assert math.isinf(exc.value.retry_after_s) and clock.slept == []


def test_block_for_pauses_every_call(clock):
    rl = limiter(clock)
    rl.block_for(30, "Groq answered 429 (rate limited)")
    with pytest.raises(RateLimitExceeded, match="429"):
        rl.acquire(10, max_wait_s=5)
    rl.acquire(10, max_wait_s=40)
    assert sum(clock.slept) == pytest.approx(30, abs=0.1)


def test_concurrent_requests_never_overspend():
    rl = RateLimiter(ACCOUNT)
    granted = []

    def worker():
        try:
            rl.acquire(1000)
            granted.append(1)
        except RateLimitExceeded:
            pass

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(granted) == 8


def test_one_limiter_per_model_and_limits():
    a = load_settings(_env_file=None, model="openai/gpt-oss-120b")
    assert limiter_for(a) is limiter_for(load_settings(_env_file=None, model="openai/gpt-oss-120b"))
    assert limiter_for(a) is not limiter_for(load_settings(_env_file=None, model="qwen/qwen3.6-27b"))
    assert limiter_for(a) is not limiter_for(load_settings(_env_file=None, llm_tokens_per_minute=6000))
