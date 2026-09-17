"""Client-side pacing for Groq's per-account rate limits.

Groq caps requests and tokens per minute and per day, per model. For `openai/gpt-oss-120b` on this
account that's 30 requests/min, 1,000/day, 8,000 tokens/min and 200,000 tokens/day. One ranking call
is ~6K tokens (3.9K prompt + up to 2.4K completion, measured 2026-09-15), so tokens are what bind:
about one call a minute and ~30 a day. Finding that out from a 429 is the expensive way — the call is
rejected, and with the SDK's retries a user waits 13-38 s for nothing.

Before each call the ranker reserves an estimate here. If it fits every window the call goes ahead;
if capacity frees within `llm_rate_limit_max_wait_s` it waits; otherwise `RateLimitExceeded` is
raised and the orchestrator answers with the deterministic ranker. After the call, the reservation
is corrected to the tokens Groq actually reported.

State is per process. Another process on the same key (the CLI while the API runs, or a second
uvicorn worker) isn't visible here; its usage surfaces as a Groq 429, which pauses calls for the
`retry-after` period via `block_for`.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from src.config import Settings, get_logger

logger = get_logger(__name__)

MINUTE = 60.0
DAY = 86_400.0
_EDGE_S = 0.05  # sleep just past a window edge so the oldest call has really left it


class RateLimitExceeded(Exception):
    """The call can't be made within the allowed wait. `retry_after_s` may be `math.inf`."""

    def __init__(self, reason: str, retry_after_s: float) -> None:
        super().__init__(reason)
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class RateLimits:
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int
    tokens_per_day: int

    @classmethod
    def from_settings(cls, config: Settings) -> RateLimits:
        return cls(
            requests_per_minute=config.llm_requests_per_minute,
            requests_per_day=config.llm_requests_per_day,
            tokens_per_minute=config.llm_tokens_per_minute,
            tokens_per_day=config.llm_tokens_per_day,
        )


@dataclass
class _Call:
    at: float
    tokens: int


class Reservation:
    """A granted call slot. `settle()` replaces the estimate with the tokens actually used."""

    def __init__(self, limiter: RateLimiter, call: _Call, waited_s: float = 0.0) -> None:
        self._limiter, self._call = limiter, call
        self.waited_s = waited_s  # time spent queued for capacity before the call was granted

    @property
    def tokens(self) -> int:
        return self._call.tokens

    def settle(self, tokens: int) -> None:
        with self._limiter._lock:
            self._call.tokens = max(int(tokens), 0)


class RateLimiter:
    """Sliding one-minute and one-day windows over requests and tokens. Thread-safe."""

    def __init__(
        self,
        limits: RateLimits,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.limits = limits
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._calls: deque[_Call] = deque()  # oldest first, at most one day old
        self._blocked_until = 0.0
        self._blocked_reason = ""

    def acquire(self, tokens: int, *, max_wait_s: float = 0.0) -> Reservation:
        deadline = self._clock() + max_wait_s
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                wait, reason = self._wait_needed(tokens, now)
                if wait <= 0:
                    call = _Call(now, tokens)
                    self._calls.append(call)
                    return Reservation(self, call, waited)
            if now + wait > deadline:
                message = f"Groq rate limit: {reason}" + ("" if math.isinf(wait) else f"; capacity frees in ~{math.ceil(wait)} s")
                logger.warning("llm call skipped by the rate limiter", extra={"reason": message})
                raise RateLimitExceeded(message, wait)
            logger.info("llm rate limiter waiting", extra={"wait_s": round(wait, 1), "reason": reason})
            self._sleep(wait + _EDGE_S)  # then re-check: another thread may have taken the capacity
            waited += wait + _EDGE_S

    def block_for(self, seconds: float, reason: str) -> None:
        """Refuse every call for `seconds`, e.g. after Groq answers 429."""
        with self._lock:
            until = self._clock() + seconds
            if until > self._blocked_until:
                self._blocked_until, self._blocked_reason = until, reason

    def usage(self) -> dict[str, int]:
        with self._lock:
            now = self._clock()
            self._prune(now)
            minute = [c for c in self._calls if c.at > now - MINUTE]
            return {
                "requests_last_minute": len(minute),
                "tokens_last_minute": sum(c.tokens for c in minute),
                "requests_last_day": len(self._calls),
                "tokens_last_day": sum(c.tokens for c in self._calls),
            }

    def _prune(self, now: float) -> None:
        while self._calls and self._calls[0].at <= now - DAY:
            self._calls.popleft()

    def _wait_needed(self, tokens: int, now: float) -> tuple[float, str]:
        """Seconds until a call of `tokens` fits every limit (0 = now, inf = never), and the binding limit."""
        worst, reason = 0.0, ""
        if self._blocked_until > now:
            worst, reason = self._blocked_until - now, self._blocked_reason
        lim = self.limits
        for span, unit, max_requests, max_tokens in (
            (MINUTE, "minute", lim.requests_per_minute, lim.tokens_per_minute),
            (DAY, "day", lim.requests_per_day, lim.tokens_per_day),
        ):
            if tokens > max_tokens:
                return math.inf, f"one call needs ~{tokens:,} tokens, more than the {max_tokens:,} tokens per {unit} limit"
            window = [c for c in self._calls if c.at > now - span]

            excess = len(window) + 1 - max_requests
            if excess > 0:  # wait for the oldest `excess` calls to leave the window
                wait = window[excess - 1].at + span - now
                if wait > worst:
                    worst, reason = wait, f"{max_requests:,} requests per {unit} reached"

            used, expired = sum(c.tokens for c in window), 0
            while used + tokens > max_tokens:  # terminates: tokens <= max_tokens
                used -= window[expired].tokens
                expired += 1
            if expired:
                wait = window[expired - 1].at + span - now
                if wait > worst:
                    worst, reason = wait, f"{max_tokens:,} tokens per {unit} would be exceeded"
        return worst, reason


# ---------------------------------------------------------------------------
# Process-wide limiters
# ---------------------------------------------------------------------------

_registry: dict[tuple[str, RateLimits], RateLimiter] = {}
_registry_lock = threading.Lock()


def _sleep(seconds: float) -> None:  # indirection so tests can forbid real sleeps
    time.sleep(seconds)


def limiter_for(config: Settings) -> RateLimiter:
    """One limiter per model and limit set for the life of the process (Groq limits are per model)."""
    key = (config.model, RateLimits.from_settings(config))
    with _registry_lock:
        if key not in _registry:
            _registry[key] = RateLimiter(key[1], sleep=_sleep)
        return _registry[key]


def reset_limiters() -> None:
    with _registry_lock:
        _registry.clear()
