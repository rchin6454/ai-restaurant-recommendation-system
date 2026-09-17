"""Groq client wrapper, per-model request profiles, and a hard call cap (plan task 3.1).

The SDK already retries connection errors, 408, 409, 429 and 5xx twice with backoff (L-04, L-06),
so nothing in this package retries on top of it (L-07).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from functools import lru_cache

import groq
import httpx

from src.config import Settings, get_logger
from src.core.models import RankingTrace

logger = get_logger(__name__)


class LLMUnavailable(RuntimeError):
    """Any reason the LLM ranker can't give a usable answer. The orchestrator falls back on it (§5.6).

    `trace` carries token usage when the failure happened after a billed call.
    """

    def __init__(self, reason: str, *, trace: RankingTrace | None = None) -> None:
        super().__init__(reason)
        self.trace = trace


@dataclass(frozen=True)
class ModelProfile:
    """What differs between Groq models. Everything else about the call is shared."""

    strict_schema: bool  # constrained decoding: output always matches the JSON schema
    reasoning: dict[str, object] = field(default_factory=dict)
    max_completion_tokens: int = 8000  # reasoning tokens count against this — leave headroom (L-10)
    input_usd_per_m: float | None = None
    cached_input_usd_per_m: float | None = None
    output_usd_per_m: float | None = None


MODEL_PROFILES: dict[str, ModelProfile] = {
    # Strict json_schema, automatic prompt caching (cached input at 50%).
    "openai/gpt-oss-120b": ModelProfile(
        strict_schema=True,
        reasoning={"reasoning_effort": "medium", "include_reasoning": False},
        input_usd_per_m=0.15,
        cached_input_usd_per_m=0.075,
        output_usd_per_m=0.60,
    ),
    # Best-effort json_schema only; `reasoning_format` must be hidden/parsed when JSON output is on.
    "qwen/qwen3.6-27b": ModelProfile(
        strict_schema=False,
        reasoning={"reasoning_effort": "default", "reasoning_format": "hidden"},
        input_usd_per_m=0.60,
        cached_input_usd_per_m=0.60,  # no published cached-input discount
        output_usd_per_m=3.00,
    ),
}
_UNKNOWN_MODEL = ModelProfile(strict_schema=False, max_completion_tokens=4000)


def profile_for(model: str) -> ModelProfile:
    profile = MODEL_PROFILES.get(model)
    if profile is None:
        logger.warning("no profile for model; using best-effort JSON, no reasoning params", extra={"model": model})
        return _UNKNOWN_MODEL
    return profile


def estimate_cost_usd(profile: ModelProfile, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float | None:
    if profile.input_usd_per_m is None or profile.output_usd_per_m is None:
        return None
    cached_rate = profile.cached_input_usd_per_m if profile.cached_input_usd_per_m is not None else profile.input_usd_per_m
    usd = (
        (prompt_tokens - cached_tokens) * profile.input_usd_per_m
        + cached_tokens * cached_rate
        + completion_tokens * profile.output_usd_per_m
    ) / 1_000_000
    return round(usd, 6)


class GroqClient(groq.Groq):
    """`groq.Groq` that doesn't retry 429s.

    The SDK answers a 429 by sleeping out Groq's reset and retrying, which at 8K tokens/min meant
    13-38 s stalls. `src.llm.rate_limit` already paces calls, so a 429 fails fast, pauses further
    calls for `retry-after`, and the request degrades. Other retries (408, 409, 5xx, connection) stay.
    """

    def _should_retry(self, response: httpx.Response) -> bool:
        return response.status_code != 429 and super()._should_retry(response)


@lru_cache(maxsize=4)
def _client(api_key: str, timeout_s: float) -> groq.Groq:
    # Built once per key, not per request. `max_retries` stays at the SDK default of 2 (L-07).
    return GroqClient(api_key=api_key, timeout=timeout_s)


def get_client(config: Settings) -> groq.Groq:
    if config.groq_api_key is None:  # L-01, L-02: an empty env value is already None
        raise LLMUnavailable("GROQ_API_KEY is not set")
    return _client(config.groq_api_key.get_secret_value(), config.llm_timeout_s)


class CallBudget:
    """Hard cap on LLM calls in this process (L-24). `None` = unlimited, which is what a server wants."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cap: int | None = None
        self.used = 0

    def set_cap(self, cap: int | None) -> None:
        with self._lock:
            self.cap, self.used = cap, 0

    def spend(self) -> None:
        with self._lock:
            if self.cap is not None and self.used >= self.cap:
                raise LLMUnavailable(f"LLM call cap of {self.cap} reached for this run")
            self.used += 1


call_budget = CallBudget()
