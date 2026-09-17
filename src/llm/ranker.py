"""Serialize candidates → call Groq → parse structured output (architecture §5.2-§5.4; plan 3.3-3.4, 3.7).

The model returns IDs and prose only. Checking those IDs against the candidate set is the
grounding gate's job (`src.core.recommender.validate_and_join`), not this module's.
`structured_completion` is the one guarded path to Groq; the eval judge (plan 5.9) uses it too.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import groq
import httpx
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.config import Settings, get_logger
from src.core.models import Preferences, RankingTrace, Relaxation
from src.data.cleaning import BUDGET_BANDS
from src.llm.client import LLMUnavailable, call_budget, estimate_cost_usd, get_client, profile_for
from src.llm.prompts import RANKING_SYSTEM_PROMPT
from src.llm.rate_limit import RateLimitExceeded, limiter_for

logger = get_logger(__name__)

MAX_DISHES = 5  # L-26
SCHEMA_NAME = "restaurant_recommendations"

# Rate-limit reservation per call, corrected to real usage afterwards. Measured on gpt-oss-120b
# (2026-09-15): 3.28 prompt characters per token, 1.0-2.4K completion tokens including reasoning.
# Both are rounded in the safe direction so an estimate rarely undercounts.
CHARS_PER_TOKEN = 3.0
COMPLETION_TOKEN_RESERVE = 2500
PAUSE_AFTER_429_S = 60.0  # when Groq's 429 carries no usable `retry-after`


class RankedPick(BaseModel):
    """Deliberately lenient: ranks may be gapped or 0-based and IDs oddly cased. The grounding gate
    normalizes those (L-14…L-17); rejecting the whole response for them would discard good picks."""

    id: str
    rank: int
    explanation: str
    match_highlights: list[str]


class RankedOutput(BaseModel):
    picks: list[RankedPick]
    summary: str
    caveats: list[str]


_STRINGS = {"type": "array", "items": {"type": "string"}}

# Written out by hand for Groq strict mode: every property required, every object closed,
# no `$ref` or defaults. `test_ranker` checks it against `RankedOutput`.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "picks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "rank": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "match_highlights": _STRINGS,
                },
                "required": ["id", "rank", "explanation", "match_highlights"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "string"},
        "caveats": _STRINGS,
    },
    "required": ["picks", "summary", "caveats"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class LLMRanking:
    output: RankedOutput
    trace: RankingTrace


# ---------------------------------------------------------------------------
# Serialization (3.3)
# ---------------------------------------------------------------------------


def _dumps(value: object) -> str:
    # Byte-stable (L-23) and compact: the same candidates always serialize to the same prompt.
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _opt_bool(value: object) -> bool | None:
    return None if pd.isna(value) else bool(value)


def candidate_record(row: pd.Series) -> dict[str, object]:
    """Only ranking-relevant fields (§5.2). No address, URL or internal scores."""
    return {
        "id": row["restaurant_id"],
        "name": row["name"],
        "area": row["location"] if isinstance(row["location"], str) else None,
        "cuisines": list(row["cuisines"]),
        "type": list(row["rest_type"]),
        "rating": None if pd.isna(row["rating"]) else round(float(row["rating"]), 1),
        "votes": int(row["votes"]),
        "cost_for_two": None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"]),
        "budget_band": row["budget_band"] if row["budget_band"] in BUDGET_BANDS else None,
        "dishes": list(row["dish_liked"])[:MAX_DISHES],
        "online_order": _opt_bool(row["online_order"]),
        "book_table": _opt_bool(row["book_table"]),
        "meets_request": bool(row.get("meets_request", True)),
    }


def serialize_candidates(candidates: pd.DataFrame) -> str:
    return _dumps([candidate_record(row) for _, row in candidates.iterrows()])


def serialize_request(prefs: Preferences, top_n: int, relaxations: Sequence[Relaxation] = ()) -> str:
    return _dumps(
        {
            "location": prefs.location,
            "budget": prefs.budget,
            "cuisines": prefs.cuisines,
            "min_rating": prefs.min_rating,
            "party_size": prefs.party_size,
            "online_order": prefs.online_order,
            "book_table": prefs.book_table,
            "free_text": prefs.free_text,
            "max_picks": top_n,
            "relaxations": [r.reason for r in relaxations],
        }
    )


def build_messages(
    candidates: pd.DataFrame, prefs: Preferences, top_n: int, relaxations: Sequence[Relaxation] = ()
) -> list[dict[str, str]]:
    """Static system prompt first (the cached prefix), then candidates, then preferences (§5.2)."""
    user = (
        f"CANDIDATES:\n{serialize_candidates(candidates)}\n\n"
        f"REQUEST:\n{serialize_request(prefs, top_n, relaxations)}"
    )
    return [{"role": "system", "content": RANKING_SYSTEM_PROMPT}, {"role": "user", "content": user}]


def estimate_tokens(messages: Sequence[dict[str, str]], completion_reserve: int = COMPLETION_TOKEN_RESERVE) -> int:
    return math.ceil(sum(len(m["content"]) for m in messages) / CHARS_PER_TOKEN) + completion_reserve


def _retry_after_s(response: httpx.Response | None) -> float:
    try:
        seconds = float(response.headers["retry-after"])  # type: ignore[union-attr]
    except (AttributeError, KeyError, TypeError, ValueError):  # absent, or an HTTP-date
        return PAUSE_AFTER_429_S
    return min(max(seconds, 1.0), 86_400.0)


# ---------------------------------------------------------------------------
# The guarded call (3.4, 3.7) — shared with the eval judge
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredCompletion:
    content: str
    trace: RankingTrace  # usage and cost of the billed call


def structured_completion(
    messages: Sequence[dict[str, str]],
    *,
    schema_name: str,
    schema: dict,
    config: Settings,
    client: groq.Groq | None = None,
    completion_reserve: int = COMPLETION_TOKEN_RESERVE,
) -> StructuredCompletion:
    """One JSON-schema completion with every guard: key check (L-01), call cap (L-24), the Groq rate
    limiter, the 429 pause, and finish-reason and empty-content checks. Raises `LLMUnavailable` for
    every failure; callers never see a Groq exception. Parsing the content is the caller's job."""
    client = client or get_client(config)
    profile = profile_for(config.model)
    call_budget.spend()
    limiter = limiter_for(config)
    try:
        reservation = limiter.acquire(
            estimate_tokens(messages, completion_reserve), max_wait_s=config.llm_rate_limit_max_wait_s
        )
    except RateLimitExceeded as exc:  # degrade now rather than earn a 429
        raise LLMUnavailable(str(exc)) from exc

    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=config.model,
            messages=list(messages),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": profile.strict_schema, "schema": schema},
            },
            max_completion_tokens=profile.max_completion_tokens,
            **profile.reasoning,
        )
    except groq.RateLimitError as exc:  # L-04: limits used elsewhere on this key; GroqClient doesn't retry
        pause = _retry_after_s(exc.response)
        reservation.settle(0)  # a rejected request consumed no tokens; it still counts as a request
        limiter.block_for(pause, "Groq answered 429 (rate limited)")
        raise LLMUnavailable(f"Groq API error: RateLimitError (429); LLM calls paused for {math.ceil(pause)} s") from exc
    except groq.APIError as exc:  # L-03…L-08, after the SDK's own retries; the reservation stands
        status = getattr(exc, "status_code", None)
        raise LLMUnavailable(f"Groq API error: {type(exc).__name__}" + (f" ({status})" if status else "")) from exc

    usage = completion.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    if prompt_tokens or completion_tokens:
        reservation.settle(prompt_tokens + completion_tokens)
    trace = RankingTrace(
        ranker="llm",
        model=config.model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost_usd(profile, prompt_tokens, cached_tokens, completion_tokens),
        llm_latency_ms=round((time.perf_counter() - started) * 1000),
        rate_limit_wait_ms=round(reservation.waited_s * 1000),
    )
    # 3.7: an identical rerun should log cached_tokens > 0; zero means volatile bytes in the prefix (L-21).
    logger.info("llm call", extra={"schema": schema_name, **trace.model_dump(include={
        "model", "prompt_tokens", "cached_tokens", "completion_tokens", "cost_usd", "llm_latency_ms", "rate_limit_wait_ms"})})

    choice = completion.choices[0] if completion.choices else None
    if choice is None:
        raise LLMUnavailable("response had no choices", trace=trace)
    if choice.finish_reason != "stop":  # L-09 filtered, L-10 truncated at max_completion_tokens
        raise LLMUnavailable(f"finish_reason={choice.finish_reason}", trace=trace)
    content = choice.message.content
    if not content or not content.strip():  # L-11
        raise LLMUnavailable("empty response content", trace=trace)
    return StructuredCompletion(content=content, trace=trace)


def llm_rank(
    candidates: pd.DataFrame,
    prefs: Preferences,
    top_n: int,
    *,
    relaxations: Sequence[Relaxation] = (),
    config: Settings,
    client: groq.Groq | None = None,
) -> LLMRanking:
    """Raises `LLMUnavailable` for every failure mode; callers never see a Groq exception."""
    result = structured_completion(
        build_messages(candidates, prefs, top_n, relaxations),
        schema_name=SCHEMA_NAME,
        schema=RESPONSE_SCHEMA,
        config=config,
        client=client,
    )
    try:
        output = RankedOutput.model_validate_json(result.content)
    except ValidationError as exc:
        raise LLMUnavailable(f"response failed schema validation ({exc.error_count()} errors)", trace=result.trace) from exc
    return LLMRanking(output=output, trace=result.trace)
