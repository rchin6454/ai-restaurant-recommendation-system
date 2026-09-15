"""Serialize candidates → call Groq → parse structured output (architecture §5.2-§5.4; plan 3.3-3.4, 3.7).

The model returns IDs and prose only. Checking those IDs against the candidate set is the
grounding gate's job (`src.core.recommender.validate_and_join`), not this module's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import groq
import pandas as pd
from pydantic import BaseModel, ValidationError

from src.config import Settings, get_logger
from src.core.models import Preferences, RankingTrace, Relaxation
from src.data.cleaning import BUDGET_BANDS
from src.llm.client import LLMUnavailable, call_budget, estimate_cost_usd, get_client, profile_for
from src.llm.prompts import RANKING_SYSTEM_PROMPT

logger = get_logger(__name__)

MAX_DISHES = 5  # L-26
SCHEMA_NAME = "restaurant_recommendations"


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


# ---------------------------------------------------------------------------
# The call (3.4, 3.7)
# ---------------------------------------------------------------------------


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
    client = client or get_client(config)
    profile = profile_for(config.model)
    call_budget.spend()

    started = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=config.model,
            messages=build_messages(candidates, prefs, top_n, relaxations),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "strict": profile.strict_schema, "schema": RESPONSE_SCHEMA},
            },
            max_completion_tokens=profile.max_completion_tokens,
            **profile.reasoning,
        )
    except groq.APIError as exc:  # L-03…L-08, after the SDK's own retries
        status = getattr(exc, "status_code", None)
        raise LLMUnavailable(f"Groq API error: {type(exc).__name__}" + (f" ({status})" if status else "")) from exc

    usage = completion.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    cached_tokens = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    trace = RankingTrace(
        ranker="llm",
        model=config.model,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        cost_usd=estimate_cost_usd(profile, prompt_tokens, cached_tokens, completion_tokens),
        llm_latency_ms=round((time.perf_counter() - started) * 1000),
    )
    # 3.7: an identical rerun should log cached_tokens > 0; zero means volatile bytes in the prefix (L-21).
    logger.info("llm call", extra=trace.model_dump(include={"model", "prompt_tokens", "cached_tokens",
                                                            "completion_tokens", "cost_usd", "llm_latency_ms"}))

    choice = completion.choices[0] if completion.choices else None
    if choice is None:
        raise LLMUnavailable("response had no choices", trace=trace)
    if choice.finish_reason != "stop":  # L-09 filtered, L-10 truncated at max_completion_tokens
        raise LLMUnavailable(f"finish_reason={choice.finish_reason}", trace=trace)
    content = choice.message.content
    if not content or not content.strip():  # L-11
        raise LLMUnavailable("empty response content", trace=trace)
    try:
        output = RankedOutput.model_validate_json(content)
    except ValidationError as exc:
        raise LLMUnavailable(f"response failed schema validation ({exc.error_count()} errors)", trace=trace) from exc
    return LLMRanking(output=output, trace=trace)
