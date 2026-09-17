"""Request/response models shared by every layer (architecture §4.1, §5.4, §7; plan task 2.1).

The full response shape — transparency fields included — is defined up front so the API and
UI never have to be retrofitted.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Budget = Literal["low", "medium", "high"]
Outcome = Literal["results", "relaxed_results", "empty_with_reason", "coverage_error"]
ConstraintName = Literal["location", "budget", "cuisines", "min_rating", "online_order", "book_table"]
MAX_FREE_TEXT_CHARS = 500  # §13 / I-14
MAX_CUISINE_CHARS = 60  # the longest catalog cuisine is ~20 characters (5.3)
MAX_PARTY_SIZE = 100
# C0 control characters except tab and newline, plus DEL: invisible in the UI, noise in the prompt (5.3).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class Preferences(BaseModel):
    """What the user asked for. `free_text` is never turned into a filter — it's the LLM's job (§4.1)."""

    model_config = ConfigDict(extra="forbid")

    location: str | None = Field(None, max_length=100, description="Bengaluru area")
    budget: Budget | None = None  # I-12: anything else is a validation error
    cuisines: list[Annotated[str, StringConstraints(max_length=MAX_CUISINE_CHARS)]] = Field(
        default_factory=list, max_length=20
    )  # OR-matched
    min_rating: float | None = Field(None, ge=0, le=5)  # I-11
    party_size: int | None = Field(None, ge=1, le=MAX_PARTY_SIZE)  # I-20: context only, never filtered on
    free_text: str | None = Field(None, max_length=MAX_FREE_TEXT_CHARS)
    online_order: bool | None = None
    book_table: bool | None = None

    @field_validator("location", "free_text", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:  # I-13
        if isinstance(v, str):
            v = _CONTROL_CHARS.sub("", v).strip()
            return v or None
        return v

    @field_validator("cuisines", mode="before")
    @classmethod
    def _clean_cuisines(cls, v: object) -> object:  # I-08
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            return v
        seen: dict[str, str] = {}
        for item in v:
            if not isinstance(item, str):
                return v  # let pydantic report the bad element
            item = _CONTROL_CHARS.sub("", item).strip()
            if item:
                seen.setdefault(item.casefold(), item)
        return list(seen.values())


class Interpretation(BaseModel):
    """How a free-typed value was resolved, e.g. "koramangla" → "Koramangala" (I-03)."""

    field: Literal["location", "cuisines"]
    input: str
    matched: str | None
    score: float
    note: str


class Relaxation(BaseModel):
    """One widening step. Serialized with `from`/`to` keys (§7)."""

    model_config = ConfigDict(populate_by_name=True)

    field: Literal["min_rating", "budget", "location", "cuisines"]
    from_: float | list[str] | None = Field(alias="from")
    to: float | list[str] | None
    step: int = Field(description="0 = budget stretch (§4.2); 1-4 = ladder position (§4.3)")
    matches_before: int
    matches_after: int
    reason: str


class AppliedFilters(BaseModel):
    """The constraints the results actually satisfy — after any relaxation."""

    location: list[str] | None = None
    budget: list[Budget] | None = None
    cuisines: list[str] | None = None
    min_rating: float | None = None
    online_order: bool | None = None
    book_table: bool | None = None


class Pick(BaseModel):
    """A ranker's choice: an ID plus prose. Deliberately carries no facts (§5.4)."""

    id: str
    rank: int = Field(ge=1)
    explanation: str
    match_highlights: list[str] = Field(default_factory=list)


class Recommendations(BaseModel):
    """Ranker output — the LLM's structured-output schema, and what `deterministic_rank` returns."""

    picks: list[Pick]
    summary: str
    caveats: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    """A result card. Every fact comes from the catalog row, never from the ranker."""

    rank: int = Field(ge=1)
    restaurant_id: str
    name: str
    cuisines: list[str]
    rating: float | None  # None = new/unrated, never 0.0 (U-02)
    votes: int
    cost_for_two: int | None  # None = cost unavailable (U-03)
    budget_band: Budget | None
    stretch: bool = Field(False, description="admitted one budget band above the request (§4.2)")
    area: str | None
    url: str | None
    explanation: str
    match_highlights: list[str] = Field(default_factory=list)


class RankingTrace(BaseModel):
    """How the ranking was produced — for the CLI, eval metrics (M-02, M-22, M-23) and debugging.

    Operational metadata only; nothing here is ever displayed as a restaurant fact.
    """

    ranker: Literal["llm", "deterministic"]
    model: str | None = None
    fallback_reason: str | None = None  # why the deterministic ranker answered instead of the LLM
    prompt_tokens: int = 0
    cached_tokens: int = 0  # §5.2 / 3.7: > 0 on a repeat query means the prompt prefix is stable
    completion_tokens: int = 0
    cost_usd: float | None = None  # None = no price known for this model
    llm_latency_ms: int = 0
    rate_limit_wait_ms: int = 0  # queued behind the Groq rate limiter before the call; M-20 excludes it
    model_picks: int = 0  # picks the model returned, before the grounding gate
    dropped_ids: list[str] = Field(default_factory=list)  # G-07: unknown/duplicate IDs the gate removed
    backfilled: int = 0  # G-04: slots refilled from pre-ranked order


class RecommendationResponse(BaseModel):
    outcome: Outcome
    recommendations: list[Recommendation] = Field(default_factory=list)
    summary: str
    caveats: list[str] = Field(default_factory=list)
    applied_filters: AppliedFilters = Field(default_factory=AppliedFilters)
    relaxations: list[Relaxation] = Field(default_factory=list)
    interpretations: list[Interpretation] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    # Empty pool only (F-02): constraints whose removal alone yields matches, most matches first.
    # Lets the UI offer one-click relaxation (U-05) without parsing `summary`.
    blocking_constraints: list[ConstraintName] = Field(default_factory=list)
    # True whenever explanations are templates rather than the LLM's (§5.6).
    degraded: bool
    candidates_considered: int = 0
    latency_ms: int = 0
    cached: bool = False  # served from the response cache (5.1)
    trace: RankingTrace | None = None  # None when no ranking ran (coverage error, empty pool)
