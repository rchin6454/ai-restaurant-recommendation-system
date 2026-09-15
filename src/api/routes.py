"""HTTP routes (architecture §7; plan 4.1-4.3).

Thin by design: validate → `recommend()` → return. Retrieval, ranking and grounding all live in
`src/core`; nothing here decides what a user sees.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.config import get_logger, settings
from src.core.models import Budget, Preferences, RecommendationResponse
from src.core.recommender import recommend
from src.data.catalog import get_area_vocabulary, get_catalog, get_cuisine_vocabulary, get_vocabulary
from src.data.cleaning import BUDGET_BANDS

logger = get_logger(__name__)
router = APIRouter()

MAX_TOP_N = 25  # never more than the candidates sent to the ranker (LLM_CANDIDATE_K default)


class BudgetBand(BaseModel):
    """A budget band's real rupee range in this catalog, so the UI can label the radio (§8)."""

    band: Budget
    min_cost: int
    max_cost: int
    restaurants: int


@dataclass(frozen=True)
class CatalogInfo:
    """What the API needs from the catalog, loaded once at startup (A-02)."""

    rows: int
    locations: list[str]
    cuisines: list[str]
    budgets: list[BudgetBand]


def budget_bands(catalog: pd.DataFrame) -> list[BudgetBand]:
    costed = catalog.dropna(subset=["cost_for_two"])
    out = []
    for band in BUDGET_BANDS:
        costs = costed.loc[costed["budget_band"] == band, "cost_for_two"]
        if len(costs):
            out.append(BudgetBand(band=band, min_cost=int(costs.min()), max_cost=int(costs.max()), restaurants=len(costs)))
    return out


def load_catalog_info() -> CatalogInfo:
    """Warm every process-wide cache `recommend()` and `/meta/*` read. Raises `CatalogError` (A-01)."""
    catalog = get_catalog()
    get_vocabulary()
    return CatalogInfo(
        rows=len(catalog),
        locations=get_area_vocabulary(),
        cuisines=get_cuisine_vocabulary(),
        budgets=budget_bands(catalog),
    )


# ---------------------------------------------------------------------------
# Dependencies — tests override these instead of touching the real catalog or LLM
# ---------------------------------------------------------------------------

Recommender = Callable[..., RecommendationResponse]


def catalog_info(request: Request) -> CatalogInfo:
    info = getattr(request.app.state, "catalog_info", None)
    if info is None:  # A-08: never serve empty dropdowns as if they were real
        raise HTTPException(503, "The restaurant catalog is still loading. Try again in a moment.")
    return info


def get_recommender() -> Recommender:
    return recommend


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class RecommendRequest(Preferences):
    """`Preferences` plus per-request options. Inherits its bounds and `extra="forbid"` (I-11, I-12, I-14)."""

    top_n: int | None = Field(None, ge=1, le=MAX_TOP_N)  # I-19
    use_llm: bool = True

    def preferences(self) -> Preferences:
        return Preferences.model_validate(self.model_dump(include=set(Preferences.model_fields)))


class Health(BaseModel):
    status: Literal["ok", "starting"]
    catalog_loaded: bool
    rows: int
    model: str
    # False → every response is degraded (template explanations). The key's validity isn't probed:
    # a bad key still degrades per request with the reason in `trace.fallback_reason` (A-09).
    llm_configured: bool


@router.get("/health", response_model=Health)
def health(request: Request):
    info: CatalogInfo | None = getattr(request.app.state, "catalog_info", None)
    body = Health(
        status="ok" if info else "starting",
        catalog_loaded=info is not None,
        rows=info.rows if info else 0,
        model=settings.model,
        llm_configured=settings.llm_enabled,
    )
    return body if info else JSONResponse(body.model_dump(), status_code=503)


@router.get("/meta/locations", response_model=list[str])
def meta_locations(info: CatalogInfo = Depends(catalog_info)) -> list[str]:
    return info.locations


@router.get("/meta/cuisines", response_model=list[str])
def meta_cuisines(info: CatalogInfo = Depends(catalog_info)) -> list[str]:
    return info.cuisines


@router.get("/meta/budgets", response_model=list[BudgetBand])
def meta_budgets(info: CatalogInfo = Depends(catalog_info)) -> list[BudgetBand]:
    return info.budgets


# `def`, not `async def`: recommend() blocks for seconds on the LLM call, so FastAPI runs it in its
# threadpool instead of stalling the event loop.
@router.post("/recommend", response_model=RecommendationResponse)
def post_recommend(
    body: RecommendRequest,
    request: Request,
    _: CatalogInfo = Depends(catalog_info),
    recommender: Recommender = Depends(get_recommender),
) -> RecommendationResponse:
    response = recommender(body.preferences(), body.top_n, use_llm=body.use_llm)
    logger.info(
        "recommendation served",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "outcome": response.outcome,
            "picks": len(response.recommendations),
            "degraded": response.degraded,
            "ranker": response.trace.ranker if response.trace else None,
            "latency_ms": response.latency_ms,
        },
    )
    return response
