"""Orchestration (architecture §6; plan 3.5-3.6).

normalize → retrieve → pre-rank → LLM rank (deterministic fallback) → grounding gate → response.
This is the only module that knows the full sequence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import groq
import pandas as pd

from src.config import Settings, get_logger, settings as default_settings
from src.core import filters
from src.core.models import (
    Preferences,
    RankingTrace,
    Recommendation,
    RecommendationResponse,
    Recommendations,
)
from src.core.ranking import deterministic_rank, explain, pre_rank
from src.data.catalog import Vocabulary, build_vocabulary, get_catalog, get_vocabulary
from src.data.cleaning import BUDGET_BANDS
from src.llm.client import LLMUnavailable
from src.llm.ranker import RankedOutput, llm_rank

logger = get_logger(__name__)


def recommend(
    prefs: Preferences,
    top_n: int | None = None,
    *,
    catalog: pd.DataFrame | None = None,
    config: Settings | None = None,
    use_llm: bool = True,
    llm_client: groq.Groq | None = None,
) -> RecommendationResponse:
    started = time.perf_counter()
    config = config or default_settings
    top_n = top_n or config.default_top_n
    if top_n < 1:
        raise ValueError("top_n must be at least 1")  # I-19
    if catalog is None:
        catalog, vocab = get_catalog(), get_vocabulary()
    else:
        vocab = build_vocabulary(catalog)
    llm_available = use_llm and (config.llm_enabled or llm_client is not None)

    def respond(**fields) -> RecommendationResponse:
        fields.setdefault("degraded", not llm_available)
        return RecommendationResponse(latency_ms=round((time.perf_counter() - started) * 1000), **fields)

    normalized = filters.normalize(prefs, vocab)
    if normalized.coverage_error:  # I-02 / W-1: stop — never answer a Delhi query with Bengaluru results
        return respond(
            outcome="coverage_error",
            summary=normalized.coverage_error,
            interpretations=normalized.interpretations,
            suggestions=list(normalized.suggestions),
        )

    caveats: list[str] = []
    if normalized.unknown_cuisines:  # I-06
        missing = ", ".join(repr(c) for c in normalized.unknown_cuisines)
        closest = f" Closest available: {', '.join(normalized.suggestions)}." if normalized.suggestions else ""
        if not normalized.prefs.cuisines:
            return respond(
                outcome="empty_with_reason",
                summary=f"No restaurant in the catalog serves {missing}.{closest}",
                interpretations=normalized.interpretations,
                suggestions=list(normalized.suggestions),
            )
        caveats.append(f"Ignored {missing}: not a cuisine in the catalog.{closest}")

    requested = filters.Constraints.from_prefs(normalized.prefs, vocab)
    retrieval = filters.retrieve(catalog, normalized.prefs, vocab, min_candidates=config.min_candidates)
    applied = filters.applied_filters(retrieval.constraints, vocab)
    caveats.extend(r.reason for r in retrieval.relaxations)  # F-09
    if retrieval.hidden_unrated:  # F-10
        caveats.append(
            f"{retrieval.hidden_unrated} new or unrated restaurant{'s' if retrieval.hidden_unrated != 1 else ''} "
            "hidden by the rating filter."
        )

    if retrieval.pool.empty:  # F-02: nothing to rank, so no ranker and no LLM call
        return respond(
            outcome="empty_with_reason",
            summary=_empty_summary(retrieval, vocab),
            caveats=caveats,
            applied_filters=applied,
            relaxations=retrieval.relaxations,
            interpretations=normalized.interpretations,
            blocking_constraints=[name for name, _ in retrieval.blocking],
        )

    candidates = pre_rank(
        retrieval.pool,
        requested,
        weights=config.rank_weights,
        k=config.llm_candidate_k,
        max_chain_outlets=config.max_chain_outlets,
        rating_prior=vocab.rating_prior,
    )
    stretch_band = _stretch_band(retrieval, requested)
    if not use_llm:
        trace = RankingTrace(ranker="deterministic", fallback_reason="LLM ranking disabled for this request")
    elif not llm_available:  # L-01, L-02: no key → don't even try
        trace = RankingTrace(ranker="deterministic", fallback_reason="GROQ_API_KEY is not set")
    else:
        trace = None
    grounded, summary, ranker_caveats, trace = _rank_and_ground(
        candidates, requested, normalized.prefs, top_n,
        retrieval=retrieval, stretch_band=stretch_band, config=config, llm_client=llm_client, trace=trace,
    )
    return respond(
        outcome="relaxed_results" if retrieval.relaxations else "results",
        recommendations=grounded.recommendations,
        summary=summary,
        caveats=caveats + ranker_caveats,
        applied_filters=applied,
        relaxations=retrieval.relaxations,
        interpretations=normalized.interpretations,
        candidates_considered=len(candidates),
        degraded=trace.ranker == "deterministic",
        trace=trace,
    )


def _rank_and_ground(
    candidates: pd.DataFrame,
    requested: filters.Constraints,
    prefs: Preferences,
    top_n: int,
    *,
    retrieval: filters.Retrieval,
    stretch_band: str | None,
    config: Settings,
    llm_client: groq.Groq | None,
    trace: RankingTrace | None,
) -> tuple[Grounded, str, list[str], RankingTrace]:
    """LLM first, unless `trace` already says why not. Any failure — transport, response shape, or no
    pick surviving the grounding gate — falls back to the deterministic ranker (§5.6)."""
    if trace is None:
        try:
            result = llm_rank(candidates, prefs, top_n, relaxations=retrieval.relaxations, config=config, client=llm_client)
        except LLMUnavailable as exc:
            logger.warning("llm ranking failed; using deterministic fallback", extra={"reason": str(exc)})
            trace = (exc.trace or RankingTrace(ranker="deterministic", model=config.model)).model_copy(
                update={"ranker": "deterministic", "fallback_reason": str(exc)}
            )
        else:
            grounded = validate_and_join(result.output, candidates, top_n, requested=requested, stretch_band=stretch_band)
            trace = result.trace.model_copy(
                update={
                    "model_picks": len(result.output.picks),
                    "dropped_ids": grounded.dropped_ids,
                    "backfilled": grounded.backfilled,
                }
            )
            if grounded.from_model:
                summary = result.output.summary.strip() or f"{len(grounded.recommendations)} picks for your request."
                return grounded, summary, [c.strip() for c in result.output.caveats if c.strip()], trace
            logger.warning("grounding: no valid picks from the model; using deterministic fallback",  # G-02, L-12
                           extra={"model_picks": len(result.output.picks)})
            trace = trace.model_copy(
                update={"ranker": "deterministic", "fallback_reason": "no valid picks survived the grounding gate"}
            )

    ranked = deterministic_rank(candidates, requested, top_n, pool_size=len(retrieval.pool), free_text=prefs.free_text)
    grounded = validate_and_join(ranked, candidates, top_n, requested=requested, stretch_band=stretch_band)
    return grounded, ranked.summary, ranked.caveats, trace


# ---------------------------------------------------------------------------
# Grounding gate (3.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grounded:
    recommendations: list[Recommendation]
    dropped_ids: list[str]  # unknown or duplicate IDs, exactly as the ranker returned them
    backfilled: int  # cards filled from pre-ranked order after drops
    from_model: int  # cards whose pick came from the ranker


def validate_and_join(
    ranked: RankedOutput | Recommendations,
    candidates: pd.DataFrame,
    top_n: int,
    *,
    requested: filters.Constraints,
    stretch_band: str | None = None,
) -> Grounded:
    """The grounding gate (§6). Every card in every response passes through here.

    Keeps only IDs present in *this* candidate set (G-01, G-03), orders by the ranker's ranks and
    renumbers 1..N (L-15), drops duplicates (L-14), and joins every displayed fact from the catalog
    row — the ranker contributes order and prose, nothing else (G-05).
    """
    rows = candidates.set_index("restaurant_id", drop=False)
    by_key = {_id_key(rid): rid for rid in rows.index}

    chosen: list[tuple[str, str, list[str]]] = []
    dropped: list[str] = []
    for _, pick in sorted(enumerate(ranked.picks), key=lambda t: (t[1].rank, t[0])):
        rid = by_key.get(_id_key(pick.id))  # L-16: case/whitespace drift isn't a hallucination
        if rid is None or any(rid == c[0] for c in chosen):
            dropped.append(pick.id)
            continue
        chosen.append((rid, pick.explanation, pick.match_highlights))
    if dropped:
        logger.warning("grounding: dropped %d unknown or duplicate IDs", len(dropped), extra={"dropped_ids": dropped})

    chosen = chosen[:top_n]  # L-13
    from_model = len(chosen)
    backfilled = 0
    # G-04: refill only the slots dropped picks vacated. A ranker that returned fewer picks on purpose
    # is not padded (§5.6), and one that returned nothing valid is the caller's fallback, not a backfill.
    target = min(top_n, len(ranked.picks))
    if from_model:
        taken = {c[0] for c in chosen}
        for rid in rows.index:
            if len(chosen) >= target:
                break
            if rid not in taken:
                chosen.append((rid, "", []))
                backfilled += 1

    cards = []
    for rank, (rid, explanation, highlights) in enumerate(chosen, start=1):
        row = rows.loc[rid]
        if not explanation.strip():  # L-17, and every backfilled card
            explanation, highlights = explain(row, requested)
        cards.append(_card(row, rank, explanation.strip(), [h.strip() for h in highlights if h.strip()], stretch_band))
    return Grounded(recommendations=cards, dropped_ids=dropped, backfilled=backfilled, from_model=from_model)


def _id_key(value: str) -> str:
    return value.strip().casefold()


def _card(row: pd.Series, rank: int, explanation: str, highlights: list[str], stretch_band: str | None) -> Recommendation:
    band = row["budget_band"] if row["budget_band"] in BUDGET_BANDS else None
    return Recommendation(
        rank=rank,
        restaurant_id=row["restaurant_id"],
        name=row["name"],
        cuisines=list(row["cuisines"]),
        rating=None if pd.isna(row["rating"]) else float(row["rating"]),
        votes=int(row["votes"]),
        cost_for_two=None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"]),
        budget_band=band,
        stretch=band is not None and band == stretch_band,
        area=row["location"] if isinstance(row["location"], str) else None,
        url=row["url"] if isinstance(row["url"], str) and row["url"] else None,
        explanation=explanation,
        match_highlights=highlights,
    )


def _stretch_band(retrieval: filters.Retrieval, requested: filters.Constraints) -> str | None:
    if not requested.bands or not any(r.step == 0 for r in retrieval.relaxations):
        return None
    idx = BUDGET_BANDS.index(requested.bands[0]) + 1
    return BUDGET_BANDS[idx] if idx < len(BUDGET_BANDS) else None


def _empty_summary(retrieval: filters.Retrieval, vocab: Vocabulary) -> str:
    c = retrieval.constraints
    labels = {"min_rating": "rating", "budget": "budget", "location": "location", "cuisines": "cuisine"}
    relaxed = list(dict.fromkeys(labels[r.field] for r in retrieval.relaxations))  # ladder order
    prefix = f"No restaurants match, even after relaxing {', '.join(relaxed)}." if relaxed else "No restaurants match."
    if retrieval.blocking:
        name, count = retrieval.blocking[0]
        return (
            f"{prefix} Blocking constraint: {filters.describe_constraint(name, c, vocab)} "
            f"— without it there would be {count:,} match{'es' if count != 1 else ''}."
        )
    active = [n for n in ("location", "budget", "cuisines", "min_rating", "online_order", "book_table") if _is_active(c, n)]
    described = "; ".join(filters.describe_constraint(n, c, vocab) for n in active)
    return f"{prefix} No single constraint is to blame — the combination is impossible: {described}."


def _is_active(c: filters.Constraints, name: str) -> bool:
    attr = {"location": "areas", "budget": "bands"}.get(name, name)
    return getattr(c, attr) is not None
