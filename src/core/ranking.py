"""Pre-ranking score, diversity trim, and the deterministic ranker (architecture §4.4, §5.6; plan 2.5-2.6).

`deterministic_rank` is the phase-2 ranker and, from phase 3 on, the LLM fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RankWeights
from src.core.filters import Constraints, combined_mask
from src.core.models import Pick, Recommendations
from src.data.cleaning import BUDGET_BANDS


def _minmax(values: pd.Series) -> pd.Series:
    lo, hi = values.min(), values.max()
    if len(values) == 0 or hi - lo < 1e-12:  # R-02: a flat term contributes a constant, never NaN
        return pd.Series(1.0, index=values.index)
    return (values - lo) / (hi - lo)


def cuisine_overlap(pool: pd.DataFrame, cuisines: frozenset[str] | None) -> pd.Series:
    if not cuisines:
        return pd.Series(0.0, index=pool.index)
    return pd.Series([len(cuisines.intersection(c)) / len(cuisines) for c in pool["cuisines_norm"]], index=pool.index)


def budget_fit(pool: pd.DataFrame, budget: str | None) -> pd.Series:
    """1.0 exact band, 0.5 adjacent, 0 otherwise (incl. unknown cost)."""
    if budget is None:
        return pd.Series(0.0, index=pool.index)
    want = BUDGET_BANDS.index(budget)
    fit = {b: {0: 1.0, 1: 0.5}.get(abs(BUDGET_BANDS.index(b) - want), 0.0) for b in BUDGET_BANDS}
    return pd.Series([fit.get(b, 0.0) if isinstance(b, str) else 0.0 for b in pool["budget_band"]], index=pool.index)


def score_pool(pool: pd.DataFrame, requested: Constraints, weights: RankWeights, rating_prior: float) -> pd.DataFrame:
    """Adds `meets_request`, `score`, `cuisine_overlap`, `budget_fit`, sorted best-first.

    Scored against what the user *asked for* (unrelaxed). Rows meeting the whole original request
    always come first; relaxed rows (a budget stretch, a nearby area…) only fill in behind them.
    Otherwise a 4.7★ stretch row beats every exact match — the budget weight is too small to stop it.
    """
    requested_budget = requested.bands[0] if requested.bands else None
    bayes = pool["bayesian_rating"].astype("float64").fillna(rating_prior)  # R-04
    terms = {
        "rating": (weights.rating, _minmax(bayes)),
        "votes": (weights.votes, _minmax(np.log1p(pool["votes"].astype("float64")))),
    }
    overlap = cuisine_overlap(pool, requested.cuisines)
    fit = budget_fit(pool, requested_budget)
    if requested.cuisines:
        terms["cuisine"] = (weights.cuisine, overlap)
    if requested_budget:
        terms["budget"] = (weights.budget, fit)
    total = sum(w for w, _ in terms.values())  # R-03: unused terms' weight is redistributed
    score = sum((w / total) * term for w, term in terms.values()) if total else pd.Series(0.0, index=pool.index)

    scored = pool.assign(
        meets_request=combined_mask(pool, requested), score=score.round(9), cuisine_overlap=overlap, budget_fit=fit
    )
    # R-01: stable, run-to-run identical order.
    return scored.sort_values(
        ["meets_request", "score", "votes", "restaurant_id"], ascending=[False, False, False, True], kind="mergesort"
    )


def diversity_trim(ranked: pd.DataFrame, *, k: int, max_chain_outlets: int) -> pd.DataFrame:
    """Walk the whole ranked pool, keeping at most `max_chain_outlets` per chain, until `k` rows.

    Chains are exact `name_norm` matches (R-06). Because trimming happens before truncation,
    rows cut for being a third outlet are backfilled by the next-best other restaurants (R-05);
    a list is only shorter than `k` when the pool genuinely has too few distinct options (R-08).
    """
    kept: list[int] = []
    per_chain: dict[str, int] = {}
    for pos, name in enumerate(ranked["name_norm"]):
        if per_chain.get(name, 0) >= max_chain_outlets:
            continue
        per_chain[name] = per_chain.get(name, 0) + 1
        kept.append(pos)
        if len(kept) >= k:
            break
    return ranked.iloc[kept]


def pre_rank(
    pool: pd.DataFrame,
    requested: Constraints,
    *,
    weights: RankWeights,
    k: int,
    max_chain_outlets: int,
    rating_prior: float,
) -> pd.DataFrame:
    """The token gate: score, trim chains, keep the top `k` (§4.4)."""
    if pool.empty:
        return pool.assign(score=pd.Series(dtype="float64"))
    return diversity_trim(score_pool(pool, requested, weights, rating_prior), k=k, max_chain_outlets=max_chain_outlets)


# ---------------------------------------------------------------------------
# Deterministic ranker (2.6)
# ---------------------------------------------------------------------------


def deterministic_rank(
    candidates: pd.DataFrame,
    requested: Constraints,
    top_n: int,
    *,
    pool_size: int,
    free_text: str | None = None,
) -> Recommendations:
    """Keep the pre-ranked order; explain each pick with a template built only from catalog fields."""
    rows = candidates.head(top_n)  # I-18: fewer if that's all there is — never pad
    picks = []
    for rank, (_, row) in enumerate(rows.iterrows(), start=1):
        explanation, highlights = explain(row, requested)
        picks.append(Pick(id=row["restaurant_id"], rank=rank, explanation=explanation, match_highlights=highlights))

    summary = (
        f"Top {len(picks)} of {pool_size:,} matching restaurant{'s' if pool_size != 1 else ''}, "
        "ranked by rating, popularity, cuisine and budget fit."
    )
    caveats = []
    if free_text:
        caveats.append("Your free-text preferences weren't used: they need the AI ranker, and these results are ranked without it.")
    return Recommendations(picks=picks, summary=summary, caveats=caveats)


def explain(row: pd.Series, requested: Constraints) -> tuple[str, list[str]]:
    """`"4.3★ from 512 votes · North Indian · ₹800 for two — matches your budget and cuisine."`"""
    rating, votes, cost = row["rating"], int(row["votes"]), row["cost_for_two"]
    cuisines = list(row["cuisines"])
    matched_cuisines = [c for c in cuisines if requested.cuisines and c.casefold() in requested.cuisines]

    facts = [
        f"{float(rating):.1f}★ from {votes:,} vote{'s' if votes != 1 else ''}"
        if not pd.isna(rating)
        else ("New — not yet rated" if bool(row["is_new"]) else "Not yet rated"),  # U-02
    ]
    shown_cuisines = matched_cuisines or cuisines[:2]
    if shown_cuisines:
        facts.append(", ".join(shown_cuisines))
    facts.append(f"₹{int(cost):,} for two" if not pd.isna(cost) else "cost not listed")  # U-03

    matches: list[str] = []
    misses: list[str] = []
    highlights: list[str] = []
    area = row["location"] if isinstance(row["location"], str) else None

    if requested.areas is not None:
        if row["location_norm"] in requested.areas:
            matches.append("area")
            highlights.append(area or "your area")
        else:
            misses.append(f"it's in {area or 'another area'}, outside the area you chose")
    requested_budget = requested.bands[0] if requested.bands else None
    if requested_budget:
        band = row["budget_band"] if isinstance(row["budget_band"], str) else None
        if band == requested_budget:
            matches.append("budget")
            highlights.append("within budget")
        elif band is not None:
            direction = "above" if BUDGET_BANDS.index(band) > BUDGET_BANDS.index(requested_budget) else "below"
            misses.append(f"it's one budget band {direction} your {requested_budget} budget")
    if requested.cuisines:
        if matched_cuisines:
            matches.append("cuisine")
            highlights.extend(matched_cuisines)
        else:
            misses.append("it doesn't list the cuisine you asked for")
    if requested.min_rating is not None and not pd.isna(rating):
        if float(rating) >= requested.min_rating - 1e-9:
            matches.append("rating")
            highlights.append(f"{requested.min_rating:g}★+")
        else:
            misses.append(f"it's rated below your {requested.min_rating:g}★ minimum")

    text = " · ".join(facts)
    if matches:
        text += f" — matches your {_join(matches)}."
    elif not misses:
        text += "."
    if misses:
        text += f"{' ' if matches else ' — '}Filters were relaxed: {'; '.join(misses)}."
    return text, highlights


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else f"{', '.join(words[:-1])} and {words[-1]}"
