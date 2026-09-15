"""Preference normalization, filter chain, and relaxation ladder (architecture §4.1-§4.3, plan 2.2-2.4).

Each predicate is its own boolean mask so a response can say *which* constraint emptied the pool.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.config import get_logger
from src.core.models import AppliedFilters, Interpretation, Preferences, Relaxation
from src.data.catalog import VocabEntry, Vocabulary
from src.data.cleaning import AREA_ALIASES, BUDGET_BANDS, CUISINE_ALIASES

logger = get_logger(__name__)

FUZZY_THRESHOLD = 85  # §4.2
SUGGESTION_MIN_SCORE = 60  # I-05: near misses worth offering
STRETCH_THRESHOLD = 15  # §4.2 budget soft edge
RATING_STEP = 0.3  # §4.3 step 1
RATING_FLOOR = 3.0
CITY_TERMS = frozenset({"bangalore", "bengaluru", "blr", "bangaluru"})

_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Normalization (2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Match:
    key: str | None
    display: str | None
    score: float
    suggestions: tuple[str, ...] = ()


def fuzzy_match(term: str, entries: Mapping[str, VocabEntry], aliases: Mapping[str, str] | None = None) -> Match:
    """Exact → alias → best `fuzz.ratio` ≥ 85. Equal scores go to the entry with more votes (I-04)."""
    query = _WHITESPACE.sub(" ", term).strip().casefold()
    if not query or not entries:
        return Match(None, None, 0.0)
    if query in entries:
        return Match(query, entries[query].display, 100.0)
    alias = (aliases or {}).get(query)
    if alias and alias.casefold() in entries:
        key = alias.casefold()
        return Match(key, entries[key].display, 100.0)

    scored = sorted(((fuzz.ratio(query, k), k) for k in entries), key=lambda t: (-t[0], -entries[t[1]].votes, t[1]))
    best_score, best_key = scored[0]
    if best_score >= FUZZY_THRESHOLD:
        return Match(best_key, entries[best_key].display, round(best_score, 1))
    suggestions = tuple(entries[k].display for s, k in scored[:3] if s >= SUGGESTION_MIN_SCORE)
    return Match(None, None, round(best_score, 1), suggestions)


@dataclass(frozen=True)
class Normalized:
    prefs: Preferences  # location/cuisines replaced by canonical display names; unknown cuisines removed
    interpretations: list[Interpretation]
    coverage_error: str | None = None  # I-02: set → stop, never fall back to Bengaluru results
    unknown_cuisines: tuple[str, ...] = ()  # I-06
    suggestions: tuple[str, ...] = ()


def normalize(prefs: Preferences, vocab: Vocabulary) -> Normalized:
    interpretations: list[Interpretation] = []
    location = prefs.location
    coverage_error = None
    suggestions: list[str] = []

    if location is not None:
        query = _WHITESPACE.sub(" ", location).strip().casefold()
        if query in CITY_TERMS:
            interpretations.append(
                Interpretation(field="location", input=location, matched=None, score=100.0,
                               note=f"'{location}' is the whole city, so no area filter was applied")
            )
            location = None
        else:
            match = fuzzy_match(location, vocab.areas, AREA_ALIASES)
            if match.key is None:
                coverage_error = _coverage_message(location, match, vocab)
                suggestions.extend(match.suggestions)
            else:
                if match.key != query:
                    interpretations.append(
                        Interpretation(field="location", input=location, matched=match.display, score=match.score,
                                       note=f"interpreted '{location}' as '{match.display}'")
                    )
                location = match.display

    cuisines: list[str] = []
    unknown: list[str] = []
    for raw in prefs.cuisines:
        match = fuzzy_match(raw, vocab.cuisines, CUISINE_ALIASES)
        if match.key is None:
            unknown.append(raw)
            suggestions.extend(s for s in match.suggestions if s not in suggestions)
            continue
        if match.display not in cuisines:
            cuisines.append(match.display)
        if match.key != raw.strip().casefold():
            interpretations.append(
                Interpretation(field="cuisines", input=raw, matched=match.display, score=match.score,
                               note=f"interpreted '{raw}' as '{match.display}'")
            )

    return Normalized(
        prefs=prefs.model_copy(update={"location": location, "cuisines": cuisines}),
        interpretations=interpretations,
        coverage_error=coverage_error,
        unknown_cuisines=tuple(unknown),
        suggestions=tuple(suggestions),
    )


def _coverage_message(location: str, match: Match, vocab: Vocabulary) -> str:
    top = sorted(vocab.areas.values(), key=lambda e: (-e.rows, e.display))[:8]
    message = (
        f"'{location}' isn't an area in this dataset, which covers Bengaluru only. "
        f"Areas available include {', '.join(e.display for e in top)} ({len(vocab.areas)} in total)."
    )
    if match.suggestions:
        message += f" Did you mean {' or '.join(repr(s) for s in match.suggestions)}?"
    return message


# ---------------------------------------------------------------------------
# Filter chain (2.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """The effective filters at one point on the ladder. `None` = not constrained."""

    areas: frozenset[str] | None = None  # `location_norm` values
    location_key: str | None = None  # the matched area, for finding neighbours
    bands: tuple[str, ...] | None = None  # kept in low→high order
    cuisines: frozenset[str] | None = None  # `cuisines_norm` values
    min_rating: float | None = None
    online_order: bool | None = None
    book_table: bool | None = None

    @classmethod
    def from_prefs(cls, prefs: Preferences, vocab: Vocabulary) -> Constraints:
        """`prefs` must be normalized: location/cuisines are canonical display names."""
        key = prefs.location.casefold() if prefs.location else None
        return cls(
            areas=frozenset(vocab.area_families.get(key, (key,))) if key else None,
            location_key=key,
            bands=(prefs.budget,) if prefs.budget else None,
            cuisines=frozenset(c.casefold() for c in prefs.cuisines) or None,
            min_rating=prefs.min_rating,
            online_order=prefs.online_order,
            book_table=prefs.book_table,
        )


def location_mask(df: pd.DataFrame, areas: frozenset[str]) -> np.ndarray:
    return df["location_norm"].isin(sorted(areas)).to_numpy(dtype=bool)


def budget_mask(df: pd.DataFrame, bands: tuple[str, ...]) -> np.ndarray:
    # A null cost has no band, so it drops out whenever a budget is requested (D-10).
    return df["budget_band"].isin(list(bands)).to_numpy(dtype=bool)


def cuisine_mask(df: pd.DataFrame, cuisines: frozenset[str]) -> np.ndarray:
    # OR: any requested cuisine (§4.2). AND over three cuisines empties the set.
    return np.fromiter((not cuisines.isdisjoint(c) for c in df["cuisines_norm"]), dtype=bool, count=len(df))


def rating_mask(df: pd.DataFrame, min_rating: float) -> np.ndarray:
    # Unrated rows are NA, so even min_rating=0 excludes them (I-10).
    return (df["rating"] >= min_rating - 1e-9).fillna(False).to_numpy(dtype=bool)


def flag_mask(df: pd.DataFrame, column: str, value: bool) -> np.ndarray:
    return (df[column] == value).fillna(False).to_numpy(dtype=bool)


def constraint_masks(df: pd.DataFrame, c: Constraints) -> dict[str, np.ndarray]:
    """Only constraints that are set get a mask — `None` never narrows the pool (F-12)."""
    masks: dict[str, np.ndarray] = {}
    if c.areas is not None:
        masks["location"] = location_mask(df, c.areas)
    if c.bands is not None:
        masks["budget"] = budget_mask(df, c.bands)
    if c.cuisines is not None:
        masks["cuisines"] = cuisine_mask(df, c.cuisines)
    if c.min_rating is not None:
        masks["min_rating"] = rating_mask(df, c.min_rating)
    if c.online_order is not None:
        masks["online_order"] = flag_mask(df, "online_order", c.online_order)
    if c.book_table is not None:
        masks["book_table"] = flag_mask(df, "book_table", c.book_table)
    return masks


def combined_mask(df: pd.DataFrame, c: Constraints) -> np.ndarray:
    masks = list(constraint_masks(df, c).values())
    return np.logical_and.reduce(masks) if masks else np.ones(len(df), dtype=bool)


# ---------------------------------------------------------------------------
# Relaxation ladder (2.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Retrieval:
    pool: pd.DataFrame
    constraints: Constraints  # final, after relaxation
    relaxations: list[Relaxation]
    hidden_unrated: int  # F-10: unrated rows the rating filter excluded
    blocking: list[tuple[str, int]]  # when the pool is empty: (constraint, matches without it)


Step = Callable[[Constraints, Vocabulary], "tuple[Constraints, str, object, object] | None"]


def _relax_rating(c: Constraints, vocab: Vocabulary):
    if c.min_rating is None or c.min_rating <= RATING_FLOOR:  # F-04
        return None
    to = max(RATING_FLOOR, round(c.min_rating - RATING_STEP, 1))
    return replace(c, min_rating=to), "min_rating", c.min_rating, to


def _widen_bands(bands: tuple[str, ...], up_only: bool = False) -> tuple[str, ...] | None:
    idx = sorted(BUDGET_BANDS.index(b) for b in bands)
    if idx[-1] + 1 < len(BUDGET_BANDS):
        return tuple(BUDGET_BANDS[i] for i in [*idx, idx[-1] + 1])
    if not up_only and idx[0] > 0:  # F-05: already at `high` → widen downward
        return tuple(BUDGET_BANDS[i] for i in [idx[0] - 1, *idx])
    return None


def _relax_budget(c: Constraints, vocab: Vocabulary):
    if c.bands is None:
        return None
    wider = _widen_bands(c.bands)
    if wider is None:
        return None
    return replace(c, bands=wider), "budget", list(c.bands), list(wider)


def _relax_location_nearby(c: Constraints, vocab: Vocabulary):
    if c.areas is None or c.location_key is None:
        return None
    extra = set(vocab.nearby_areas.get(c.location_key, ())) - c.areas
    if not extra:  # F-06: no neighbours known → the next step widens to the whole city
        return None
    wider = c.areas | extra
    return replace(c, areas=frozenset(wider)), "location", _area_names(c.areas, vocab), _area_names(wider, vocab)


def _relax_location_city(c: Constraints, vocab: Vocabulary):
    if c.areas is None:
        return None
    return replace(c, areas=None), "location", _area_names(c.areas, vocab), None


def _relax_cuisine(c: Constraints, vocab: Vocabulary):
    if c.cuisines is None:
        return None
    return replace(c, cuisines=None), "cuisines", _cuisine_names(c.cuisines, vocab), None


# §4.3 order. Rating first (a 0.3 delta is rarely a real difference), cuisine last (what users care
# about most). Location is one ladder position with two widenings. Order is pinned by a test (F-07).
LADDER: tuple[tuple[int, Step], ...] = (
    (1, _relax_rating),
    (2, _relax_budget),
    (3, _relax_location_nearby),
    (3, _relax_location_city),
    (4, _relax_cuisine),
)


def retrieve(catalog: pd.DataFrame, prefs: Preferences, vocab: Vocabulary, *, min_candidates: int) -> Retrieval:
    """Filter, then relax until `len(pool) >= min_candidates` (F-08) or the ladder runs out.

    The ladder is a fixed, finite sequence and every applied step strictly widens a constraint,
    so it always terminates (F-03).
    """
    c = Constraints.from_prefs(prefs, vocab)
    relaxations: list[Relaxation] = []
    count = int(combined_mask(catalog, c).sum())

    # §4.2 soft edge: a thin exact-budget match admits the band above, flagged as a stretch.
    if c.bands is not None and count < STRETCH_THRESHOLD:
        wider = _widen_bands(c.bands, up_only=True)
        if wider is not None:
            widened = replace(c, bands=wider)
            after = int(combined_mask(catalog, widened).sum())
            if after > count:
                relaxations.append(
                    _record("budget", list(c.bands), list(wider), 0, count, after, vocab)
                )
                c, count = widened, after

    for position, step in LADDER:
        if count >= min_candidates:
            break
        result = step(c, vocab)
        if result is None:
            continue
        widened, field, before_value, after_value = result
        after = int(combined_mask(catalog, widened).sum())
        relaxations.append(_record(field, before_value, after_value, position, count, after, vocab))
        c, count = widened, after

    mask = combined_mask(catalog, c)
    pool = catalog[mask]
    if relaxations:
        logger.info("relaxation applied", extra={"steps": [r.field for r in relaxations], "pool": len(pool)})
    return Retrieval(
        pool=pool,
        constraints=c,
        relaxations=relaxations,
        hidden_unrated=_hidden_unrated(catalog, c),
        blocking=blocking_constraints(catalog, c) if pool.empty else [],
    )


def _hidden_unrated(df: pd.DataFrame, c: Constraints) -> int:
    if c.min_rating is None:
        return 0
    return int((combined_mask(df, replace(c, min_rating=None)) & df["rating"].isna().to_numpy(dtype=bool)).sum())


def blocking_constraints(df: pd.DataFrame, c: Constraints) -> list[tuple[str, int]]:
    """Constraints whose removal alone would yield matches, most-restrictive first (F-02)."""
    masks = constraint_masks(df, c)
    out = []
    for name in masks:
        others = [m for n, m in masks.items() if n != name]
        count = int(np.logical_and.reduce(others).sum()) if others else len(df)
        if count:
            out.append((name, count))
    return sorted(out, key=lambda t: (-t[1], t[0]))


def _record(field: str, before, after, step: int, count_before: int, count_after: int, vocab: Vocabulary) -> Relaxation:
    return Relaxation(
        field=field,
        from_=before,
        to=after,
        step=step,
        matches_before=count_before,
        matches_after=count_after,
        reason=_reason(field, before, after, step, count_before),
    )


def _reason(field: str, before, after, step: int, count: int) -> str:
    only = "no restaurants" if count == 0 else f"only {count} restaurant{'s' if count != 1 else ''}"
    if field == "min_rating":
        return f"Rating filter relaxed from {before} to {after} — {only} met {before}."
    if field == "budget":
        added = [b for b in after if b not in before]
        label = "stretched to include" if step == 0 else "widened to include"
        return f"Budget {label} {', '.join(added)} — {only} matched {'/'.join(before)}."
    if field == "location" and after is not None:
        added = [a for a in after if a not in before]
        return f"Area widened from {_short(before)} to nearby {_short(added)} — {only} matched."
    if field == "location":
        return f"Area filter dropped to search all of Bengaluru — {only} matched in {_short(before)}."
    return f"Cuisine filter ({', '.join(before)}) dropped — {only} matched it with the other filters."


def _short(names: list[str], limit: int = 4) -> str:
    return ", ".join(names[:limit]) + (f" +{len(names) - limit} more" if len(names) > limit else "")


def _area_names(keys, vocab: Vocabulary) -> list[str]:
    return sorted((vocab.areas[k].display if k in vocab.areas else k for k in keys), key=str.casefold)


def _cuisine_names(keys, vocab: Vocabulary) -> list[str]:
    return sorted((vocab.cuisines[k].display if k in vocab.cuisines else k for k in keys), key=str.casefold)


def applied_filters(c: Constraints, vocab: Vocabulary) -> AppliedFilters:
    return AppliedFilters(
        location=_area_names(c.areas, vocab) if c.areas is not None else None,
        budget=list(c.bands) if c.bands is not None else None,
        cuisines=_cuisine_names(c.cuisines, vocab) if c.cuisines is not None else None,
        min_rating=c.min_rating,
        online_order=c.online_order,
        book_table=c.book_table,
    )


def describe_constraint(name: str, c: Constraints, vocab: Vocabulary) -> str:
    if name == "location":
        return f"area {_short(_area_names(c.areas or (), vocab))}"
    if name == "budget":
        return f"budget {'/'.join(c.bands or ())}"
    if name == "cuisines":
        return f"cuisine {', '.join(_cuisine_names(c.cuisines or (), vocab))}"
    if name == "min_rating":
        return f"rating ≥ {c.min_rating}"
    value = getattr(c, name)
    return f"{name.replace('_', ' ')} = {'yes' if value else 'no'}"
