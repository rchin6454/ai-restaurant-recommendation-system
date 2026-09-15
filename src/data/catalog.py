"""Cached catalog loader and vocabularies (architecture §3.4, plan task 1.7).

The catalog is read once per process. A re-ingest while a long-running process is
alive is NOT picked up — restart it, or call `clear_cache()` (D-30).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.config import get_logger, settings
from src.data.cleaning import rating_prior_mean

logger = get_logger(__name__)

# The contract every downstream layer relies on. `ingest.py` writes exactly these, in this order.
CATALOG_COLUMNS: tuple[str, ...] = (
    "restaurant_id",
    "name",
    "name_norm",
    "address",
    "url",
    "location",  # the restaurant's own area (display form)
    "location_norm",
    # `listed_in(city)` is the area a *listing* was filed under, and differs between listings of
    # one restaurant — so after dedup it's a list, not a scalar `city`.
    "listed_areas",
    "listed_areas_norm",
    "rating",
    "is_new",
    "is_unrated",
    "votes",
    "cost_for_two",
    "budget_band",
    "cuisines",
    "cuisines_norm",
    "rest_type",
    "dish_liked",
    "online_order",
    "book_table",
    "listed_types",
    "bayesian_rating",
    "popularity_pct",
    "text_blob",
)
LIST_COLUMNS: frozenset[str] = frozenset(
    {"listed_areas", "listed_areas_norm", "cuisines", "cuisines_norm", "rest_type", "dish_liked", "listed_types"}
)


class CatalogError(RuntimeError):
    """The catalog is missing or doesn't match the expected schema."""


def load_catalog(path: Path) -> pd.DataFrame:
    """Read and schema-check a catalog Parquet. Uncached — use `get_catalog()` in app code."""
    if not path.exists():
        raise CatalogError(
            f"Catalog not found at {path}. Build it with `python -m src.data.ingest`, "
            "or point CATALOG_PATH at an existing file."
        )
    df = pd.read_parquet(path)
    missing = [c for c in CATALOG_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogError(f"Catalog at {path} is missing columns {missing}; re-run the ingest.")
    # Parquet hands list columns back as numpy arrays; give callers the list[str] they were written as.
    for col in LIST_COLUMNS:
        df[col] = pd.Series([list(v) for v in df[col]], index=df.index, dtype=object)
    logger.info("catalog loaded", extra={"path": str(path), "rows": len(df)})
    return df


@lru_cache(maxsize=1)
def get_catalog() -> pd.DataFrame:
    """The process-wide catalog. Treat it as read-only: callers share one DataFrame."""
    return load_catalog(settings.catalog_path)


@lru_cache(maxsize=1)
def get_area_vocabulary() -> list[str]:
    """Sorted distinct display area names — UI dropdown and fuzzy-match target."""
    return _distinct(get_catalog()["location"])


@lru_cache(maxsize=1)
def get_cuisine_vocabulary() -> list[str]:
    """Sorted distinct display cuisine names, flattened from the per-row lists."""
    return _distinct(get_catalog()["cuisines"].explode())


@lru_cache(maxsize=1)
def get_vocabulary() -> Vocabulary:
    return build_vocabulary(get_catalog())


def clear_cache() -> None:
    for fn in (get_catalog, get_area_vocabulary, get_cuisine_vocabulary, get_vocabulary):
        fn.cache_clear()


# ---------------------------------------------------------------------------
# Matching vocabulary (used by preference normalization and the relaxation ladder)
# ---------------------------------------------------------------------------

# A zone (a `listed_areas` value) counts as serving an area once this share of the area's
# restaurants are listed under it.
NEARBY_ZONE_SHARE = 0.10


@dataclass(frozen=True)
class VocabEntry:
    display: str
    rows: int
    votes: int  # tie-break for equally good fuzzy matches (I-04)


@dataclass(frozen=True)
class Vocabulary:
    areas: dict[str, VocabEntry]  # key: casefolded display name == `location_norm`
    cuisines: dict[str, VocabEntry]  # key: casefolded display name == `cuisines_norm` element
    # "koramangala" → itself plus "koramangala 1st block" …; "whitefield" → "itpl main road, whitefield" …
    area_families: dict[str, tuple[str, ...]]
    # Areas sharing Zomato delivery zones, excluding the family. Derived, not hand-maintained:
    # an area's neighbours are the areas whose main zone is one that serves it.
    nearby_areas: dict[str, tuple[str, ...]]
    rating_prior: float  # corpus mean rating C, the score unrated rows fall back to (R-04)


def build_vocabulary(df: pd.DataFrame) -> Vocabulary:
    areas = _entries(df["location"], df["votes"])
    exploded = df[["cuisines", "votes"]].explode("cuisines")
    cuisines = _entries(exploded["cuisines"], exploded["votes"])

    keys = sorted(areas)
    families = {
        k: tuple(y for y in keys if y == k or y.startswith(k + " ") or y.endswith(", " + k)) for k in keys
    }

    listings = df[["location_norm", "listed_areas_norm"]].explode("listed_areas_norm").dropna()
    shares = listings.groupby("location_norm")["listed_areas_norm"].value_counts(normalize=True)
    serving: dict[str, set[str]] = {}
    main_zone: dict[str, str] = {}
    for (area, zone), share in sorted(shares.items(), key=lambda kv: (kv[0][0], -kv[1], kv[0][1])):
        main_zone.setdefault(area, zone)
        if share >= NEARBY_ZONE_SHARE:
            serving.setdefault(area, set()).add(zone)
    nearby = {
        k: tuple(y for y in keys if main_zone.get(y) in serving.get(k, set()) and y not in families[k])
        for k in keys
    }
    return Vocabulary(
        areas=areas,
        cuisines=cuisines,
        area_families=families,
        nearby_areas=nearby,
        rating_prior=rating_prior_mean(df["rating"]) or 0.0,
    )


def _entries(display: pd.Series, votes: pd.Series) -> dict[str, VocabEntry]:
    frame = pd.DataFrame({"display": display.to_numpy(), "votes": votes.to_numpy()})
    frame = frame[[isinstance(v, str) and v.strip() != "" for v in frame["display"]]]
    frame["key"] = [v.casefold() for v in frame["display"]]
    out: dict[str, VocabEntry] = {}
    for key, group in frame.groupby("key", sort=True):
        out[key] = VocabEntry(
            display=group["display"].value_counts().index[0],
            rows=len(group),
            votes=int(group["votes"].sum()),
        )
    return out


def _distinct(values: pd.Series) -> list[str]:
    cleaned = {v.strip() for v in values if isinstance(v, str)}
    return sorted(cleaned - {""}, key=str.casefold)
