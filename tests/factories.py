"""Small hand-built catalogs for retrieval/ranking tests. Values are chosen, not derived from quantiles."""

from __future__ import annotations

import pandas as pd

from src.data.catalog import CATALOG_COLUMNS


def restaurant(i: int, **overrides) -> dict:
    location = overrides.get("location", "Indiranagar")
    row = {
        "restaurant_id": f"r_{i:012x}",
        "name": f"Restaurant {i}",
        "address": f"{i} Main Road",
        "url": f"https://example.com/{i}",
        "location": location,
        "listed_areas": [location] if location else [],
        "rating": 4.0,
        "is_new": False,
        "votes": 100,
        "cost_for_two": 500,
        "budget_band": "medium",
        "cuisines": ["North Indian"],
        "rest_type": ["Casual Dining"],
        "dish_liked": [],
        "online_order": True,
        "book_table": False,
        "listed_types": ["Delivery"],
    }
    row.update(overrides)
    return row


def make_catalog(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["name_norm"] = df["name"].str.casefold()
    df["location_norm"] = [v.casefold() if isinstance(v, str) else None for v in df["location"]]
    df["listed_areas_norm"] = [[a.casefold() for a in v] for v in df["listed_areas"]]
    df["cuisines_norm"] = [[c.casefold() for c in v] for v in df["cuisines"]]
    df["rating"] = df["rating"].astype("Float64")
    df["is_unrated"] = df["rating"].isna()
    df["cost_for_two"] = df["cost_for_two"].astype("Int64")
    df["budget_band"] = df["budget_band"].astype("str")
    df["online_order"] = df["online_order"].astype("boolean")
    df["book_table"] = df["book_table"].astype("boolean")
    df["votes"] = df["votes"].astype("int64")
    df["bayesian_rating"] = df["rating"]
    df["popularity_pct"] = df["votes"].rank(pct=True)
    df["text_blob"] = df["name"].str.casefold()
    return df[list(CATALOG_COLUMNS)].reset_index(drop=True)
