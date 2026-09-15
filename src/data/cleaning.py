"""Per-field cleaning, deduplication and derived fields (architecture §3.2-§3.3, plan 1.2-1.4).

Every function is pure: Series in, new Series out (`deduplicate` takes a DataFrame).
Rules are shaped by profiling the raw dump (task 1.1); findings that went beyond the
architecture's table are noted inline as "profile:".
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping

import ftfy
import numpy as np
import pandas as pd

RAW_COLUMNS: tuple[str, ...] = (
    "url",
    "address",
    "name",
    "online_order",
    "book_table",
    "rate",
    "votes",
    "phone",
    "location",
    "rest_type",
    "dish_liked",
    "cuisines",
    "approx_cost(for two people)",
    "listed_in(type)",
    "listed_in(city)",
)
HEAVY_COLUMNS: tuple[str, ...] = ("reviews_list", "menu_item")

BUDGET_BANDS: tuple[str, ...] = ("low", "medium", "high")
BAYES_VOTE_PRIOR = 50

# Keys are casefolded. Extend these as the vocabulary review in the quality report finds pairs (D-16).
CUISINE_ALIASES: dict[str, str] = {
    "afghani": "Afghan",  # profile: 64 "Afghan" vs 8 "Afghani"
    "biriyani": "Biryani",
    "café": "Cafe",
    "hot dogs": "Hot Dogs",  # profile: the only non-title-cased cuisine
}
REST_TYPE_ALIASES: dict[str, str] = {
    "irani cafee": "Irani Cafe",  # profile: typo in the source
    "café": "Cafe",
}
AREA_ALIASES: dict[str, str] = {
    # profile: the source abbreviates these; users type the full name.
    "btm": "BTM Layout",
    "hsr": "HSR Layout",
}

_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[\W_]+")
# profile: half the rated rows carry a space before the slash ("3.9 /5").
_RATE = re.compile(r"^(\d+(?:\.\d+)?)\s*/\s*5$")
_NO_RATING_TOKENS = {"-", ""}


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def _memo_map(s: pd.Series, fn: Callable[[str], object]) -> list[object]:
    """Apply `fn` to each string cell once per distinct value; non-strings become None."""
    memo: dict[str, object] = {}
    out: list[object] = []
    for v in s:
        if not isinstance(v, str):
            out.append(None)
            continue
        if v not in memo:
            memo[v] = fn(v)
        out.append(memo[v])
    return out


def _fix(text: str) -> str:
    # profile: names/addresses are mojibake'd several layers deep ("CafÃÂÃÂ…©"); ftfy unwinds all of it.
    return _WHITESPACE.sub(" ", ftfy.fix_text(text)).strip()


def clean_display_text(s: pd.Series) -> pd.Series:
    """Encoding repair + whitespace collapse, case preserved. Blank → NA."""
    return pd.Series(_memo_map(s, lambda t: _fix(t) or None), index=s.index, dtype="str")


def normalize_text(s: pd.Series) -> pd.Series:
    """Matching form: `clean_display_text` + casefold."""
    return pd.Series(_memo_map(s, lambda t: _fix(t).casefold() or None), index=s.index, dtype="str")


def normalize_key(s: pd.Series) -> pd.Series:
    """Dedup-key form: `normalize_text` with punctuation removed ("No. 12," ≡ "no 12")."""
    return pd.Series(
        _memo_map(s, lambda t: _WHITESPACE.sub(" ", _NON_WORD.sub(" ", _fix(t).casefold())).strip() or None),
        index=s.index,
        dtype="str",
    )


def clean_area(s: pd.Series) -> pd.Series:
    """Area display name with aliases applied (`"BTM"` → `"BTM Layout"`)."""

    def fix(t: str) -> str | None:
        text = _fix(t)
        return AREA_ALIASES.get(text.casefold(), text) or None

    return pd.Series(_memo_map(s, fix), index=s.index, dtype="str")


def split_list_field(s: pd.Series, aliases: Mapping[str, str] | None = None) -> pd.Series:
    """`"A, B, "` → `["A", "B"]`: split, repair, drop empties (D-14), apply aliases (D-16),
    dedupe case-insensitively keeping first-seen order (D-15). Null → `[]`."""
    folded_aliases = {k.casefold(): v for k, v in (aliases or {}).items()}

    def split(text: str) -> tuple[str, ...]:
        items: dict[str, str] = {}
        for part in text.split(","):
            item = _fix(part)
            if not item:
                continue
            item = folded_aliases.get(item.casefold(), item)
            items.setdefault(item.casefold(), item)
        return tuple(items.values())

    parsed = _memo_map(s, split)
    return pd.Series([list(p) if p is not None else [] for p in parsed], index=s.index, dtype=object)


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def classify_rate(v: object) -> tuple[float | None, str]:
    """→ (rating, status); status is one of ok / new / no_rating / out_of_range / unparsed."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None, "no_rating"
    if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
        value = float(v)  # D-06: mixed-dtype column
    else:
        text = str(v).strip()  # D-05
        if text.casefold() == "new":
            return None, "new"
        if text in _NO_RATING_TOKENS:
            return None, "no_rating"
        match = _RATE.match(text)
        try:
            value = float(match.group(1)) if match else float(text)
        except ValueError:
            return None, "unparsed"
    if math.isnan(value):
        return None, "no_rating"
    if not 0 <= value <= 5:
        return None, "out_of_range"  # D-07: drop, never clamp into a fake rating
    return value, "ok"


def clean_rate(s: pd.Series) -> pd.Series:
    """`"4.1/5"` → 4.1; NEW / "-" / NaN / junk → NA. Never imputed (D-08)."""
    return pd.Series(_classify_all(s, classify_rate, 0), index=s.index, dtype="Float64")


def rate_status(s: pd.Series) -> pd.Series:
    return pd.Series(_classify_all(s, classify_rate, 1), index=s.index, dtype="str")


def is_new_rate(s: pd.Series) -> pd.Series:
    return rate_status(s).eq("new").astype(bool)


def classify_cost(v: object) -> tuple[int | None, str]:
    """→ (rupees, status); status is one of ok / null / unparsed."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None, "null"
    if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        value = int(v)
    elif isinstance(v, (float, np.floating)):
        if not float(v).is_integer():
            return None, "unparsed"
        value = int(v)
    else:
        text = str(v).strip().replace(",", "")  # D-09
        if text == "":
            return None, "null"
        if not text.isdigit():
            return None, "unparsed"
        value = int(text)
    return (value, "ok") if value > 0 else (None, "unparsed")


def clean_cost(s: pd.Series) -> pd.Series:
    """`"1,200"` → 1200 as nullable Int64 (D-09, D-10)."""
    return pd.Series(_classify_all(s, classify_cost, 0), index=s.index, dtype="Int64")


def cost_status(s: pd.Series) -> pd.Series:
    return pd.Series(_classify_all(s, classify_cost, 1), index=s.index, dtype="str")


def _classify_all(s: pd.Series, fn: Callable[[object], tuple], field: int) -> list:
    memo: dict[object, tuple] = {}
    out = []
    for v in s:
        key = v if isinstance(v, (str, int, float)) and not (isinstance(v, float) and math.isnan(v)) else None
        if key is None:
            out.append(fn(v)[field])
            continue
        if key not in memo:
            memo[key] = fn(v)
        out.append(memo[key][field])
    return out


def clean_yes_no(s: pd.Series) -> pd.Series:
    """`"Yes"`/`"No"` → nullable boolean; anything else → NA."""
    mapping = {"yes": True, "no": False}
    values = [mapping.get(v.strip().casefold()) if isinstance(v, str) else None for v in s]
    return pd.Series(values, index=s.index, dtype="boolean")


def clean_votes(s: pd.Series) -> pd.Series:
    """Integer vote count; missing or negative → 0 (votes weight confidence, they never filter)."""
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0).astype("int64")


# ---------------------------------------------------------------------------
# Deduplication (1.3)
# ---------------------------------------------------------------------------


def make_restaurant_id(key: pd.Series) -> pd.Series:
    """Stable content hash (D-23: hashlib, never the per-process-salted `hash()`); 12 hex chars (D-24)."""
    return pd.Series(
        ["r_" + hashlib.sha1(k.encode("utf-8")).hexdigest()[:12] for k in key],
        index=key.index,
        dtype="str",
    )


def dedup_key(name: pd.Series, address: pd.Series, url: pd.Series) -> pd.Series:
    """`name_key \\x1f address_key`. A blank address can't prove two rows are one place, so it
    falls back to the listing URL path — or the row label if that's missing too (D-21)."""
    name_key = normalize_key(name).fillna("")
    address_key = normalize_key(address)
    url_path = pd.Series(
        [u.split("?", 1)[0].strip() if isinstance(u, str) and u.strip() else None for u in url],
        index=url.index,
        dtype="str",
    )
    fallback = ("url:" + url_path).fillna(pd.Series([f"row:{i}" for i in url.index], index=url.index))
    return name_key + "\x1f" + address_key.fillna(fallback)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse listings of one restaurant (D-19).

    Input needs `name`, `address`, `url`, `votes`, `listed_type`, `listed_area`. The row with the
    most votes is kept whole (fields are never mixed across rows); ties go to the smallest URL
    so the choice is deterministic. `listed_type`/`listed_area` become sorted lists.
    """
    key = dedup_key(df["name"], df["address"], df["url"])
    work = df.assign(_key=key)
    lists = work.groupby("_key", sort=False).agg(
        listed_types=("listed_type", _sorted_distinct),
        listed_areas=("listed_area", _sorted_distinct),
    )
    canonical = work.sort_values(["votes", "url"], ascending=[False, True], na_position="last")
    canonical = canonical.drop_duplicates("_key").drop(columns=["listed_type", "listed_area"])
    out = canonical.join(lists, on="_key")
    out.insert(0, "restaurant_id", make_restaurant_id(out["_key"]))
    return out.drop(columns="_key").sort_values("restaurant_id").reset_index(drop=True)


def _sorted_distinct(s: pd.Series) -> list[str]:
    return sorted({v for v in s if isinstance(v, str) and v})


# ---------------------------------------------------------------------------
# Derived fields (1.4)
# ---------------------------------------------------------------------------


def budget_band_edges(cost: pd.Series) -> list[int]:
    """Inclusive upper rupee limit of each band but the top one.

    A tertile cut that respects ties: costs are heavily clustered (₹300, ₹400…), so plain
    quantile edges can put 45% of rows in one band. Instead choose the cut points, among the
    observed values, whose band shares are closest to a third each. Fewer than three distinct
    costs yield fewer bands instead of raising (D-12).
    """
    values = cost.dropna().astype("int64")
    counts = values.value_counts().sort_index()
    if len(counts) < 2:
        return []
    points = counts.index.to_numpy()
    if len(counts) == 2:
        return [int(points[0])]
    cum = counts.cumsum().to_numpy() / len(values)
    best: tuple[float, int, int] | None = None
    for i in range(len(points) - 2):
        for j in range(i + 1, len(points) - 1):
            shares = (cum[i], cum[j] - cum[i], 1 - cum[j])
            err = sum((x - 1 / 3) ** 2 for x in shares)
            if best is None or err < best[0] - 1e-12:
                best = (err, i, j)
    assert best is not None
    return [int(points[best[1]]), int(points[best[2]])]


def assign_budget_band(cost: pd.Series, edges: list[int]) -> pd.Series:
    """Null cost → NA band (D-10); such rows are only excluded when a budget is requested."""
    labels = {0: ("medium",), 1: ("low", "high"), 2: BUDGET_BANDS}[len(edges)]

    def band(v: object) -> str | None:
        if pd.isna(v):
            return None
        for edge, label in zip(edges, labels):
            if v <= edge:
                return label
        return labels[-1]

    return pd.Series([band(v) for v in cost], index=cost.index, dtype="str")


def describe_budget_bands(cost: pd.Series, band: pd.Series) -> list[dict[str, object]]:
    """Per band: rupee min/max, row count, share of costed rows — for the report and the gate."""
    costed = int(cost.notna().sum())
    rows = []
    for name in BUDGET_BANDS:
        in_band = cost[band.eq(name).fillna(False).astype(bool)].dropna()
        if in_band.empty:
            continue
        rows.append(
            {
                "band": name,
                "min": int(in_band.min()),
                "max": int(in_band.max()),
                "rows": len(in_band),
                "share": len(in_band) / costed,
            }
        )
    return rows


def rating_prior_mean(rating: pd.Series) -> float | None:
    rated = rating.dropna()
    return round(float(rated.mean()), 4) if len(rated) else None


def bayesian_rating(rating: pd.Series, votes: pd.Series, prior_votes: int = BAYES_VOTE_PRIOR) -> pd.Series:
    """`(v·R + m·C) / (v + m)`. Unrated rows stay NA — ranking decides how to treat them."""
    mean = rating_prior_mean(rating)
    if mean is None:
        return pd.Series(pd.NA, index=rating.index, dtype="Float64")
    r = rating.astype("Float64")
    v = votes.astype("Float64")
    return ((v * r + prior_votes * mean) / (v + prior_votes)).round(4)


def popularity_pct(votes: pd.Series) -> pd.Series:
    """Percentile rank of votes in (0, 1]; ties share the average rank."""
    return votes.rank(pct=True, method="average").round(4).astype("float64")


def text_blob(
    name: pd.Series, cuisines: pd.Series, rest_type: pd.Series, dish_liked: pd.Series, location: pd.Series
) -> pd.Series:
    """Casefolded `name cuisines rest_type dish_liked location` for keyword/embedding match."""
    blobs = []
    for n, cu, rt, dl, loc in zip(name, cuisines, rest_type, dish_liked, location):
        parts = [n if isinstance(n, str) else "", *cu, *rt, *dl, loc if isinstance(loc, str) else ""]
        blobs.append(" ".join(p for p in parts if p).casefold())
    return pd.Series(blobs, index=name.index, dtype="str")
