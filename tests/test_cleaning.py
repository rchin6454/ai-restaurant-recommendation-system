"""Phase 1 cleaning rules, one case per junk value found while profiling the raw dump (plan 1.8).

IDs (D-xx) point into docs/edge-case.md §2.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.data import cleaning as c
from src.data.catalog import CATALOG_COLUMNS
from src.data.ingest import IngestError, build_catalog

# Verbatim from the raw dump: "Café Down The Alley" mojibake'd several layers deep.
MOJIBAKE_CAFE = (
    "CafÃÂÃÂÃÂÃ"
    "ÂÃÂÃÂÃÂÃÂ©"
    " Down The Alley"
)


def s(*values, dtype=object) -> pd.Series:
    return pd.Series(list(values), dtype=dtype)


def na(v) -> bool:
    return v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v))


# --- rate (D-01..D-08) ------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "status"),
    [
        ("4.1/5", 4.1, "ok"),  # D-01
        ("3.9 /5", 3.9, "ok"),  # D-05: ~half the dataset has this space
        (" 4.1/5 ", 4.1, "ok"),  # D-05
        ("4.1 / 5", 4.1, "ok"),
        ("NEW", None, "new"),  # D-02
        ("-", None, "no_rating"),  # D-03
        (None, None, "no_rating"),  # D-04
        (np.nan, None, "no_rating"),  # D-04
        ("", None, "no_rating"),  # D-04
        (4.1, 4.1, "ok"),  # D-06: already a float
        (3, 3.0, "ok"),  # D-06
        ("6.2/5", None, "out_of_range"),  # D-07
        ("-1", None, "out_of_range"),  # D-07
        ("abc", None, "unparsed"),
    ],
)
def test_classify_rate(raw, expected, status):
    value, got_status = c.classify_rate(raw)
    assert got_status == status
    assert value == expected if expected is not None else value is None


def test_clean_rate_mixed_dtype_column_and_flags():  # D-06
    raw = s("4.1/5", 3.5, "NEW", "-", None, "2.0 /5")
    rating = c.clean_rate(raw)
    assert str(rating.dtype) == "Float64"
    assert rating.tolist()[:2] == [4.1, 3.5] and rating.tolist()[5] == 2.0
    assert rating.isna().tolist() == [False, False, True, True, True, False]
    assert c.is_new_rate(raw).tolist() == [False, False, True, False, False, False]


def test_unrated_rows_are_never_imputed():  # D-08
    raw = s("4.5/5", "NEW", "-", None, "3.0/5")
    assert int(c.clean_rate(raw).isna().sum()) == 3


# --- cost (D-09, D-10) -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected", "status"),
    [
        ("1,200", 1200, "ok"),  # D-09
        ("800", 800, "ok"),
        (" 1,050 ", 1050, "ok"),
        (500, 500, "ok"),
        (500.0, 500, "ok"),
        (None, None, "null"),  # D-10
        (np.nan, None, "null"),
        ("", None, "null"),
        ("abc", None, "unparsed"),
        ("0", None, "unparsed"),
        (99.5, None, "unparsed"),
    ],
)
def test_classify_cost(raw, expected, status):
    assert c.classify_cost(raw) == (expected, status)


def test_clean_cost_is_nullable_int():
    cost = c.clean_cost(s("1,200", None, "300"))
    assert str(cost.dtype) == "Int64"
    assert cost.iloc[0] == 1200 and cost.iloc[2] == 300 and cost.isna().iloc[1]


# --- list fields (D-14..D-16) -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("North Indian, Mughlai, Chinese", ["North Indian", "Mughlai", "Chinese"]),
        ("North Indian, Chinese, ", ["North Indian", "Chinese"]),  # D-14
        ("Chinese, Chinese", ["Chinese"]),  # D-15
        ("Afghani, Afghan", ["Afghan"]),  # D-16
        ("Hot dogs, Burger", ["Hot Dogs", "Burger"]),  # casing drift
        (None, []),
        ("", []),
        (" , ,", []),
    ],
)
def test_split_list_field_cuisines(raw, expected):
    assert c.split_list_field(s(raw), c.CUISINE_ALIASES).iloc[0] == expected


def test_split_list_field_rest_type_typo_and_null():  # D-18
    out = c.split_list_field(s("Irani Cafee", None, "Casual Dining, Bar"), c.REST_TYPE_ALIASES)
    assert out.tolist() == [["Irani Cafe"], [], ["Casual Dining", "Bar"]]


def test_split_list_rows_do_not_share_list_objects():
    out = c.split_list_field(s("A, B", "A, B"))
    out.iloc[0].append("C")
    assert out.iloc[1] == ["A", "B"]


# --- text ------------------------------------------------------------------------


def test_mojibake_is_repaired():
    assert c.clean_display_text(s(MOJIBAKE_CAFE)).iloc[0] == "Café Down The Alley"
    assert c.normalize_text(s(MOJIBAKE_CAFE)).iloc[0] == "café down the alley"


def test_whitespace_and_blank_text():
    out = c.clean_display_text(s("  Jalsa \t Cafe ", "   ", None))
    assert out.iloc[0] == "Jalsa Cafe"
    assert out.isna().tolist() == [False, True, True]


def test_normalize_key_ignores_punctuation():
    keys = c.normalize_key(s("No. 12, 5th 'A' Block", "no 12 5th A block"))
    assert keys.iloc[0] == keys.iloc[1] == "no 12 5th a block"


def test_area_aliases_keep_display_form():
    assert c.clean_area(s("BTM", "HSR", "Koramangala 5th Block", None)).tolist()[:3] == [
        "BTM Layout",
        "HSR Layout",
        "Koramangala 5th Block",
    ]


def test_yes_no():
    assert c.clean_yes_no(s("Yes", "No", " yes", "maybe", None)).tolist() == [True, False, True, pd.NA, pd.NA]


# --- deduplication (D-19..D-24) ------------------------------------------------


def listings(rows: list[dict]) -> pd.DataFrame:
    base = {"url": None, "votes": 0, "listed_type": "Delivery", "listed_area": "BTM Layout", "rating": 4.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_dedup_collapses_listings_and_keeps_max_votes():  # D-19, D-20
    df = listings(
        [
            {"name": "Jalsa", "address": "942, 21st Main Road", "url": "u1", "votes": 10, "rating": 3.9,
             "listed_type": "Delivery", "listed_area": "Banashankari"},
            {"name": "JALSA ", "address": "942 21st main road", "url": "u2", "votes": 775, "rating": 4.1,
             "listed_type": "Buffet", "listed_area": "Jayanagar"},
            {"name": "Jalsa", "address": "942, 21st Main Road", "url": "u3", "votes": 775, "rating": 4.0,
             "listed_type": "Dine-out", "listed_area": "Banashankari"},
            {"name": "Jalsa", "address": "942, 21st Main Road", "url": "u4", "votes": 5,
             "listed_type": "Delivery", "listed_area": "Banashankari"},
        ]
    )
    out = c.deduplicate(df)
    assert len(out) == 1  # >50% reduction
    row = out.iloc[0]
    assert row["url"] == "u2" and row["rating"] == 4.1  # max votes; tie broken by smallest URL
    assert row["listed_types"] == ["Buffet", "Delivery", "Dine-out"]
    assert row["listed_areas"] == ["Banashankari", "Jayanagar"]
    assert "listed_type" not in out.columns


def test_dedup_keeps_chain_outlets_at_different_addresses():  # D-22
    df = listings(
        [
            {"name": "Third Wave Coffee", "address": "Koramangala 5th Block", "url": "a"},
            {"name": "Third Wave Coffee", "address": "Indiranagar 100 Feet Road", "url": "b"},
        ]
    )
    assert len(c.deduplicate(df)) == 2


def test_dedup_does_not_merge_blank_addresses():  # D-21
    df = listings(
        [
            {"name": "Chai Point", "address": None, "url": "https://z.com/bangalore/chai-point-1?ctx=a"},
            {"name": "Chai Point", "address": " ", "url": "https://z.com/bangalore/chai-point-2?ctx=a"},
            {"name": "Chai Point", "address": None, "url": None},
            {"name": "Chai Point", "address": None, "url": None},
            # Same listing path, different query context → the same restaurant.
            {"name": "Chai Point", "address": None, "url": "https://z.com/bangalore/chai-point-1?ctx=b"},
        ]
    )
    assert len(c.deduplicate(df)) == 4


def test_restaurant_id_is_a_stable_content_hash():  # D-23, D-24
    df = listings([{"name": "Jalsa", "address": "942, 21st Main Road, 2nd Stage, Banashankari, Bangalore", "url": "u"}])
    # Pinned literal: if this changes, every eval gold label breaks (eval.md §3.4 rule 5).
    assert c.deduplicate(df).iloc[0]["restaurant_id"] == "r_225fa219c0d4"


def test_dedup_output_independent_of_input_order():
    df = listings(
        [
            {"name": f"R{i % 7}", "address": f"Addr {i % 7}", "url": f"u{i}", "votes": i, "listed_type": t}
            for i, t in zip(range(40), ["Delivery", "Dine-out", "Cafes", "Buffet"] * 10)
        ]
    )
    a = c.deduplicate(df)
    b = c.deduplicate(df.sample(frac=1, random_state=3).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 7


# --- derived fields ------------------------------------------------------------


def test_budget_band_edges_balance_ties():  # D-11
    cost = pd.Series([300] * 40 + [400] * 25 + [500] * 5 + [800] * 20 + [3000] * 10, dtype="Int64")
    edges = c.budget_band_edges(cost)
    bands = c.assign_budget_band(cost, edges)
    shares = bands.value_counts(normalize=True)
    assert edges == [300, 500]
    assert shares["low"] == 0.40 and shares["medium"] == 0.30 and shares["high"] == 0.30


def test_budget_bands_monotonic_and_null_cost():  # D-10
    cost = pd.Series(list(range(100, 1000, 100)) + [None], dtype="Int64")
    edges = c.budget_band_edges(cost)
    bands = c.assign_budget_band(cost, edges)
    assert edges == [300, 600]
    assert bands.tolist()[:9] == ["low"] * 3 + ["medium"] * 3 + ["high"] * 3
    assert na(bands.iloc[9])
    described = c.describe_budget_bands(cost, bands)
    assert [b["band"] for b in described] == ["low", "medium", "high"]
    assert all(lo["max"] < hi["min"] for lo, hi in zip(described, described[1:]))


@pytest.mark.parametrize(
    ("values", "edges", "labels"),
    [
        ([400] * 5, [], {"medium"}),  # D-12: constant column must not raise
        ([200, 200, 900], [200], {"low", "high"}),
        ([None, None], [], set()),
    ],
)
def test_budget_bands_degenerate_distributions(values, edges, labels):
    cost = pd.Series(values, dtype="Int64")
    assert c.budget_band_edges(cost) == edges
    assert set(c.assign_budget_band(cost, edges).dropna()) == labels


def test_bayesian_rating():
    rating = pd.Series([4.0, 2.0, None, 3.0], dtype="Float64")  # mean C = 3.0
    votes = pd.Series([0, 1_000_000, 10, 50], dtype="int64")
    out = c.bayesian_rating(rating, votes, prior_votes=50)
    assert out.iloc[0] == 3.0  # no votes → the prior
    assert abs(out.iloc[1] - 2.0) < 0.001  # huge votes → its own rating
    assert out.isna().iloc[2]  # unrated stays unrated
    assert out.iloc[3] == 3.0


def test_popularity_pct_and_text_blob():
    pct = c.popularity_pct(pd.Series([0, 10, 10, 500]))
    assert pct.iloc[3] == 1.0 and pct.iloc[1] == pct.iloc[2] and pct.iloc[0] < pct.iloc[1]
    blob = c.text_blob(s("Jalsa"), s(["North Indian"]), s(["Casual Dining"]), s([]), s(None))
    assert blob.iloc[0] == "jalsa north indian casual dining"


# --- end to end ----------------------------------------------------------------


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    base = {
        "url": None, "address": "1 MG Road, Bangalore", "name": "X", "online_order": "Yes", "book_table": "No",
        "rate": "4.0/5", "votes": 10, "phone": "080 1234 5678", "location": "MG Road", "rest_type": "Cafe",
        "dish_liked": None, "cuisines": "Cafe", "approx_cost(for two people)": "400",
        "listed_in(type)": "Delivery", "listed_in(city)": "MG Road",
    }
    return pd.DataFrame([{**base, "url": f"https://z/{i}", **r} for i, r in enumerate(rows)])


def test_build_catalog_end_to_end():
    raw = raw_frame(
        [
            {"name": "Jalsa", "address": "942, 21st Main", "rate": "4.1/5", "votes": 775, "listed_in(type)": "Buffet"},
            {"name": "Jalsa", "address": "942 21st Main", "rate": "4.1 /5", "votes": 775, "listed_in(type)": "Delivery",
             "listed_in(city)": "BTM"},
            {"name": "Spice Elephant", "address": "2 Church St", "rate": "NEW", "votes": 0,
             "approx_cost(for two people)": "1,200", "cuisines": "Chinese, Thai, "},
            {"name": "Hole", "address": "3 Brigade Rd", "rate": "-", "approx_cost(for two people)": None},
            {"name": "   ", "address": "4 Nowhere"},  # D-25
        ]
    )
    cat, stats = build_catalog(raw, min_rows=1)
    assert list(cat.columns) == list(CATALOG_COLUMNS)
    assert "phone" not in cat.columns  # S-06
    assert (stats.raw_rows, stats.dropped_empty_names, stats.deduped_rows) == (5, 1, 3)
    assert stats.unparsed_rate_values == [] and stats.unparsed_cost_values == []
    assert cat["restaurant_id"].is_unique

    by_name = cat.set_index("name")
    assert by_name.loc["Jalsa", "listed_areas"] == ["BTM Layout", "MG Road"]
    assert by_name.loc["Jalsa", "listed_types"] == ["Buffet", "Delivery"]
    assert by_name.loc["Spice Elephant", "cuisines_norm"] == ["chinese", "thai"]
    assert bool(by_name.loc["Spice Elephant", "is_new"]) and bool(by_name.loc["Spice Elephant", "is_unrated"])
    assert na(by_name.loc["Spice Elephant", "rating"]) and na(by_name.loc["Hole", "rating"])  # D-08
    assert na(by_name.loc["Hole", "budget_band"])  # D-10
    assert not (cat["is_unrated"] & cat["rating"].notna()).any()
    assert "chinese" in by_name.loc["Spice Elephant", "text_blob"]


def test_build_catalog_reports_unparsed_values():
    raw = raw_frame([{"name": "A", "rate": "4.1/5"}, {"name": "B", "rate": "7/5", "approx_cost(for two people)": "free"}])
    _, stats = build_catalog(raw, min_rows=1)
    assert stats.unparsed_rate_values == ["7/5"]
    assert stats.unparsed_cost_values == ["free"]


def test_build_catalog_rejects_missing_column():  # D-29
    with pytest.raises(IngestError, match="approx_cost"):
        build_catalog(raw_frame([{"name": "A"}]).drop(columns="approx_cost(for two people)"), min_rows=1)


def test_build_catalog_fails_loudly_when_too_few_rows():  # D-31
    with pytest.raises(IngestError, match="survived"):
        build_catalog(raw_frame([{"name": "A"}]), min_rows=2)
