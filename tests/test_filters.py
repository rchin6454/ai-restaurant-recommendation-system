"""Phase 2 normalization, filter chain and relaxation ladder (plan 2.2-2.4, 2.8).

IDs (I-xx, F-xx) point into docs/edge-case.md §3-§4.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.core import filters as f
from src.core.models import Preferences
from src.data.catalog import VocabEntry, build_vocabulary
from tests.factories import make_catalog, restaurant


def vocab_of(rows):
    cat = make_catalog(rows)
    return cat, build_vocabulary(cat)


@pytest.fixture
def city():
    rows = [
        restaurant(1, location="Koramangala 5th Block", listed_areas=["Koramangala 5th Block"], votes=900),
        restaurant(2, location="Koramangala 6th Block", listed_areas=["Koramangala 5th Block"]),
        restaurant(3, location="BTM Layout", cuisines=["Chinese"], listed_areas=["BTM Layout"]),
        restaurant(4, location="Koramangala", listed_areas=["Koramangala 5th Block"]),
        restaurant(5, location="Indiranagar", cuisines=["Biryani"], budget_band="low", cost_for_two=200),
    ]
    return vocab_of(rows)


# --- Preferences validation (I-08, I-11..I-14) -------------------------------------------


def test_preferences_normalize_blank_and_duplicates():
    p = Preferences(location="  ", cuisines=["Chinese", "chinese ", ""], free_text="")
    assert p.location is None and p.free_text is None  # I-13
    assert p.cuisines == ["Chinese"]  # I-08


@pytest.mark.parametrize(
    "kwargs",
    [{"min_rating": 7}, {"min_rating": -1}, {"budget": "cheap"}, {"free_text": "x" * 501}, {"surprise": 1}],
)
def test_preferences_reject_invalid(kwargs):  # I-11, I-12, I-14
    with pytest.raises(ValidationError):
        Preferences(**kwargs)


# --- normalization (I-01..I-06) ----------------------------------------------------------------


def test_location_typo_is_resolved_and_reported(city):  # I-03
    _, vocab = city
    n = f.normalize(Preferences(location="koramangla 5th block"), vocab)
    assert n.prefs.location == "Koramangala 5th Block"
    assert n.interpretations[0].note == "interpreted 'koramangla 5th block' as 'Koramangala 5th Block'"


def test_exact_match_ignoring_case_is_not_reported(city):
    _, vocab = city
    n = f.normalize(Preferences(location="btm layout", cuisines=["chinese"]), vocab)
    assert (n.prefs.location, n.prefs.cuisines, n.interpretations) == ("BTM Layout", ["Chinese"], [])


def test_area_alias(city):
    _, vocab = city
    assert f.normalize(Preferences(location="BTM"), vocab).prefs.location == "BTM Layout"


def test_out_of_coverage_location_is_an_explicit_error(city):  # I-02, W-1
    _, vocab = city
    n = f.normalize(Preferences(location="Delhi", cuisines=["Chinese"]), vocab)
    assert n.coverage_error and "Bengaluru only" in n.coverage_error
    assert n.prefs.location == "Delhi"  # not silently dropped


def test_city_name_means_no_area_filter(city):
    _, vocab = city
    n = f.normalize(Preferences(location="Bengaluru"), vocab)
    assert n.prefs.location is None and n.coverage_error is None and n.interpretations


def test_near_miss_offers_suggestions():  # I-05
    entries = {"koramangala 5th block": VocabEntry("Koramangala 5th Block", 10, 10)}
    m = f.fuzzy_match("koramangala 5th", entries)
    assert m.key is None and m.suggestions == ("Koramangala 5th Block",)


def test_ambiguous_match_prefers_more_votes():  # I-04
    entries = {"abcdefghijx": VocabEntry("Abcdefghijx", 5, 10), "abcdefghijy": VocabEntry("Abcdefghijy", 5, 999)}
    m = f.fuzzy_match("abcdefghijz", entries)
    assert m.display == "Abcdefghijy" and m.score >= f.FUZZY_THRESHOLD


def test_cuisine_typo_alias_and_unknown(city):  # I-06
    _, vocab = city
    n = f.normalize(Preferences(cuisines=["north indain", "Biriyani", "Ethiopian"]), vocab)
    assert n.prefs.cuisines == ["North Indian", "Biryani"]
    assert n.unknown_cuisines == ("Ethiopian",)
    assert [i.matched for i in n.interpretations] == ["North Indian", "Biryani"]


# --- vocabulary --------------------------------------------------------------------------------


def test_area_family_and_nearby_from_zones(city):
    _, vocab = city
    assert vocab.area_families["koramangala"] == ("koramangala", "koramangala 5th block", "koramangala 6th block")
    assert vocab.area_families["btm layout"] == ("btm layout",)
    # 6th Block's restaurants are listed under the 5th Block zone.
    assert "koramangala 6th block" in vocab.nearby_areas["koramangala 5th block"]
    assert vocab.nearby_areas["indiranagar"] == ()


# --- masks (F-12, I-10, D-10) --------------------------------------------------------------------


def test_masks():
    cat = make_catalog(
        [
            restaurant(1, rating=None, online_order=False),
            restaurant(2, rating=0.0 + 3.0, cost_for_two=None, budget_band=None),
            restaurant(3, cuisines=["Chinese", "Thai"]),
        ]
    )
    everything = f.combined_mask(cat, f.Constraints())
    assert everything.all()  # F-12: nothing set → nothing narrowed; null cost/rating included (D-10)
    assert f.rating_mask(cat, 0).tolist() == [False, True, True]  # I-10: 0 still excludes unrated
    assert f.budget_mask(cat, ("medium",)).tolist() == [True, False, True]
    assert f.cuisine_mask(cat, frozenset({"thai", "sushi"})).tolist() == [False, False, True]  # OR
    assert f.flag_mask(cat, "online_order", True).tolist() == [False, True, True]
    assert "online_order" not in f.constraint_masks(cat, f.Constraints(book_table=True))


# --- relaxation ladder (F-01..F-10) ----------------------------------------------------------------


def ladder_catalog():
    """Nothing matches Koramangala 5th + low + Thai + 4.5 until every ladder step has fired."""
    rows = [
        # the only exact-area rows: medium budget, rated 4.3
        restaurant(1, location="Koramangala 5th Block", listed_areas=["Koramangala 5th Block"], cuisines=["Thai"], rating=4.3),
        # nearby (listed under the 5th Block zone) but not Thai
        restaurant(2, location="Koramangala 6th Block", listed_areas=["Koramangala 5th Block"], rating=4.3),
    ]
    rows += [restaurant(10 + i, location="Whitefield", rating=4.3, cuisines=["Pizza"]) for i in range(12)]
    return vocab_of(rows)


def test_ladder_order_is_rating_budget_location_cuisine():  # F-07
    cat, vocab = ladder_catalog()
    prefs = Preferences(location="Koramangala 5th Block", budget="low", cuisines=["Thai"], min_rating=4.5)
    r = f.retrieve(cat, prefs, vocab, min_candidates=10)
    ladder = [(x.step, x.field) for x in r.relaxations if x.step > 0]
    assert ladder == [(1, "min_rating"), (2, "budget"), (3, "location"), (3, "location"), (4, "cuisines")]
    assert r.relaxations[0].to == 4.2
    assert r.relaxations[1].from_ == ["low"] and r.relaxations[1].to == ["low", "medium"]
    assert r.relaxations[2].to == ["Koramangala 5th Block", "Koramangala 6th Block"]
    assert r.relaxations[3].to is None
    assert len(r.pool) == 14
    assert r.relaxations[0].model_dump(by_alias=True)["from"] == 4.5


def test_stretch_admits_one_band_up_before_the_ladder():  # §4.2
    cat, vocab = vocab_of(
        [restaurant(i, budget_band="low") for i in range(3)] + [restaurant(10 + i, budget_band="medium") for i in range(20)]
    )
    r = f.retrieve(cat, Preferences(budget="low"), vocab, min_candidates=10)
    assert [(x.step, x.field, x.to) for x in r.relaxations] == [(0, "budget", ["low", "medium"])]
    assert len(r.pool) == 23


def test_rating_at_floor_is_skipped():  # F-04
    cat, vocab = vocab_of([restaurant(1, rating=3.0), restaurant(2, rating=2.5, budget_band="low")])
    r = f.retrieve(cat, Preferences(budget="high", min_rating=3.0), vocab, min_candidates=10)
    assert r.relaxations[0].field == "budget"


def test_budget_high_widens_downward():  # F-05
    cat, vocab = vocab_of([restaurant(1, budget_band="high")] + [restaurant(i, budget_band="medium") for i in range(2, 30)])
    r = f.retrieve(cat, Preferences(budget="high"), vocab, min_candidates=10)
    assert [x.to for x in r.relaxations] == [["medium", "high"]]


def test_no_neighbours_goes_straight_to_whole_city():  # F-06
    cat, vocab = vocab_of([restaurant(1, location="Peenya")] + [restaurant(i, location="Whitefield") for i in range(2, 15)])
    r = f.retrieve(cat, Preferences(location="Peenya"), vocab, min_candidates=10)
    assert [(x.field, x.to) for x in r.relaxations] == [("location", None)]


@pytest.mark.parametrize(("matches", "relaxed"), [(9, True), (10, False), (11, False)])
def test_min_candidates_boundary(matches, relaxed):  # F-08
    rows = [restaurant(i, rating=4.6) for i in range(matches)] + [restaurant(100 + i, rating=4.3) for i in range(5)]
    cat, vocab = vocab_of(rows)
    r = f.retrieve(cat, Preferences(min_rating=4.5), vocab, min_candidates=10)
    assert bool(r.relaxations) is relaxed


def test_impossible_query_stops_and_names_blocker():  # F-02, F-03
    cat, vocab = vocab_of([restaurant(i, book_table=False) for i in range(20)])
    r = f.retrieve(cat, Preferences(book_table=True, cuisines=["North Indian"]), vocab, min_candidates=10)
    assert r.pool.empty
    assert len(r.relaxations) <= 5
    assert r.blocking[0] == ("book_table", 20)


def test_no_single_blocker_is_reported_as_a_combination():
    cat, vocab = vocab_of([restaurant(i, book_table=False, rating=4.0) for i in range(20)])
    r = f.retrieve(cat, Preferences(book_table=True, min_rating=4.9), vocab, min_candidates=10)
    assert r.pool.empty and r.blocking == []  # dropping either filter alone still leaves nothing


def test_hidden_unrated_count():  # F-10
    cat, vocab = vocab_of([restaurant(i) for i in range(12)] + [restaurant(50 + i, rating=None) for i in range(3)])
    r = f.retrieve(cat, Preferences(min_rating=3.5), vocab, min_candidates=10)
    assert r.hidden_unrated == 3


def test_applied_filters_reflect_final_constraints(city):
    cat, vocab = city
    n = f.normalize(Preferences(location="Koramangala", budget="medium"), vocab)
    r = f.retrieve(cat, n.prefs, vocab, min_candidates=1)
    applied = f.applied_filters(r.constraints, vocab)
    assert applied.location == ["Koramangala", "Koramangala 5th Block", "Koramangala 6th Block"]
    assert applied.budget == ["medium"] and applied.cuisines is None
    assert np.all(r.pool["location_norm"].str.startswith("koramangala"))
