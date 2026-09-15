"""Phase 2 pre-ranking, diversity trim, deterministic ranker, and end-to-end recommend (plan 2.5-2.6, 2.8).

IDs (R-xx, F-xx, I-xx) point into docs/edge-case.md.
"""

import pandas as pd
import pytest

from src.config import RankWeights, load_settings
from src.core.filters import Constraints
from src.core.models import Preferences
from src.core.ranking import deterministic_rank, diversity_trim, explain, pre_rank, score_pool
from src.core.recommender import recommend
from tests.factories import make_catalog, restaurant

WEIGHTS = RankWeights()


def rank(cat, requested=Constraints(), k=25, cap=2):
    return pre_rank(cat, requested, weights=WEIGHTS, k=k, max_chain_outlets=cap, rating_prior=3.6)


# --- scoring -------------------------------------------------------------------------------------


def test_identical_scores_tiebreak_by_votes_then_id():  # R-01, R-02
    cat = make_catalog([restaurant(3), restaurant(1), restaurant(2, votes=100), restaurant(4, votes=100)])
    first = rank(cat)["restaurant_id"].tolist()
    shuffled = rank(cat.sample(frac=1, random_state=7))["restaurant_id"].tolist()
    assert first == shuffled == ["r_000000000001", "r_000000000002", "r_000000000003", "r_000000000004"]
    assert rank(cat)["score"].notna().all()


def test_unrated_rows_score_on_the_prior():  # R-04
    cat = make_catalog([restaurant(1, rating=4.5), restaurant(2, rating=None), restaurant(3, rating=3.0)])
    scored = score_pool(cat, Constraints(), WEIGHTS, rating_prior=3.6)
    assert scored["score"].notna().all()
    assert scored["restaurant_id"].tolist() == ["r_000000000001", "r_000000000002", "r_000000000003"]


def test_unused_weights_are_redistributed():  # R-03
    cat = make_catalog([restaurant(1, rating=4.5, votes=10), restaurant(2, rating=3.5, votes=1000)])
    no_prefs = score_pool(cat, Constraints(), WEIGHTS, 3.6).set_index("restaurant_id")["score"]
    # Only rating (0.45) and votes (0.20) apply → normalized to 0.692 / 0.308; each row maxes one term.
    assert no_prefs["r_000000000001"] == pytest.approx(0.45 / 0.65)
    assert no_prefs["r_000000000002"] == pytest.approx(0.20 / 0.65)


def test_rows_meeting_the_request_rank_before_relaxed_rows():  # M-04
    cat = make_catalog(
        [
            restaurant(1, budget_band="high", cost_for_two=1500, rating=4.9, votes=9000),  # stretch row
            restaurant(2, budget_band="medium", rating=3.5, votes=5),
        ]
    )
    out = rank(cat, Constraints(bands=("medium",)))
    assert out["restaurant_id"].tolist() == ["r_000000000002", "r_000000000001"]
    assert out["meets_request"].tolist() == [True, False]


def test_cuisine_overlap_and_budget_fit():
    cat = make_catalog(
        [
            restaurant(1, cuisines=["Thai", "Chinese"], budget_band="medium"),
            restaurant(2, cuisines=["Thai"], budget_band="high"),
            restaurant(3, cuisines=["Pizza"], budget_band="low", cost_for_two=None),
        ]
    )
    requested = Constraints(cuisines=frozenset({"thai", "chinese"}), bands=("medium",))
    s = score_pool(cat, requested, WEIGHTS, 3.6).set_index("restaurant_id")
    assert s.loc["r_000000000001", ["cuisine_overlap", "budget_fit"]].tolist() == [1.0, 1.0]
    assert s.loc["r_000000000002", ["cuisine_overlap", "budget_fit"]].tolist() == [0.5, 0.5]
    assert s.loc["r_000000000003", ["cuisine_overlap", "budget_fit"]].tolist() == [0.0, 0.5]
    assert s.index[0] == "r_000000000001"


# --- diversity ---------------------------------------------------------------------------------


def test_chain_cap_backfills_from_other_restaurants():  # F-11, R-05
    chain = [restaurant(i, name="Domino's Pizza", votes=5000 - i) for i in range(1, 7)]
    others = [restaurant(100 + i, votes=10 + i) for i in range(4)]
    out = rank(make_catalog(chain + others), k=5)
    assert len(out) == 5
    assert (out["name_norm"] == "domino's pizza").sum() == 2


def test_chain_detection_is_exact_name():  # R-06
    cat = make_catalog([restaurant(1, name="Domino's Pizza"), restaurant(2, name="Dominos Pizza"), restaurant(3, name="Domino's Pizza")])
    assert len(diversity_trim(cat, k=10, max_chain_outlets=1)) == 2


def test_fewer_candidates_than_k_are_not_padded():  # R-08
    assert len(rank(make_catalog([restaurant(1), restaurant(2)]), k=25)) == 2


# --- deterministic ranker (2.6) -----------------------------------------------------------------


def test_template_explanation_matches_architecture():
    row = make_catalog([restaurant(1, rating=4.3, votes=512, cost_for_two=800, budget_band="medium")]).iloc[0]
    text, highlights = explain(row, Constraints(bands=("medium",), cuisines=frozenset({"north indian"})))
    assert text == "4.3★ from 512 votes · North Indian · ₹800 for two — matches your budget and cuisine."
    assert highlights == ["within budget", "North Indian"]


def test_explanation_null_states_and_relaxed_misses():  # U-02, U-03
    row = make_catalog([restaurant(1, rating=None, is_new=True, cost_for_two=None, budget_band=None, location="Whitefield")]).iloc[0]
    text, _ = explain(row, Constraints(areas=frozenset({"indiranagar"}), cuisines=frozenset({"thai"})))
    assert text.startswith("New — not yet rated · North Indian · cost not listed — Filters were relaxed:")
    assert "outside the area you chose" in text and "doesn't list the cuisine" in text
    assert "0.0" not in text and "nan" not in text


def test_deterministic_rank_top_n_and_free_text_caveat():  # I-18
    cands = rank(make_catalog([restaurant(i) for i in range(3)]))
    out = deterministic_rank(cands, Constraints(), top_n=5, pool_size=3, free_text="family-friendly")
    assert [p.rank for p in out.picks] == [1, 2, 3]
    assert out.caveats and "free-text" in out.caveats[0]


# --- end to end ------------------------------------------------------------------------------------


# No key: these tests pin the deterministic path, even if GROQ_API_KEY is set in the shell.
CONFIG = load_settings(_env_file=None, groq_api_key=None, min_candidates=3, llm_candidate_k=6)


@pytest.fixture
def cat():
    rows = [restaurant(i, location="Koramangala 5th Block", listed_areas=["Koramangala 5th Block"],
                       name="Chai Point" if i < 4 else f"Place {i}", votes=1000 - i) for i in range(10)]
    rows += [restaurant(20 + i, location="Whitefield", cuisines=["Thai"], budget_band="high", cost_for_two=1500) for i in range(5)]
    return make_catalog(rows)


def test_recommend_results_are_grounded_and_diverse(cat):
    r = recommend(Preferences(location="koramangala 5th block"), catalog=cat, config=CONFIG)
    assert r.outcome == "results" and r.degraded
    ids = [x.restaurant_id for x in r.recommendations]
    assert len(ids) == 5 and len(set(ids)) == 5
    assert sum(x.name == "Chai Point" for x in r.recommendations) <= 2  # M-06
    rows = cat.set_index("restaurant_id")
    for x in r.recommendations:  # every displayed fact is the catalog row's
        assert (x.name, x.votes, x.area) == (rows.loc[x.restaurant_id, "name"], rows.loc[x.restaurant_id, "votes"], "Koramangala 5th Block")
    assert r.applied_filters.location == ["Koramangala 5th Block"]
    assert r.candidates_considered == 6


def test_recommend_empty_preferences(cat):  # I-01
    r = recommend(Preferences(), catalog=cat, config=CONFIG)
    assert r.outcome == "results" and len(r.recommendations) == 5
    assert r.applied_filters.model_dump(exclude_none=True) == {}


def test_recommend_coverage_error_never_returns_results(cat):  # I-02, W-1
    r = recommend(Preferences(location="Delhi", cuisines=["Italian"], budget="low", min_rating=4.5), catalog=cat, config=CONFIG)
    assert r.outcome == "coverage_error" and r.recommendations == []
    assert "Bengaluru only" in r.summary


def test_recommend_relaxation_is_disclosed(cat):  # F-09
    r = recommend(Preferences(location="Whitefield", min_rating=4.1), catalog=cat, config=CONFIG)
    assert r.outcome == "relaxed_results"
    assert r.relaxations[0].field == "min_rating"
    assert r.caveats[0] == r.relaxations[0].reason


def test_recommend_unknown_cuisine_and_impossible_query(cat):  # I-06, F-02
    unknown = recommend(Preferences(cuisines=["Ethiopian"]), catalog=cat, config=CONFIG)
    assert unknown.outcome == "empty_with_reason" and "'Ethiopian'" in unknown.summary
    impossible = recommend(Preferences(book_table=True), catalog=cat, config=CONFIG)
    assert impossible.outcome == "empty_with_reason" and impossible.recommendations == []
    assert "Blocking constraint: book table = yes" in impossible.summary


def test_recommend_budget_stretch_is_flagged():
    rows = [restaurant(i, budget_band="low", cost_for_two=200) for i in range(2)]
    rows += [restaurant(10 + i, budget_band="medium", votes=5000) for i in range(3)]
    r = recommend(Preferences(budget="low"), catalog=make_catalog(rows), config=CONFIG)
    assert r.relaxations[0].step == 0
    assert {x.stretch for x in r.recommendations if x.budget_band == "medium"} == {True}
    assert {x.stretch for x in r.recommendations if x.budget_band == "low"} == {False}
    assert r.recommendations[0].budget_band == "low"  # exact-band fit outranks more votes
