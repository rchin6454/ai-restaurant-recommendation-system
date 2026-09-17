"""Eval code graders (evals/metrics.py; docs/eval.md §4.1-§4.2) and the query labels file."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import pytest

from evals.labels import CATEGORY_COUNTS, Expect, Predicates, Query, RelaxationExpect, Signals, load_queries
from evals.metrics import aggregate, audit_explanation, grade_judge, grade_query, meets_any_signal, meets_predicates, scorecard
from src.config import load_settings
from src.core.models import Preferences, Relaxation
from src.core.recommender import recommend, shortlist
from tests.factories import make_catalog, restaurant

NO_KEY = load_settings(_env_file=None, groq_api_key=None, min_candidates=3)


@pytest.fixture
def catalog():
    rows = [restaurant(i, votes=1000 - i, cuisines=["North Indian"] if i < 8 else ["Chinese"]) for i in range(12)]
    rows.append(restaurant(20, name="Chain Co", votes=5000))
    rows.append(restaurant(21, name="Chain Co", votes=4900, address="21 Other Road"))
    rows.append(restaurant(22, name="Chain Co", votes=4800, address="22 Third Road"))
    return make_catalog(rows)


def query(prefs: Preferences, **expect) -> Query:
    expect.setdefault("outcome", "results")
    expect.setdefault("relaxation", RelaxationExpect(expected=False))
    return Query(id="q-1", category="ordinary", prefs=prefs, expect=Expect(**expect), notes="")


def grade(q: Query, response, cat, **kwargs):
    listed = shortlist(q.prefs, catalog=cat, config=NO_KEY)
    return grade_query(
        q, response, cat.set_index("restaurant_id", drop=False),
        candidate_ids=set(listed.candidates["restaurant_id"]), requested=listed.requested,
        max_chain_outlets=2, band_edges={250, 469}, mode="deterministic", **kwargs,
    )


def test_a_real_deterministic_response_passes_every_blocking_check(catalog):
    q = query(Preferences(cuisines=["North Indian"]), min_picks=5, acceptable=Predicates(cuisines_any=["North Indian"]))
    response = recommend(q.prefs, catalog=catalog, config=NO_KEY)
    checks = grade(q, response, catalog)
    assert all(checks[m]["passed"] for m in ("M-01", "M-03", "M-05", "M-06", "M-07", "M-12")), checks
    assert checks["M-09"]["num"] == 0  # template explanations quote only catalog numbers
    assert checks["M-10"]["num"] == checks["M-10"]["den"] == 5


def test_grounding_catches_a_changed_fact_and_a_pick_outside_the_candidate_set(catalog):
    q = query(Preferences(cuisines=["Chinese"]))
    response = recommend(q.prefs, catalog=catalog, config=NO_KEY)
    cards = response.recommendations
    cards[0] = cards[0].model_copy(update={"name": "Hotel Fictional"})
    cards[1] = cards[1].model_copy(update={"restaurant_id": "r_000000000000"})  # a real North Indian row, not shortlisted
    checks = grade(q, response, catalog)
    assert checks["M-01"]["num"] == 2 and checks["M-01"]["passed"] is False
    assert "name" in checks["M-01"]["detail"] and "not in the candidate set" in checks["M-01"]["detail"]


def test_first_ladder_step_ignores_the_budget_stretch(catalog):
    q = query(Preferences(), outcome="results", relaxation=RelaxationExpect(expected=True, first_field="min_rating"))
    response = recommend(q.prefs, catalog=catalog, config=NO_KEY)
    response.relaxations = [
        Relaxation(field="budget", from_=["low"], to=["low", "medium"], step=0, matches_before=1, matches_after=4, reason="stretch"),
        Relaxation(field="min_rating", from_=4.5, to=4.2, step=1, matches_before=4, matches_after=9, reason="rating"),
    ]
    assert grade(q, response, catalog)["M-07"]["passed"]
    q.expect.relaxation.first_field = "budget"
    assert "first ladder step min_rating" in grade(q, response, catalog)["M-07"]["detail"]


def test_undisclosed_constraint_break_fails_relaxation_disclosure(catalog):
    q = query(Preferences(cuisines=["Chinese"]))
    response = recommend(Preferences(), catalog=catalog, config=NO_KEY)  # North Indian picks, no relaxation recorded
    checks = grade(q, response, catalog)
    assert checks["M-05"]["passed"] is False and "breaks cuisines" in checks["M-05"]["detail"]
    response.relaxations = [Relaxation(field="cuisines", from_=["Chinese"], to=None, step=4, matches_before=4,
                                       matches_after=15, reason="dropped")]
    assert grade(q, response, catalog)["M-05"]["passed"]


def test_diversity_catches_a_third_chain_outlet(catalog):
    q = query(Preferences())
    response = recommend(q.prefs, catalog=catalog, config=NO_KEY)
    assert grade(q, response, catalog)["M-06"]["passed"]
    third = response.recommendations[0].model_copy(update={"restaurant_id": "r_000000000016", "name": "Chain Co"})
    response.recommendations.append(third)
    assert "3 outlets of chain co" in grade(q, response, catalog)["M-06"]["detail"]


def test_forbidden_strings_are_searched_in_every_field_but_the_trace(catalog):
    q = query(Preferences(), forbidden_strings=["Hotel Fictional"])
    response = recommend(q.prefs, catalog=catalog, config=NO_KEY)
    response.trace.fallback_reason = "Hotel Fictional"  # operational metadata, never shown
    assert grade(q, response, catalog)["M-08"]["passed"]
    response.caveats.append("We skipped hotel fictional.")
    assert grade(q, response, catalog)["M-08"]["passed"] is False


@pytest.mark.parametrize(
    ("text", "problems"),
    [
        ("Rated 4.3★ from 512 votes, ₹800 for two.", 0),
        ("A 4.3 rating from 500+ votes and ₹800 for two.", 0),
        ("A 4.6 rating and ₹1,200 for two.", 2),
        ("Well within your budget, under ₹469 for two.", 0),  # a band boundary, not a price claim
        ("It clears your 4.0 minimum.", 0),
        ("Over 900 votes.", 1),
    ],
)
def test_fact_audit(text, problems):
    row = pd.Series({"rating": 4.3, "cost_for_two": 800, "votes": 512})
    assert len(audit_explanation(text, row, ignore_ratings={4.0}, band_edges={250, 469})) == problems


@pytest.mark.parametrize(
    ("text", "problems"),
    [("over 2000 votes", 0), ("more than 2,000 votes", 0), ("about 2,100 votes", 0), ("~2000 votes", 0),
     ("over 3000 votes", 1), ("about 1,500 votes", 1), ("2000 votes", 1)],
)
def test_fact_audit_accepts_truthful_vote_qualifiers(text, problems):  # found in the first live run
    row = pd.Series({"rating": 4.5, "cost_for_two": 600, "votes": 2073})
    assert len(audit_explanation(text, row, ignore_ratings=set(), band_edges=set())) == problems


def test_zero_invalid_ids_is_not_a_failure_and_no_broken_picks_means_full_disclosure():
    checks = {"M-02": {"passed": None, "num": 0, "den": 5, "detail": ""}, "M-05": {"passed": True, "num": 0, "den": 0, "detail": ""}}
    agg = aggregate([record("a", checks), record("b", {"M-02": {"passed": None, "num": 1, "den": 5, "detail": "r_x"}})])
    assert agg["M-02"]["failing"] == ["b"]
    assert agg["M-05"]["value"] == 1.0 and agg["M-05"]["passed"] is True


def test_fact_audit_flags_any_rating_on_an_unrated_restaurant():
    row = pd.Series({"rating": pd.NA, "cost_for_two": pd.NA, "votes": 0})
    assert audit_explanation("A solid 4.1★ spot.", row, ignore_ratings=set(), band_edges=set())


def test_area_predicates_cover_blocks_and_signals_match_any(catalog):
    row = make_catalog([restaurant(1, location="Koramangala 5th Block", rest_type=["Pub"], book_table=False)]).iloc[0]
    assert meets_predicates(row, Predicates(area_in=["Koramangala"]))
    assert not meets_predicates(row, Predicates(area_in=["Koramangala"], book_table=True))
    assert meets_any_signal(row, Signals(rest_type_any=["Casual Dining", "Pub"], book_table=True))


def record(qid: str, checks: dict, category: str = "ordinary", repeat: int = 0, pick_ids=()) -> dict:
    return {"type": "query", "query_id": qid, "category": category, "repeat": repeat, "checks": checks,
            "retrieval_ms": 12.0, "pick_ids": list(pick_ids)}


def test_aggregate_scores_blocking_metrics_and_run_validity():
    ok = {"passed": True, "num": 1, "den": 1, "detail": ""}
    degraded = {"passed": False, "num": 1, "den": 1, "detail": "rate limit"}
    records = [
        record("a", {"M-01": {"passed": True, "num": 0, "den": 5, "detail": ""}, "M-07": ok, "M-19": ok | {"num": 0, "passed": True}},
               pick_ids=["x", "y"]),
        record("b", {"M-01": {"passed": False, "num": 1, "den": 5, "detail": "bad"}, "M-07": ok, "M-19": degraded}),
        record("a", {}, repeat=1, pick_ids=["x", "z"]),
        record("c", {"M-17": {"passed": None, "num": 1, "den": 1, "detail": "win"}}, category="free_text"),
        record("d", {"M-17": {"passed": None, "num": 0, "den": 1, "detail": "loss"}}),
    ]
    agg = aggregate(records)
    assert agg["M-01"]["value"] == 1 and agg["M-01"]["passed"] is False and agg["M-01"]["failing"] == ["b"]
    assert agg["M-19"]["value"] == 0.5 and agg["M-19"]["passed"] is False
    assert agg["M-17"]["value"] == 0.5 and agg["M-17"]["free_text"] == 1.0
    assert agg["M-18"]["value"] == pytest.approx(1 / 3)
    assert "**FAIL**" in scorecard(agg) and "not measured" in scorecard(agg)


def test_judge_grading_flags_ungrounded_picks_and_unacknowledged_conflicts():
    q = Query(id="con-1", category="contradictory", prefs=Preferences(), notes="",
              expect=Expect(outcome="results", relaxation=RelaxationExpect(expected=False)))
    judge = {"picks": [{"id": "a", "grounded": 5, "preference_specific": 4, "concise": 5},
                       {"id": "b", "grounded": 2, "preference_specific": 3, "concise": 4}],
             "conflict_acknowledged": False}
    checks = grade_judge(q, judge)
    assert checks["M-15"]["num"] / checks["M-15"]["den"] == pytest.approx(23 / 6) and checks["M-15"]["low_grounded"] == 1
    assert checks["M-16"]["passed"] is False


def test_query_set_matches_the_eval_plan():  # docs/eval.md §3.1-§3.2
    queries = load_queries()
    assert dict(Counter(q.category for q in queries)) == CATEGORY_COUNTS
    assert sum(bool(q.expect.gold_ids) for q in queries) >= 15
    for q in queries:
        assert q.notes, q.id
        if q.expect.outcome == "coverage_error":
            assert q.expect.min_picks == 0
        if q.category in ("contradictory", "thin"):
            assert q.expect.caveat_required, q.id
        if q.category == "adversarial":
            assert q.expect.forbidden_strings or q.expect.forbidden_ids, q.id
