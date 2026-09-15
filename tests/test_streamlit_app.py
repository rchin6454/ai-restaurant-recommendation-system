"""UI states from edge-case §9, run headlessly with Streamlit's AppTest against a fake API (no server)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
META = {
    "/meta/locations": ["Indiranagar", "Koramangala"],
    "/meta/cuisines": ["Chinese", "North Indian"],
    "/meta/budgets": [
        {"band": "low", "min_cost": 40, "max_cost": 250, "restaurants": 4000},
        {"band": "medium", "min_cost": 300, "max_cost": 469, "restaurants": 4000},
        {"band": "high", "min_cost": 500, "max_cost": 6000, "restaurants": 4000},
    ],
}


def card(rank: int, **overrides) -> dict:
    return {
        "rank": rank, "restaurant_id": f"r_{rank:012x}", "name": f"Restaurant {rank}", "cuisines": ["North Indian"],
        "rating": 4.3, "votes": 512, "cost_for_two": 800, "budget_band": "high", "area": "Koramangala",
        "url": f"https://www.zomato.com/bangalore/r{rank}", "explanation": f"Model prose for pick {rank}.",
        "match_highlights": ["well-reviewed"], **overrides,
    }


def response(**overrides) -> dict:
    return {
        "outcome": "results", "recommendations": [card(1)], "summary": "Picked for you.", "degraded": False,
        "candidates_considered": 25, "latency_ms": 3100,
        "trace": {"ranker": "llm", "model": "openai/gpt-oss-120b"}, **overrides,
    }


class FakeApi:
    """Stands in for `httpx.request`. POSTs get the queued answers in order; the last one repeats."""

    def __init__(self, *answers) -> None:
        self.answers = answers
        self.posts: list[dict] = []

    def __call__(self, method: str, url: str, **kwargs):
        path = httpx.URL(url).path
        if method == "GET":
            return httpx.Response(200, json=META[path])
        self.posts.append(kwargs["json"])
        answer = self.answers[min(len(self.posts), len(self.answers)) - 1]
        status, body = answer if isinstance(answer, tuple) else (200, answer)
        return httpx.Response(status, json=body)


@pytest.fixture(autouse=True)
def _fresh_meta_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def start(monkeypatch, api) -> AppTest:
    monkeypatch.setattr(httpx, "request", api)
    at = AppTest.from_file(APP, default_timeout=15).run()
    assert not at.exception
    return at


def submit(at: AppTest) -> AppTest:
    at.button(key="submit").click().run()
    assert not at.exception
    return at


def page_html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_api_unreachable_shows_a_friendly_error(monkeypatch):  # U-01
    def refuse(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    at = start(monkeypatch, refuse)
    assert "Can't reach the recommendation service" in at.error[0].value
    assert "Traceback" not in page_html(at)


def test_form_labels_area_for_bengaluru_and_shows_rupee_bands(monkeypatch):  # U-12, §8
    at = start(monkeypatch, FakeApi(response()))
    assert at.selectbox(key="area").label == "Area (Bengaluru)"
    assert at.radio(key="budget").options == ["Any budget", "Low · ₹40–250", "Medium · ₹300–469", "High · ₹500–6,000"]


def test_cards_show_null_states_and_keep_ai_prose_apart(monkeypatch):  # U-02, U-03, U-04, U-11
    unrated = card(2, name="<script>alert(1)</script>", rating=None, votes=0, cost_for_two=None, url=None,
                   budget_band=None, explanation="New place.\n\nNo reviews yet.")
    api = FakeApi(response(recommendations=[card(1), unrated]))
    at = start(monkeypatch, api)
    at.selectbox(key="area").set_value("Koramangala")
    at.multiselect(key="cuisines").select("Chinese")
    at.slider(key="min_rating").set_value(4.0)
    submit(at)

    assert api.posts == [{"location": "Koramangala", "cuisines": ["Chinese"], "min_rating": 4.0}]
    html = page_html(at)
    assert "New — not yet rated" in html and "★ 4.3 · 512 votes" in html
    assert "Cost unavailable" in html and "₹800 for two" in html
    assert "0.0" not in html and "₹0" not in html and "nan" not in html.lower()
    assert html.count("View on Zomato") == 1
    assert html.count("✦ AI explanation") == 2 and "Why it matches" not in html
    assert "&lt;script&gt;" in html and "<script>" not in html
    assert at.session_state["area"] == "Koramangala"  # U-09: the form survives the rerun


def test_degraded_banner_and_visible_relaxations(monkeypatch):  # U-06, U-10
    reason = "Rating filter relaxed from 4.5 to 4.2 — only 3 restaurants met 4.5."
    api = FakeApi(response(
        outcome="relaxed_results", degraded=True, caveats=[reason, "2 new or unrated restaurants hidden by the rating filter."],
        relaxations=[{"field": "min_rating", "from": 4.5, "to": 4.2, "step": 1, "matches_before": 3,
                      "matches_after": 14, "reason": reason}],
        trace={"ranker": "deterministic", "fallback_reason": "GROQ_API_KEY is not set"},
    ))
    at = submit(start(monkeypatch, api))

    assert any("AI ranking isn't available" in i.value for i in at.info)
    assert "Rating filter relaxed from 4.5 to 4.2" in at.warning[0].value
    assert any("hidden by the rating filter" in c.value for c in at.caption)
    html = page_html(at)
    assert "Why it matches" in html and "AI explanation" not in html


def test_backfilled_cards_are_not_labelled_as_ai(monkeypatch):
    api = FakeApi(response(recommendations=[card(1), card(2), card(3)],
                           trace={"ranker": "llm", "model": "openai/gpt-oss-120b", "backfilled": 1}))
    html = page_html(submit(start(monkeypatch, api)))
    assert html.count("✦ AI explanation") == 2 and html.count("Why it matches") == 1


def test_no_results_offers_one_click_relaxation(monkeypatch):  # U-05
    empty = response(outcome="empty_with_reason", recommendations=[], trace=None, blocking_constraints=["book_table"],
                     summary="No restaurants match. Blocking constraint: book table = yes — without it there would be 36 matches.")
    api = FakeApi(empty, response())
    at = start(monkeypatch, api)
    at.radio(key="book_table").set_value("Yes")
    submit(at)

    assert api.posts[0] == {"book_table": True}
    assert "Blocking constraint: book table = yes" in at.warning[0].value
    at.button(key="relax_book_table").click().run()
    assert api.posts[1] == {}
    assert at.session_state["book_table"] == "Any"
    assert "Restaurant 1" in page_html(at)


def test_api_error_envelope_message_is_shown(monkeypatch):
    envelope = {"error": {"code": "invalid_request", "message": "The request is invalid. See `details`.",
                          "details": [{"field": "free_text", "message": "String should have at most 500 characters"}],
                          "request_id": "abc"}}
    at = submit(start(monkeypatch, FakeApi((422, envelope))))
    assert "free_text: String should have at most 500 characters" in at.error[0].value
