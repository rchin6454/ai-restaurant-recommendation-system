"""Streamlit UI (architecture §8; plan 4.5-4.7).

    uvicorn src.api.main:app              # terminal 1
    streamlit run app/streamlit_app.py    # terminal 2

The UI only talks to the API over HTTP (`API_URL`, default http://localhost:8000). It never imports
the recommender, so the browser shows exactly what the API returned. On every card the catalog
facts and the ranker's prose are styled apart (U-11).
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any

import httpx
import streamlit as st
from pydantic import ValidationError

from src.config import settings
from src.core.models import MAX_FREE_TEXT_CHARS, Recommendation, RecommendationResponse

ANY = "Any"
YES_NO = {"Yes": True, "No": False}
# Groq's 30 s timeout plus the SDK's retries can outlast a short client timeout; don't give up first.
TIMEOUT = httpx.Timeout(120.0, connect=5.0)

# Form widget keys and their "no preference" values. Widgets read and write these in session state,
# so a rerun never clears the form (U-09).
DEFAULTS: dict[str, Any] = {
    "area": ANY,
    "budget": ANY,
    "cuisines": [],
    "min_rating": 0.0,
    "online_order": ANY,
    "book_table": ANY,
    "free_text": "",
}
# Response constraint name → (widget key, wording for the one-click relaxation button, U-05)
CONSTRAINT_WIDGETS = {
    "location": ("area", "the area filter"),
    "budget": ("budget", "the budget filter"),
    "cuisines": ("cuisines", "the cuisine filter"),
    "min_rating": ("min_rating", "the minimum rating"),
    "online_order": ("online_order", "the online-ordering filter"),
    "book_table": ("book_table", "the table-booking filter"),
}
FILTER_LABELS = {
    "location": "Area",
    "budget": "Budget",
    "cuisines": "Cuisines",
    "min_rating": "Minimum rating",
    "online_order": "Online ordering",
    "book_table": "Table booking",
}

CSS = """<style>
.rr-card{border:1px solid rgba(128,128,128,.28);border-radius:12px;padding:14px 18px 16px;margin:0 0 14px}
.rr-head{display:flex;align-items:center;gap:10px;min-width:0}
.rr-rank{flex:none;width:28px;height:28px;border-radius:50%;background:rgba(128,128,128,.18);display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem}
.rr-name{font-size:1.15rem;font-weight:650;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
.rr-area{opacity:.7;font-size:.88rem;margin:2px 0 10px 38px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rr-chips,.rr-tags{display:flex;flex-wrap:wrap;gap:6px}
.rr-chip{border:1px solid rgba(128,128,128,.35);border-radius:999px;padding:0 10px;font-size:.8rem;line-height:1.6}
.rr-facts{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:.95rem;margin:10px 0 12px}
.rr-muted{opacity:.65;font-style:italic}
.rr-stretch{color:#b7791f}
.rr-explain{border-left:3px solid #8b5cf6;background:rgba(139,92,246,.09);border-radius:0 8px 8px 0;padding:9px 12px}
.rr-explain.rr-template{border-left-color:rgba(128,128,128,.55);background:rgba(128,128,128,.08)}
.rr-label{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;font-weight:700;opacity:.7;margin-bottom:3px}
.rr-tags{margin-top:8px}
.rr-tag{font-size:.75rem;background:rgba(139,92,246,.16);border-radius:6px;padding:0 8px;line-height:1.7}
.rr-template .rr-tag{background:rgba(128,128,128,.16)}
.rr-link{display:inline-block;margin-top:10px;font-size:.9rem}
</style>"""


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """A failure already phrased for the user."""


def api_call(method: str, path: str, **kwargs: Any) -> Any:
    try:
        resp = httpx.request(method, f"{settings.api_url.rstrip('/')}{path}", timeout=TIMEOUT, **kwargs)
    except httpx.TimeoutException:
        raise ApiError("The recommendation service took too long to answer. Please try again.") from None
    except httpx.TransportError:  # U-01
        raise ApiError(
            f"Can't reach the recommendation service at {settings.api_url}. "
            "Start it with `uvicorn src.api.main:app`, then reload this page."
        ) from None
    if resp.is_success:
        return resp.json()
    try:
        error = resp.json()["error"]
        details = "; ".join(f"{d['field']}: {d['message']}" for d in error.get("details", []))
        message = error["message"] + (f" ({details})" if details else "")
    except (ValueError, KeyError, TypeError):
        message = f"The recommendation service returned an error (HTTP {resp.status_code})."
    raise ApiError(message)


@st.cache_data(ttl=600, show_spinner="Loading the restaurant catalog…")
def load_meta() -> dict[str, Any]:
    return {
        "locations": api_call("GET", "/meta/locations"),
        "cuisines": api_call("GET", "/meta/cuisines"),
        "budgets": {b["band"]: b for b in api_call("GET", "/meta/budgets")},
    }


def fetch_recommendations(body: dict[str, Any]) -> RecommendationResponse:
    data = api_call("POST", "/recommend", json=body)
    try:
        return RecommendationResponse.model_validate(data)
    except ValidationError:
        raise ApiError("The recommendation service sent a response this page doesn't recognise. "
                       "Check that the API and UI are the same version.") from None


def request_body(state: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "location": None if state["area"] == ANY else state["area"],
        "budget": None if state["budget"] == ANY else state["budget"],
        "cuisines": list(state["cuisines"]),
        "min_rating": round(float(state["min_rating"]), 1) or None,
        "online_order": YES_NO.get(state["online_order"]),
        "book_table": YES_NO.get(state["book_table"]),
        "free_text": state["free_text"].strip() or None,
    }
    return {k: v for k, v in body.items() if v not in (None, [])}


# ---------------------------------------------------------------------------
# Formatting — every string that reaches the page from the API is escaped
# ---------------------------------------------------------------------------

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>$~])")


def md(text: str) -> str:
    """Show API text literally inside a markdown element: no emphasis, links, HTML or LaTeX."""
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def esc(text: str) -> str:
    # Collapsing whitespace also keeps a card's HTML free of blank lines, which would end the HTML block.
    return html.escape(" ".join(str(text).split()))


def rating_text(rec: Recommendation) -> str:
    if rec.rating is None:  # U-02: never 0.0★
        return "New — not yet rated"
    return f"★ {rec.rating:.1f} · {rec.votes:,} vote{'s' if rec.votes != 1 else ''}"


def cost_text(rec: Recommendation) -> str:
    return "Cost unavailable" if rec.cost_for_two is None else f"₹{rec.cost_for_two:,} for two"  # U-03


def budget_label(band: str, budgets: Mapping[str, Mapping[str, Any]]) -> str:
    if band == ANY:
        return "Any budget"
    b = budgets.get(band)
    return f"{band.capitalize()} · ₹{b['min_cost']:,}–{b['max_cost']:,}" if b else band.capitalize()


def filter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.1f} ★ and up"
    if isinstance(value, list):
        return ", ".join(value[:6]) + (f" +{len(value) - 6} more" if len(value) > 6 else "")
    return str(value)


def card_html(rec: Recommendation, *, ai_explanation: bool) -> str:
    chips = "".join(f'<span class="rr-chip">{esc(c)}</span>' for c in rec.cuisines) or (
        '<span class="rr-chip rr-muted">Cuisine not listed</span>'
    )
    facts = [
        f'<span class="{"rr-muted" if rec.rating is None else ""}">{esc(rating_text(rec))}</span>',
        f'<span class="{"rr-muted" if rec.cost_for_two is None else ""}">{esc(cost_text(rec))}</span>',
    ]
    if rec.stretch:
        facts.append('<span class="rr-stretch">One step above your budget</span>')
    label, box = ("✦ AI explanation", "rr-explain") if ai_explanation else ("Why it matches", "rr-explain rr-template")
    tags = "".join(f'<span class="rr-tag">{esc(h)}</span>' for h in rec.match_highlights)
    url = rec.url if rec.url and rec.url.startswith(("https://", "http://")) else None
    link = (  # U-04: no url, no link
        f'<a class="rr-link" href="{esc(url)}" target="_blank" rel="noopener noreferrer">View on Zomato ↗</a>'
        if url else ""
    )
    return "".join([
        '<div class="rr-card">',
        f'<div class="rr-head"><span class="rr-rank">{rec.rank}</span>',
        f'<span class="rr-name" title="{esc(rec.name)}">{esc(rec.name)}</span></div>',
        f'<div class="rr-area">{esc(rec.area or "Area not listed")}</div>',
        f'<div class="rr-chips">{chips}</div>',
        f'<div class="rr-facts">{"".join(facts)}</div>',
        f'<div class="{box}"><div class="rr-label">{label}</div><div>{esc(rec.explanation)}</div>',
        f'<div class="rr-tags">{tags}</div>' if tags else "",
        "</div>",
        link,
        "</div>",
    ])


# ---------------------------------------------------------------------------
# Callbacks — run before the script re-renders, so they may change widget state
# ---------------------------------------------------------------------------


def start_search() -> None:
    st.session_state.in_flight = True


def relax_and_search(constraints: list[str]) -> None:
    for name in constraints:
        key = CONSTRAINT_WIDGETS[name][0]
        st.session_state[key] = list(DEFAULTS[key]) if isinstance(DEFAULTS[key], list) else DEFAULTS[key]
    st.session_state.in_flight = True


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------


def render_sidebar(meta: dict[str, Any]) -> None:
    budgets = meta["budgets"]
    with st.sidebar:
        st.header("Your preferences")
        with st.form("preferences", border=False):
            st.selectbox(  # U-12: labelled for the dataset's real coverage, and a dropdown, not free text
                "Area (Bengaluru)",
                [ANY, *meta["locations"]],
                key="area",
                format_func=lambda v: "Anywhere in Bengaluru" if v == ANY else v,
                help="The catalog covers Bengaluru only. Start typing to search.",
            )
            st.radio("Budget for two", [ANY, "low", "medium", "high"], key="budget",
                     format_func=lambda b: budget_label(b, budgets))
            st.multiselect("Cuisines", meta["cuisines"], key="cuisines", placeholder="Any cuisine",
                           help="Shows restaurants serving any of the cuisines you pick.")
            st.slider("Minimum rating", 0.0, 5.0, step=0.1, format="%.1f ★", key="min_rating",
                      help="0 means no minimum. Any minimum also hides new restaurants that aren't rated yet.")
            with st.expander("More filters"):
                st.radio("Online ordering", [ANY, *YES_NO], key="online_order", horizontal=True)
                st.radio("Table booking", [ANY, *YES_NO], key="book_table", horizontal=True)
            st.text_area(
                "Anything else?",
                key="free_text",
                max_chars=MAX_FREE_TEXT_CHARS,
                placeholder="e.g. family-friendly and quiet, or a rooftop for a date night",
                help="The AI weighs this when choosing and ordering picks. It isn't a filter.",
            )
            st.form_submit_button(  # U-07: disabled while a request is in flight
                "Find restaurants", key="submit", type="primary", width="stretch",
                on_click=start_search, disabled=st.session_state.in_flight,
            )


def run_search() -> None:
    body = request_body(st.session_state)
    try:
        with st.spinner("Finding and ranking restaurants — this usually takes a few seconds…"):
            st.session_state.result, st.session_state.error = fetch_recommendations(body), None
    except ApiError as exc:
        st.session_state.result, st.session_state.error = None, str(exc)
    finally:
        st.session_state.in_flight = False
    st.rerun()  # redraw with the submit button enabled again


def render_response(r: RecommendationResponse) -> None:
    if not r.recommendations:
        render_empty(r)
        return
    if r.degraded:  # U-06
        st.info("AI ranking isn't available right now, so these picks are ordered by rating, popularity and fit, "
                "with standard descriptions instead of AI explanations.", icon="ℹ️")
    st.markdown(f"**{md(r.summary)}**")
    render_caveats(r)

    # Backfilled cards (the ranker's invalid picks replaced from pre-ranked order) carry template
    # explanations and always come last, so only the first N cards hold the model's prose.
    backfilled = r.trace.backfilled if r.trace else 0
    ai_cards = 0 if r.degraded else len(r.recommendations) - backfilled
    for rec in r.recommendations:
        st.markdown(card_html(rec, ai_explanation=rec.rank <= ai_cards), unsafe_allow_html=True)
    render_transparency(r)


def render_caveats(r: RecommendationResponse) -> None:
    reasons = {x.reason for x in r.relaxations}
    if r.relaxations:  # U-10: visible on the page, not only inside the expander
        lines = "\n".join(f"- {md(x.reason)}" for x in r.relaxations)
        st.warning(f"**Your search was widened to find enough matches:**\n\n{lines}", icon="↔️")
    for caveat in r.caveats:
        if caveat not in reasons:
            st.caption(md(caveat))


def render_empty(r: RecommendationResponse) -> None:
    st.warning(md(r.summary), icon="🔍")
    for caveat in r.caveats:
        st.caption(md(caveat))
    if r.suggestions:
        st.caption("Closest matches in the catalog: " + md(", ".join(r.suggestions)))
    if r.outcome == "empty_with_reason":  # U-05: one click to drop what blocked the search
        blocking = [n for n in r.blocking_constraints if n in CONSTRAINT_WIDGETS]
        for i, name in enumerate(blocking):
            st.button(f"Search again without {CONSTRAINT_WIDGETS[name][1]}", key=f"relax_{name}",
                      type="primary" if i == 0 else "secondary", on_click=relax_and_search, args=([name],),
                      disabled=st.session_state.in_flight)
        if not blocking:
            st.button("Clear all filters and search again", key="relax_all", type="primary",
                      on_click=relax_and_search, args=(list(CONSTRAINT_WIDGETS),), disabled=st.session_state.in_flight)
    if r.outcome != "coverage_error":
        render_transparency(r)


def render_transparency(r: RecommendationResponse) -> None:
    with st.expander("Why these results"):
        applied = {k: v for k, v in r.applied_filters.model_dump().items() if v is not None}
        lines = [f"- **{FILTER_LABELS[k]}:** {md(filter_value(v))}" for k, v in applied.items()]
        st.markdown("**Filters the results satisfy**\n\n" + ("\n".join(lines) or "- None: all of Bengaluru"))
        if r.relaxations:
            steps = "\n".join(f"{i}. {md(x.reason)} ({x.matches_before} → {x.matches_after} matches)"
                              for i, x in enumerate(r.relaxations, start=1))
            st.markdown(f"**Relaxations, in order**\n\n{steps}")
        if r.interpretations:
            st.markdown("**How your input was read**\n\n" + "\n".join(f"- {md(i.note)}" for i in r.interpretations))
        st.caption(ranking_note(r))


def ranking_note(r: RecommendationResponse) -> str:
    seconds = f"{r.latency_ms / 1000:.1f} s"
    t = r.trace
    if t is None:
        return f"No ranking was needed · {seconds}"
    if t.ranker == "llm":
        how = f"Ranked by {t.model}"
        if t.dropped_ids:
            how += f"; {len(t.dropped_ids)} invalid pick(s) removed and {t.backfilled} refilled"
    else:
        how = "Ranked without AI" + (f" ({t.fallback_reason})" if t.fallback_reason else "")
    return f"{r.candidates_considered} restaurants considered · {md(how)} · {seconds}"


def main() -> None:
    st.set_page_config(page_title="Bengaluru Restaurant Finder", page_icon="🍽️")
    st.markdown(CSS, unsafe_allow_html=True)
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = list(value) if isinstance(value, list) else value
    if "in_flight" not in st.session_state:
        st.session_state.in_flight = False

    st.title("Where to eat in Bengaluru")
    st.caption("Filters choose candidates from the Zomato Bengaluru catalog. An AI model ranks them and explains each pick.")

    try:
        meta = load_meta()
    except ApiError as exc:  # U-01
        st.session_state.in_flight = False
        st.error(str(exc), icon="🔌")
        st.stop()

    render_sidebar(meta)
    if st.session_state.in_flight:
        run_search()

    error = st.session_state.get("error")
    if error:
        st.error(error, icon="⚠️")
    result = st.session_state.get("result")
    if result is not None:
        render_response(result)
    elif not error:
        st.info("Choose your preferences in the sidebar, then press **Find restaurants**.", icon="👈")


if __name__ == "__main__":
    main()
