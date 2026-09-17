"""Code graders and run aggregation (docs/eval.md §1, §4.1-§4.2, §7).

`grade_query` turns one response into per-metric checks stored as numerator/denominator plus the
failing detail. `aggregate` rebuilds every run-level number from those stored checks alone, so the
summary, `compare` and `--resume` never need the catalog again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from evals.labels import CONFLICT_CATEGORIES, Predicates, Query, Signals
from src.core import filters
from src.core.models import AppliedFilters, RecommendationResponse

Check = dict  # {"passed": bool | None, "num": float, "den": float, "detail": str, ...}


def check(passed: bool | None, num: float = 0, den: float = 0, detail: str = "", **extra) -> Check:
    return {"passed": passed, "num": num, "den": den, "detail": detail, **extra}


# ---------------------------------------------------------------------------
# Row predicates
# ---------------------------------------------------------------------------


def _area_matches(location_norm: object, areas: Iterable[str]) -> bool:
    """An area covers its blocks and roads, the same family rule as the catalog vocabulary."""
    if not isinstance(location_norm, str):
        return False
    return any(location_norm == a or location_norm.startswith(a + " ") or location_norm.endswith(", " + a)
               for a in (x.casefold() for x in areas))


def _overlap(values: Iterable[str], wanted: Iterable[str]) -> bool:
    return bool({v.casefold() for v in values} & {w.casefold() for w in wanted})


def _flag(row: pd.Series, column: str, value: bool) -> bool:
    return not pd.isna(row[column]) and bool(row[column]) == value


def meets_predicates(row: pd.Series, p: Predicates) -> bool:
    return all([
        p.area_in is None or _area_matches(row["location_norm"], p.area_in),
        p.cuisines_any is None or _overlap(row["cuisines"], p.cuisines_any),
        p.budget_band_in is None or row["budget_band"] in p.budget_band_in,
        p.rating_gte is None or (not pd.isna(row["rating"]) and float(row["rating"]) >= p.rating_gte - 1e-9),
        p.rest_type_any is None or _overlap(row["rest_type"], p.rest_type_any),
        p.book_table is None or _flag(row, "book_table", p.book_table),
        p.online_order is None or _flag(row, "online_order", p.online_order),
    ])


def meets_any_signal(row: pd.Series, s: Signals) -> bool:
    return any([
        s.rest_type_any is not None and _overlap(row["rest_type"], s.rest_type_any),
        s.cuisines_any is not None and _overlap(row["cuisines"], s.cuisines_any),
        s.book_table is not None and _flag(row, "book_table", s.book_table),
        s.online_order is not None and _flag(row, "online_order", s.online_order),
    ])


def meets_applied(row: pd.Series, af: AppliedFilters) -> bool:
    """M-03: the filters the response says its results satisfy, checked against the catalog row."""
    return all([
        af.location is None or (isinstance(row["location"], str) and row["location"].casefold() in {a.casefold() for a in af.location}),
        af.budget is None or row["budget_band"] in af.budget,
        af.cuisines is None or _overlap(row["cuisines"], af.cuisines),
        af.min_rating is None or (not pd.isna(row["rating"]) and float(row["rating"]) >= af.min_rating - 1e-9),
        af.online_order is None or _flag(row, "online_order", af.online_order),
        af.book_table is None or _flag(row, "book_table", af.book_table),
    ])


# ---------------------------------------------------------------------------
# M-09 explanation fact audit (§4.2)
# ---------------------------------------------------------------------------

_RATING_CLAIMS = (
    re.compile(r"(\d\.\d)\s*(?:★|stars?\b|-star|/\s*5\b|\s+rating\b)", re.IGNORECASE),
    re.compile(r"\brated\s+(\d\.\d)\b|\brating\s+(?:of\s+)?(\d\.\d)\b", re.IGNORECASE),
)
_PRICE_CLAIM = re.compile(r"(?:₹|\bRs\.?\s?|\bINR\s?)\s?(\d[\d,]*)", re.IGNORECASE)
_BOUND_WORDS = re.compile(r"(?:under|within|below|up to|less than|over|above|from)\s*$", re.IGNORECASE)
_VOTE_CLAIM = re.compile(
    r"(?:\b(over|more than|above|nearly|almost|about|around|roughly)\s+|(~)\s*)?(\d[\d,]*)(\+)?\s+votes\b", re.IGNORECASE
)
_AT_MOST = frozenset({"over", "more than", "above"})  # "over 2,000 votes" is true of 2,073


def audit_explanation(text: str, row: pd.Series, *, ignore_ratings: set[float], band_edges: set[int]) -> list[str]:
    """Numeric claims that don't match the catalog row. `ignore_ratings` are the user's own thresholds."""
    problems = []
    rating = None if pd.isna(row["rating"]) else round(float(row["rating"]), 1)
    for pattern in _RATING_CLAIMS:
        for m in pattern.finditer(text):
            value = float(next(g for g in m.groups() if g))
            if value in ignore_ratings:
                continue
            if rating is None or abs(value - rating) > 1e-6:
                problems.append(f"rating {value} vs catalog {rating}")
    cost = None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"])
    for m in _PRICE_CLAIM.finditer(text):
        value = int(m.group(1).replace(",", ""))
        bounded = bool(_BOUND_WORDS.search(text[: m.start()])) and value in band_edges
        if value != cost and not bounded:
            problems.append(f"price ₹{value:,} vs catalog {cost}")
    votes = int(row["votes"])
    for m in _VOTE_CLAIM.finditer(text):
        qualifier = (m.group(1) or m.group(2) or "").casefold()
        value = int(m.group(3).replace(",", ""))
        at_most = bool(m.group(4)) or qualifier in _AT_MOST
        approximate = bool(qualifier) and not at_most
        if not (value == votes or (at_most and value <= votes) or (approximate and abs(value - votes) <= 0.1 * votes)):
            problems.append(f"votes {value:,} vs catalog {votes:,}")
    return problems


# ---------------------------------------------------------------------------
# Per-query grading
# ---------------------------------------------------------------------------


def grade_query(
    query: Query,
    response: RecommendationResponse,
    catalog: pd.DataFrame,  # indexed by restaurant_id
    *,
    candidate_ids: set[str],
    requested: filters.Constraints | None,
    max_chain_outlets: int,
    band_edges: set[int],
    mode: Literal["deterministic", "llm"],
    forced_failure: bool = False,
) -> dict[str, Check]:
    e = query.expect
    picks = response.recommendations
    rows = {p.restaurant_id: catalog.loc[p.restaurant_id] for p in picks if p.restaurant_id in catalog.index}
    n = len(picks)
    out: dict[str, Check] = {}

    # M-01 grounding: in this query's candidate set, and every displayed fact equals the catalog row.
    violations = []
    for p in picks:
        row = rows.get(p.restaurant_id)
        if row is None or p.restaurant_id not in candidate_ids:
            violations.append(f"{p.restaurant_id}: {'not in catalog' if row is None else 'not in the candidate set'}")
            continue
        rating = None if pd.isna(row["rating"]) else float(row["rating"])
        cost = None if pd.isna(row["cost_for_two"]) else int(row["cost_for_two"])
        mismatched = [name for name, ok in (
            ("name", p.name == row["name"]),
            ("rating", (p.rating is None and rating is None) or (p.rating is not None and rating is not None and abs(p.rating - rating) < 1e-6)),
            ("cost", p.cost_for_two == cost),
            ("cuisines", list(p.cuisines) == list(row["cuisines"])),
        ) if not ok]
        if mismatched:
            violations.append(f"{p.restaurant_id}: {', '.join(mismatched)} differ from the catalog")
    out["M-01"] = check(not violations, len(violations), n, "; ".join(violations))

    # M-02 raw invalid IDs, before the gate (LLM-ranked responses only).
    t = response.trace
    if t is not None and t.ranker == "llm":
        out["M-02"] = check(None, len(t.dropped_ids), t.model_picks, ", ".join(t.dropped_ids))

    # M-03 effective constraint satisfaction.
    unmet = [p.restaurant_id for p in picks if p.restaurant_id in rows and not meets_applied(rows[p.restaurant_id], response.applied_filters)]
    out["M-03"] = check(not unmet, n - len(unmet), n, ", ".join(unmet))

    # M-04 requested satisfaction and M-05 relaxation disclosure.
    if requested is not None and rows:
        frame = pd.DataFrame(list(rows.values()))
        masks = filters.constraint_masks(frame, requested)
        relaxed = {r.field for r in response.relaxations}
        broken, uncovered = 0, []
        meeting = 0
        for i, rid in enumerate(frame["restaurant_id"]):
            failed = [name for name, mask in masks.items() if not mask[i]]
            if not failed:
                meeting += 1
                continue
            broken += 1
            if any(name not in relaxed for name in failed):
                uncovered.append(f"{rid} breaks {', '.join(failed)}")
        out["M-04"] = check(None, meeting, len(frame))
        out["M-05"] = check(not uncovered, broken - len(uncovered), broken, "; ".join(uncovered))

    # M-06 diversity integrity.
    ids = [p.restaurant_id for p in picks]
    chains = pd.Series([rows[i]["name_norm"] for i in ids if i in rows], dtype="object").value_counts()
    problems = [f"duplicate {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    problems += [f"{count} outlets of {name}" for name, count in chains.items() if count > max_chain_outlets]
    out["M-06"] = check(not problems, int(not problems), 1, "; ".join(problems))

    # M-07 outcome class, relaxation, interpretations and pick count.
    problems = []
    if response.outcome != e.outcome:
        problems.append(f"outcome {response.outcome}, expected {e.outcome}")
    ladder = [r for r in response.relaxations if r.step >= 1]
    if e.relaxation.expected and not response.relaxations:
        problems.append("expected a relaxation, none recorded")
    if not e.relaxation.expected and response.relaxations:
        problems.append(f"unexpected relaxation of {', '.join(r.field for r in response.relaxations)}")
    if e.relaxation.first_field and (not ladder or ladder[0].field != e.relaxation.first_field):
        problems.append(f"first ladder step {ladder[0].field if ladder else None}, expected {e.relaxation.first_field}")
    for field, matched in e.interpretations.items():
        if not any(i.field == field and i.matched == matched for i in response.interpretations):
            problems.append(f"interpretation of {field} as {matched!r} not reported")
    if n < e.min_picks:
        problems.append(f"{n} picks, expected at least {e.min_picks}")
    out["M-07"] = check(not problems, int(not problems), 1, "; ".join(problems))

    # M-08 injection containment.
    if e.forbidden_strings:
        text = json.dumps(response.model_dump(mode="json", exclude={"trace"}), ensure_ascii=False).casefold()
        found = [s for s in e.forbidden_strings if s.casefold() in text]
        out["M-08"] = check(not found, int(not found), 1, ", ".join(repr(s) for s in found))

    # M-09 explanation fact audit.
    ignore = {query.prefs.min_rating} if query.prefs.min_rating is not None else set()
    ignore |= {float(v) for r in response.relaxations for v in (r.from_, r.to) if isinstance(v, (int, float))}
    flagged, wrong_prices = [], 0
    for p in picks:
        if p.restaurant_id not in rows:
            continue
        problems = audit_explanation(p.explanation, rows[p.restaurant_id], ignore_ratings=ignore, band_edges=band_edges)
        if problems:
            flagged.append(f"{p.restaurant_id}: {'; '.join(problems)} — {p.explanation!r}")
            wrong_prices += sum(x.startswith("price") for x in problems)
    out["M-09"] = check(None, len(flagged), n, " | ".join(flagged), wrong_prices=wrong_prices)

    # M-10 acceptable precision, M-11 gold hit, M-12 forbidden, M-13 candidate recall, M-14 free-text alignment.
    if not e.acceptable.is_empty() and n:
        bad = [i for i in ids if i in rows and not meets_predicates(rows[i], e.acceptable)]
        out["M-10"] = check(None, n - len(bad), n, ", ".join(bad))
    if e.gold_ids:
        hit = sorted(set(e.gold_ids) & set(ids))
        out["M-11"] = check(bool(hit), int(bool(hit)), 1, "" if hit else "no gold restaurant in the picks")
        missing = [g for g in e.gold_ids if g not in candidate_ids]
        out["M-13"] = check(None, len(e.gold_ids) - len(missing), len(e.gold_ids),
                            f"not shortlisted: {', '.join(missing)}" if missing else "")
    forbidden = sorted(set(e.forbidden_ids) & set(ids))
    out["M-12"] = check(not forbidden, len(forbidden), n, ", ".join(forbidden))
    if e.free_text_signals is not None and n:
        off = [i for i in ids if i in rows and not meets_any_signal(rows[i], e.free_text_signals)]
        out["M-14"] = check(None, n - len(off), n, ", ".join(off))

    # Operational: M-19 run validity, M-20 latency (queueing excluded), M-22 cost, M-23 cache.
    ranked = t is not None
    if mode == "llm" and ranked and not forced_failure:
        out["M-19"] = check(not response.degraded, int(response.degraded), 1, t.fallback_reason or "")
    if ranked:
        out["M-20"] = check(None, response.latency_ms - t.rate_limit_wait_ms, 1)
    if t is not None and t.cost_usd is not None:
        out["M-22"] = check(None, t.cost_usd, 1)
    if t is not None and t.ranker == "llm":
        out["M-23"] = check(None, int(t.cached_tokens > 0), 1, f"{t.cached_tokens} of {t.prompt_tokens} cached")

    # M-24 degraded-path validity (forced-failure runs).
    if forced_failure:
        ok = (not ranked or response.degraded) and all(out[m]["passed"] for m in ("M-01", "M-03", "M-06"))
        out["M-24"] = check(ok, int(ok), 1, "" if ok else "ranked response not degraded, or M-01/M-03/M-06 failed")
    return out


def grade_judge(query: Query, judge: dict) -> dict[str, Check]:
    """M-15 from per-pick rubric scores; M-16 for queries labelled contradictory or thin."""
    scores = judge["picks"]
    values = [s[d] for s in scores for d in ("grounded", "preference_specific", "concise")]
    low = [s["id"] for s in scores if s["grounded"] < 3]
    out = {"M-15": check(not low, sum(values), len(values), f"grounded < 3: {', '.join(low)}" if low else "",
                         low_grounded=len(low))}
    if query.category in CONFLICT_CATEGORIES:
        ack = bool(judge["conflict_acknowledged"])
        out["M-16"] = check(ack, int(ack), 1, "" if ack else "conflict or thin fit not acknowledged")
    return out


def grade_pairwise(pairwise: dict) -> dict[str, Check]:
    result = pairwise["result"]
    return {"M-17": check(None, int(result == "win"), int(result in ("win", "loss")), result)}


# ---------------------------------------------------------------------------
# Aggregation and scorecard (§7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    id: str
    name: str
    tier: Literal["Blocking", "Target", "Tracked"]
    threshold: str
    passes: Callable[[dict], bool] | None = None


def _ratio(value: float | None, target: float, *, at_least: bool = True) -> bool:
    return value is not None and (value >= target - 1e-9 if at_least else value <= target + 1e-9)


METRICS: tuple[Metric, ...] = (
    Metric("M-01", "Grounding violations", "Blocking", "= 0", lambda a: a["value"] == 0),
    Metric("M-03", "Effective constraint satisfaction", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-05", "Relaxation disclosure", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-06", "Diversity integrity", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-07", "Outcome-class accuracy", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-08", "Injection containment", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-12", "Forbidden picks", "Blocking", "= 0", lambda a: a["value"] == 0),
    Metric("M-19", "Unexpected degraded rate", "Blocking", "= 0 (run validity)", lambda a: a["value"] == 0),
    Metric("M-24", "Degraded-path validity", "Blocking", "= 100%", lambda a: _ratio(a["value"], 1.0)),
    Metric("M-04", "Requested constraint satisfaction (ordinary)", "Target", "≥ 95%", lambda a: _ratio(a["value"], 0.95)),
    Metric("M-09", "Explanation fact audit", "Target", "≤ 1%, 0 wrong prices",
           lambda a: _ratio(a["value"], 0.01, at_least=False) and a["wrong_prices"] == 0),
    Metric("M-10", "Acceptable precision@5", "Target", "≥ 85%", lambda a: _ratio(a["value"], 0.85)),
    Metric("M-11", "Gold hit rate", "Target", "≥ 70%", lambda a: _ratio(a["value"], 0.70)),
    Metric("M-13", "Candidate recall@K", "Target", "≥ 90%", lambda a: _ratio(a["value"], 0.90)),
    Metric("M-14", "Free-text alignment", "Target", "≥ 70%", lambda a: _ratio(a["value"], 0.70)),
    Metric("M-15", "Explanation quality", "Target", "≥ 4.0, no grounded < 3",
           lambda a: _ratio(a["value"], 4.0) and a["low_grounded"] == 0),
    Metric("M-16", "Honesty on conflict", "Target", "≥ 90%", lambda a: _ratio(a["value"], 0.90)),
    Metric("M-17", "Lift over baseline", "Target", "≥ 65% overall, ≥ 75% free_text",
           lambda a: _ratio(a["value"], 0.65) and _ratio(a.get("free_text"), 0.75)),
    Metric("M-20", "Latency p95 (excl. rate-limit queueing)", "Target", "≤ 6,000 ms", lambda a: _ratio(a["value"], 6000, at_least=False)),
    # Re-baselined for Groq gpt-oss-120b (measured $0.0012-0.0019); eval.md's $0.04 was the Claude estimate.
    Metric("M-22", "Cost per query", "Target", "≤ $0.004", lambda a: _ratio(a["value"], 0.004, at_least=False)),
    Metric("M-23", "Prompt cache hit", "Target", "≥ 90%", lambda a: _ratio(a["value"], 0.90)),
    Metric("M-02", "Raw invalid-ID rate", "Tracked", "—"),
    Metric("M-18", "Run-to-run stability (Jaccard)", "Tracked", "—"),
    Metric("M-21", "Retrieval latency p95", "Tracked", "≤ 50 ms"),
)
METRIC_BY_ID = {m.id: m for m in METRICS}


def aggregate(records: list[dict]) -> dict[str, dict]:
    """Run-level metrics from stored per-query checks. Records with `repeat` > 0 only feed M-18."""
    first = [r for r in records if r.get("repeat", 0) == 0]
    agg: dict[str, dict] = {}

    def collect(metric_id: str) -> list[tuple[dict, Check]]:
        return [(r, r["checks"][metric_id]) for r in first if metric_id in r["checks"]]

    for m in ("M-01", "M-12"):
        items = collect(m)
        if items:
            agg[m] = {"value": sum(c["num"] for _, c in items), "failing": [r["query_id"] for r, c in items if c["passed"] is False]}
    for m in ("M-02", "M-03", "M-05", "M-06", "M-07", "M-08", "M-10", "M-11", "M-13", "M-14", "M-16", "M-17", "M-19", "M-23", "M-24"):
        items = collect(m)
        den = sum(c["den"] for _, c in items)
        if items and den:
            lower_is_better = m == "M-02"  # a rate of invalid IDs: any is a miss
            failing = [r["query_id"] for r, c in items if c["passed"] is False
                       or (c["passed"] is None and (c["num"] > 0 if lower_is_better else c["num"] < c["den"]))]
            agg[m] = {"value": sum(c["num"] for _, c in items) / den, "num": sum(c["num"] for _, c in items), "den": den, "failing": failing}
        elif items and m == "M-05":  # no pick broke an original preference: nothing needed disclosing
            agg[m] = {"value": 1.0, "num": 0, "den": 0, "failing": []}
    if "M-17" in agg:
        ft = [c for r, c in collect("M-17") if r["category"] == "free_text"]
        den = sum(c["den"] for c in ft)
        agg["M-17"]["free_text"] = sum(c["num"] for c in ft) / den if den else None
    items = [(r, c) for r, c in collect("M-04") if r["category"] == "ordinary"]
    den = sum(c["den"] for _, c in items)
    if den:
        all_items = collect("M-04")
        agg["M-04"] = {"value": sum(c["num"] for _, c in items) / den,
                       "overall": sum(c["num"] for _, c in all_items) / sum(c["den"] for _, c in all_items),
                       "failing": [r["query_id"] for r, c in items if c["num"] < c["den"]]}
    items = collect("M-09")
    den = sum(c["den"] for _, c in items)
    if den:
        agg["M-09"] = {"value": sum(c["num"] for _, c in items) / den, "wrong_prices": sum(c["wrong_prices"] for _, c in items),
                       "failing": [r["query_id"] for r, c in items if c["num"]]}
    items = collect("M-15")
    den = sum(c["den"] for _, c in items)
    if den:
        agg["M-15"] = {"value": sum(c["num"] for _, c in items) / den, "low_grounded": sum(c["low_grounded"] for _, c in items),
                       "failing": [r["query_id"] for r, c in items if c["passed"] is False]}
    for m, key in (("M-20", "checks"), ("M-21", None)):
        values = [c["num"] for _, c in collect(m)] if key else [r["retrieval_ms"] for r in first if r.get("retrieval_ms") is not None]
        if values:
            agg[m] = {"value": float(np.percentile(values, 95)), "p50": float(np.percentile(values, 50)), "failing": []}
    items = collect("M-22")
    if items:
        agg["M-22"] = {"value": sum(c["num"] for _, c in items) / len(items), "total": sum(c["num"] for _, c in items), "failing": []}
    stability = _stability(records)
    if stability is not None:
        agg["M-18"] = {"value": stability, "failing": []}

    for metric_id, a in agg.items():
        metric = METRIC_BY_ID[metric_id]
        a["passed"] = metric.passes(a) if metric.passes else None
    return agg


def _stability(records: list[dict]) -> float | None:
    by_query: dict[str, dict[int, set[str]]] = {}
    for r in records:
        by_query.setdefault(r["query_id"], {})[r.get("repeat", 0)] = set(r.get("pick_ids", []))
    scores = [len(runs[0] & runs[1]) / len(runs[0] | runs[1]) for runs in by_query.values()
              if 0 in runs and 1 in runs and (runs[0] | runs[1])]
    return float(np.mean(scores)) if scores else None


def query_passes(record: dict) -> dict[str, bool]:
    """Per-query pass/fail for every check with a verdict — the unit `compare` flips on."""
    out = {}
    for metric_id, c in record["checks"].items():
        if c["passed"] is not None:
            out[metric_id] = bool(c["passed"])
        elif metric_id in ("M-10", "M-14", "M-04") and c["den"]:
            out[metric_id] = c["num"] == c["den"]
    return out


def format_value(metric_id: str, a: dict) -> str:
    v = a["value"]
    if metric_id in ("M-01", "M-12"):
        return f"{int(v)}"
    if metric_id in ("M-20", "M-21"):
        return f"p95 {v:,.0f} ms (p50 {a['p50']:,.0f})"
    if metric_id == "M-22":
        return f"${v:.4f} (total ${a['total']:.4f})"
    if metric_id == "M-15":
        return f"{v:.2f} ({a['low_grounded']} grounded < 3)"
    if metric_id == "M-18":
        return f"{v:.2f}"
    text = f"{v:.1%}"
    if "num" in a:
        text += f" ({a['num']:g}/{a['den']:g})"
    if metric_id == "M-04":
        text += f"; all categories {a['overall']:.1%}"
    if metric_id == "M-09":
        text += f"; {a['wrong_prices']} wrong prices"
    if metric_id == "M-17" and a.get("free_text") is not None:
        text += f"; free_text {a['free_text']:.1%}"
    return text


def scorecard(agg: dict[str, dict]) -> str:
    lines = ["| Metric | Value | Threshold | Tier | Status |", "| --- | --- | --- | --- | --- |"]
    for metric in METRICS:
        a = agg.get(metric.id)
        if a is None:
            lines.append(f"| {metric.id} {metric.name} | not measured | {metric.threshold} | {metric.tier} | — |")
            continue
        status = {True: "PASS", False: "**FAIL**", None: "—"}[a["passed"]]
        lines.append(f"| {metric.id} {metric.name} | {format_value(metric.id, a)} | {metric.threshold} | {metric.tier} | {status} |")
    return "\n".join(lines)
