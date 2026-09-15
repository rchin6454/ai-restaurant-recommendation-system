"""CLI harness: preferences → ranked picks (plan tasks 2.7, 3.6-3.7).

    python -m src.cli --location Koramangala --budget medium --cuisine "North Indian" --min-rating 4.0
    python -m src.cli --location Indiranagar --free-text "family-friendly, quiet" --verbose
    python -m src.cli --location "BTM" --cuisine Chinese --cuisine Momos --online-order --no-llm --json
"""

from __future__ import annotations

import argparse
import logging
import sys

from pydantic import ValidationError

from src.core.models import Preferences, Recommendation, RecommendationResponse
from src.core.recommender import recommend
from src.data.catalog import CatalogError
from src.llm.client import call_budget


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.cli", description="Recommend Bengaluru restaurants.")
    p.add_argument("--location", help="Bengaluru area, e.g. Koramangala (typos are fuzzy-matched)")
    p.add_argument("--budget", choices=["low", "medium", "high"])
    p.add_argument("--cuisine", action="append", default=[], dest="cuisines", help="repeatable; OR-matched")
    p.add_argument("--min-rating", type=float)
    p.add_argument("--party-size", type=int)
    p.add_argument("--free-text", help='anything else, e.g. "family-friendly" — weighed by the LLM ranker')
    p.add_argument("--online-order", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--book-table", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument("--no-llm", action="store_true", help="rank deterministically (the phase-2 reference output)")
    p.add_argument("--max-llm-calls", type=int, default=3, help="hard cap on paid LLM calls this run (default 3)")
    p.add_argument("--json", action="store_true", help="print the raw response JSON")
    p.add_argument("--verbose", action="store_true", help="show structured logs, including token usage")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verbose:
        logging.getLogger("src").setLevel(logging.WARNING)
    call_budget.set_cap(max(args.max_llm_calls, 0))  # L-24

    try:
        prefs = Preferences(
            location=args.location,
            budget=args.budget,
            cuisines=args.cuisines,
            min_rating=args.min_rating,
            party_size=args.party_size,
            free_text=args.free_text,
            online_order=args.online_order,
            book_table=args.book_table,
        )
    except ValidationError as exc:
        for err in exc.errors():
            print(f"invalid --{'-'.join(str(p) for p in err['loc']).replace('_', '-')}: {err['msg']}", file=sys.stderr)
        return 2

    try:
        response = recommend(prefs, args.top_n, use_llm=not args.no_llm)
    except CatalogError as exc:
        print(exc, file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"invalid input: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(response.model_dump_json(indent=2, by_alias=True))
    else:
        print(render(response))
    return 0


def render(r: RecommendationResponse) -> str:
    lines = [f"{r.outcome} · {r.candidates_considered} candidates · {r.latency_ms} ms{_ranker_note(r)}", "", r.summary]
    for i in r.interpretations:
        lines.append(f"  ↳ {i.note}")

    filters = {k: v for k, v in r.applied_filters.model_dump().items() if v is not None}
    if filters or r.outcome != "coverage_error":
        shown = []
        for key, value in filters.items():
            if isinstance(value, list):
                value = ", ".join(value[:6]) + (f" +{len(value) - 6} more" if len(value) > 6 else "")
            shown.append(f"{key}={value}")
        lines.append(f"\nApplied filters: {'; '.join(shown) or 'none'}")
    if r.relaxations:
        lines.append("Relaxations:")
        for x in r.relaxations:
            label = "stretch" if x.step == 0 else f"step {x.step}"
            lines.append(f"  [{label}] {x.field}: {_value(x.from_)} → {_value(x.to)}  ({x.matches_before} → {x.matches_after} matches)")
    if r.caveats:
        lines.append("Caveats:")
        lines.extend(f"  - {c}" for c in r.caveats)
    if r.suggestions:
        lines.append(f"Suggestions: {', '.join(r.suggestions)}")
    if r.trace and r.trace.dropped_ids:
        lines.append(f"Grounding: dropped {len(r.trace.dropped_ids)} invalid pick(s), backfilled {r.trace.backfilled}")

    for rec in r.recommendations:
        stretch = " [budget stretch]" if rec.stretch else ""
        lines.append(f"\n{rec.rank}. {rec.name} — {rec.area or 'area unknown'}{stretch}   ({rec.restaurant_id})")
        lines.append(f"   {_facts(rec)}")  # catalog facts…
        lines.append(f"   » {rec.explanation}")  # …kept visually apart from the ranker's prose (§8)
        if rec.match_highlights:
            lines.append(f"   [{' · '.join(rec.match_highlights)}]")
    return "\n".join(lines)


def _ranker_note(r: RecommendationResponse) -> str:
    t = r.trace
    if t is None:
        return ""
    if t.ranker == "llm":
        cost = f" · ${t.cost_usd:.4f}" if t.cost_usd is not None else ""
        return (f" · {t.model} · {t.prompt_tokens:,} in ({t.cached_tokens:,} cached) / "
                f"{t.completion_tokens:,} out{cost}")
    return f" · template explanations (no LLM: {t.fallback_reason})" if t.fallback_reason else " · template explanations"


def _facts(rec: Recommendation) -> str:
    rating = f"{rec.rating:.1f}★ ({rec.votes:,} votes)" if rec.rating is not None else "not yet rated"
    cost = f"₹{rec.cost_for_two:,} for two" if rec.cost_for_two is not None else "cost not listed"
    return " · ".join([rating, ", ".join(rec.cuisines[:3]) or "cuisine not listed", cost])


def _value(v) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, list):
        return ", ".join(v[:4]) + (f" +{len(v) - 4} more" if len(v) > 4 else "")
    return str(v)


if __name__ == "__main__":
    sys.exit(main())
