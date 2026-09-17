"""LLM judge: explanation quality (M-15, M-16), pairwise lift (M-17) and calibration (plan 5.9).

    python -m evals.judge export-sheet --run evals/results/<run>_llm.jsonl --queries ord-02,ft-02,con-01
    python -m evals.judge calibrate --run evals/results/<run>_llm.jsonl --sheet evals/results/calibration_sheet.csv

Calls go through `src.llm.ranker.structured_completion`, so they share the ranker's key check, call
cap and Groq rate limiter: a judge call spends the same daily token budget as a ranking call.
`catalog` arguments must be indexed by `restaurant_id` with the column kept (`set_index(..., drop=False)`).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import groq
import pandas as pd
from pydantic import ValidationError

from evals.judge_prompts import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPT,
    PAIRWISE_SCHEMA,
    PAIRWISE_SYSTEM_PROMPT,
    JudgeOutput,
    PairwiseOutput,
)
from src.config import PROJECT_ROOT, Settings
from src.core.models import Preferences, RecommendationResponse
from src.llm.client import LLMUnavailable
from src.llm.ranker import StructuredCompletion, candidate_record, structured_completion

JUDGE_COMPLETION_RESERVE = 1500
RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
DIMENSIONS = ("grounded", "preference_specific", "concise")
CALIBRATION_AGREEMENT = 0.80  # §4.3: within ±1 on ≥ 80% of scores
UNGROUNDED_MAX = 2  # a grounded score at or below this counts as "marked ungrounded"


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def catalog_row(catalog: pd.DataFrame, restaurant_id: str) -> dict:
    """The same fields the ranker saw, minus its internal `meets_request` flag."""
    record = candidate_record(catalog.loc[restaurant_id])
    record.pop("meets_request", None)
    return record


def _call(messages: list[dict], *, schema_name: str, schema: dict, config: Settings, client: groq.Groq | None) -> StructuredCompletion:
    return structured_completion(messages, schema_name=schema_name, schema=schema, config=config, client=client,
                                 completion_reserve=JUDGE_COMPLETION_RESERVE)


# ---------------------------------------------------------------------------
# Rubric judge (§4.3)
# ---------------------------------------------------------------------------


def judge_messages(prefs: Preferences, response: RecommendationResponse, catalog: pd.DataFrame,
                   issue_to_check: str | None) -> list[dict[str, str]]:
    document = {
        "request": prefs.model_dump(mode="json"),
        "issue_to_check": issue_to_check,
        "picks": [
            {"id": p.restaurant_id, "catalog_row": catalog_row(catalog, p.restaurant_id),
             "explanation": p.explanation, "match_highlights": p.match_highlights}
            for p in response.recommendations
        ],
        "summary": response.summary,
        "caveats": response.caveats,
    }
    return [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": _dumps(document)}]


def judge_response(
    prefs: Preferences,
    response: RecommendationResponse,
    catalog: pd.DataFrame,
    *,
    issue_to_check: str | None,
    config: Settings,
    client: groq.Groq | None = None,
) -> dict:
    """Scores for every pick, in pick order. Raises `LLMUnavailable` for transport, rate-limit or unusable output."""
    result = _call(judge_messages(prefs, response, catalog, issue_to_check), schema_name="explanation_judgement",
                   schema=JUDGE_SCHEMA, config=config, client=client)
    try:
        output = JudgeOutput.model_validate_json(result.content)
    except ValidationError as exc:
        raise LLMUnavailable(f"judge output failed validation ({exc.error_count()} errors)", trace=result.trace) from exc
    by_id = {s.id.strip().casefold(): s for s in output.picks}
    scores = []
    for p in response.recommendations:
        score = by_id.get(p.restaurant_id.casefold())
        if score is None:
            raise LLMUnavailable(f"judge skipped pick {p.restaurant_id}", trace=result.trace)
        scores.append({**score.model_dump(), "id": p.restaurant_id})
    return {"picks": scores, "conflict_acknowledged": output.conflict_acknowledged, **_usage([result])}


# ---------------------------------------------------------------------------
# Pairwise judge (§4.4)
# ---------------------------------------------------------------------------


def pairwise_messages(prefs: Preferences, list_a: Sequence[str], list_b: Sequence[str],
                      catalog: pd.DataFrame) -> list[dict[str, str]]:
    document = {
        "request": prefs.model_dump(mode="json"),
        "list_A": [catalog_row(catalog, rid) for rid in list_a],
        "list_B": [catalog_row(catalog, rid) for rid in list_b],
    }
    return [{"role": "system", "content": PAIRWISE_SYSTEM_PROMPT}, {"role": "user", "content": _dumps(document)}]


def pairwise_compare(
    prefs: Preferences,
    llm_ids: Sequence[str],
    baseline_ids: Sequence[str],
    catalog: pd.DataFrame,
    *,
    config: Settings,
    client: groq.Groq | None = None,
) -> dict:
    """LLM list vs deterministic baseline, asked in both orders. Explanations are left out of both lists,
    so the comparison is about which restaurants were chosen, not prose style. Identical lists are a tie
    without spending a call."""
    if list(llm_ids) == list(baseline_ids):
        return {"result": "tie", "verdicts": [], "note": "identical lists", "cost_usd": 0.0, "tokens": 0}
    verdicts, results = [], []
    for llm_position, (a, b) in (("A", (llm_ids, baseline_ids)), ("B", (baseline_ids, llm_ids))):
        result = _call(pairwise_messages(prefs, a, b, catalog), schema_name="shortlist_comparison",
                       schema=PAIRWISE_SCHEMA, config=config, client=client)
        results.append(result)
        try:
            output = PairwiseOutput.model_validate_json(result.content)
        except ValidationError as exc:
            raise LLMUnavailable(f"pairwise output failed validation ({exc.error_count()} errors)", trace=result.trace) from exc
        winner = "tie" if output.preferred == "tie" else ("llm" if output.preferred == llm_position else "baseline")
        verdicts.append({"llm_position": llm_position, "preferred": output.preferred, "winner": winner, "reason": output.reason})
    winners = {v["winner"] for v in verdicts}
    outcome = "win" if winners == {"llm"} else "loss" if winners == {"baseline"} else "tie"
    return {"result": outcome, "verdicts": verdicts, **_usage(results)}


def _usage(results: list[StructuredCompletion]) -> dict:
    return {
        "cost_usd": round(sum(r.trace.cost_usd or 0.0 for r in results), 6),
        "tokens": sum(r.trace.prompt_tokens + r.trace.completion_tokens for r in results),
    }


# ---------------------------------------------------------------------------
# Calibration (§4.3): a blind human sheet vs the judge's stored scores
# ---------------------------------------------------------------------------

SHEET_COLUMNS = ["query_id", "request", "restaurant_id", "catalog_row", "explanation", *DIMENSIONS]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _query_records(run_path: Path) -> list[dict]:
    return [r for r in read_jsonl(run_path) if r.get("type") == "query" and r.get("repeat", 0) == 0]


def export_sheet(run_path: Path, query_ids: list[str], out_path: Path, catalog: pd.DataFrame) -> int:
    """Write the picks to hand-score, without the judge's scores (blind)."""
    records = {r["query_id"]: r for r in _query_records(run_path)}
    missing = [q for q in query_ids if q not in records]
    if missing:
        raise ValueError(f"queries not in {run_path.name}: {missing}")
    rows = []
    for qid in query_ids:
        record = records[qid]
        for pick in record["response"]["recommendations"]:
            rows.append({
                "query_id": qid,
                "request": _dumps(record["prefs"]),
                "restaurant_id": pick["restaurant_id"],
                "catalog_row": _dumps(catalog_row(catalog, pick["restaurant_id"])),
                "explanation": pick["explanation"],
                **{d: "" for d in DIMENSIONS},
            })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def calibrate(run_path: Path, sheet_path: Path) -> tuple[bool, str]:
    """Agreement within ±1 on ≥ 80% of scores, and every pick a human marked ungrounded caught by the judge."""
    judged = {}
    for record in _query_records(run_path):
        for score in (record.get("judge") or {}).get("picks", []):
            judged[(record["query_id"], score["id"])] = score
    with sheet_path.open(newline="", encoding="utf-8") as f:
        human = list(csv.DictReader(f))

    agree = total = 0
    missed, lines = [], []
    for row in human:
        key = (row["query_id"], row["restaurant_id"])
        if key not in judged:
            raise ValueError(f"{key} has no judge score in {run_path.name}; run the eval with --judge first")
        try:
            h = {d: int(row[d]) for d in DIMENSIONS}
        except ValueError:
            raise ValueError(f"{key}: fill every score column with 1-5 before calibrating") from None
        j = judged[key]
        for d in DIMENSIONS:
            total += 1
            agree += abs(h[d] - j[d]) <= 1
        if h["grounded"] <= UNGROUNDED_MAX and j["grounded"] > UNGROUNDED_MAX:
            missed.append(key)
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(f"{h[d]} / {j[d]}" for d in DIMENSIONS) + " |")

    rate = agree / total if total else 0.0
    passed = total > 0 and rate >= CALIBRATION_AGREEMENT and not missed
    report = "\n".join([
        f"# Judge calibration — {'PASS' if passed else 'FAIL'}",
        "",
        f"- Run: `{run_path.name}`; sheet: `{sheet_path.name}`; {len(human)} picks, {total} scores",
        f"- Agreement within ±1: {rate:.1%} (need ≥ {CALIBRATION_AGREEMENT:.0%})",
        f"- Human-ungrounded picks the judge missed: {len(missed)} {missed if missed else ''}",
        "",
        "| Query | Restaurant | grounded (human / judge) | preference_specific | concise |",
        "| --- | --- | --- | --- | --- |",
        *lines,
    ])
    return passed, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.judge", description="LLM-judge calibration (plan 5.9).")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export-sheet", help="write a blind hand-scoring sheet from a judged run")
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--queries", required=True, help="comma-separated query ids (3 queries × 5 picks = 15)")
    export.add_argument("--out", type=Path, default=RESULTS_DIR / "calibration_sheet.csv")
    cal = sub.add_parser("calibrate", help="compare a filled-in sheet with the run's judge scores")
    cal.add_argument("--run", type=Path, required=True)
    cal.add_argument("--sheet", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "export-sheet":
        from src.data.catalog import get_catalog

        catalog = get_catalog().set_index("restaurant_id", drop=False)
        n = export_sheet(args.run, [q.strip() for q in args.queries.split(",") if q.strip()], args.out, catalog)
        print(f"wrote {n} picks to {args.out}. Score each 1-5 without looking at the run, then run `calibrate`.")
        return 0

    passed, report = calibrate(args.run, args.sheet)
    out = RESULTS_DIR / f"calibration_{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}.md"
    out.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nreport: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
