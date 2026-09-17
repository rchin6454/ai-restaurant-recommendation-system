"""Eval runner (docs/eval.md §5; plan 5.5).

    python -m evals.run_eval --mode deterministic                    # free and CI-safe
    python -m evals.run_eval --mode deterministic --save-baseline    # phase-2 reference, needed by --pairwise
    python -m evals.run_eval --mode llm --limit 5                    # live smoke
    python -m evals.run_eval --mode llm --judge --pairwise           # release-candidate quality
    python -m evals.run_eval --mode llm --force-llm-failure          # M-24, no Groq calls
    python -m evals.run_eval --mode llm --resume evals/results/<run>_llm.jsonl
    python -m evals.run_eval --check-labels

Live budget: the Groq account fits a ~6K-token ranking call about once a minute and ~31 a day, and
judge calls spend the same budget. The runner waits for the rate limiter (up to `LLM_WAIT_S` per
call) instead of degrading, which would invalidate the run (M-19). When a limit still can't be met,
usually the daily one, it stops, saves an incomplete run, and `--resume` continues from there.

`--via-api` sends requests to a running API instead. Start it so it neither throttles nor degrades
the eval: `LLM_RATE_LIMIT_MAX_WAIT_S=90 API_REQUESTS_PER_MINUTE=60 RESPONSE_CACHE_ENABLED=false uvicorn src.api.main:app`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
import pandas as pd
from pydantic import ValidationError

from evals.judge import judge_response, pairwise_compare, read_jsonl
from evals.judge_prompts import JUDGE_PROMPT_SHA256
from evals.labels import CONFLICT_CATEGORIES, QUERIES_PATH, Query, check_labels, file_sha256, load_queries
from evals.metrics import METRIC_BY_ID, aggregate, check, grade_judge, grade_pairwise, grade_query, scorecard
from src.api.routes import budget_bands
from src.config import PROJECT_ROOT, Settings, settings
from src.core.models import RecommendationResponse
from src.core.recommender import PROMPT_SHA256, recommend, shortlist
from src.data.catalog import get_catalog, get_vocabulary
from src.llm.client import LLMUnavailable, call_budget

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
BASELINE_PATH = RESULTS_DIR / "baseline_deterministic.json"
LLM_WAIT_S = 90.0  # one ranking call fits per minute: queue for it rather than degrade
LIMIT_MARKERS = ("Groq rate limit", "RateLimitError", "call cap")
FINGERPRINT = ("mode", "forced_failure", "via_api", "model", "rank_weights", "llm_candidate_k", "min_candidates",
               "max_chain_outlets", "ranking_prompt_sha256", "queries_sha256", "catalog_rows")


class StopRun(Exception):
    """The next call can't fit the account's limits or --max-calls. Save the run; resume later."""


class _FailingCompletions:
    def create(self, **_kwargs):
        raise groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))


FAILING_CLIENT = SimpleNamespace(chat=SimpleNamespace(completions=_FailingCompletions()))


def _limit_hit(reason: str | None) -> bool:
    return bool(reason) and any(marker in reason for marker in LIMIT_MARKERS)


_PAUSE = re.compile(r"paused for (\d+) s")
MAX_429_RETRIES = 3


def short_pause(reason: str | None) -> float | None:
    """Seconds to wait out a Groq 429 that asked for a short pause; None when it won't clear soon.

    Groq counts tokens a little differently from the local limiter, so an occasional 429 with a
    few seconds' `retry-after` still happens. That's worth waiting for; a long pause (the daily
    limit) stops the run instead."""
    m = _PAUSE.search(reason or "")
    return float(m.group(1)) + 1.0 if m and int(m.group(1)) <= LLM_WAIT_S else None


def _with_429_retry(label: str, call):
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            return call()
        except LLMUnavailable as exc:
            pause = short_pause(str(exc))
            if pause is None or attempt == MAX_429_RETRIES:
                raise
            print(f"  Groq 429 during {label}: waiting {pause:.0f} s, then retrying", flush=True)
            time.sleep(pause)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def run_config(args: argparse.Namespace) -> Settings:
    overrides: dict = {"response_cache_enabled": False}  # §5.2 rule 1: measure the system, not the cache
    if args.force_llm_failure:  # no real calls happen, so the rate limiter must not wait for them
        overrides.update(llm_rate_limit_max_wait_s=0, llm_requests_per_minute=10**9, llm_requests_per_day=10**9,
                         llm_tokens_per_minute=10**12, llm_tokens_per_day=10**12)
    elif args.mode == "llm":
        overrides["llm_rate_limit_max_wait_s"] = LLM_WAIT_S
    return settings.model_copy(update=overrides)


def _git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def snapshot(args: argparse.Namespace, config: Settings, catalog: pd.DataFrame, run_id: str) -> dict:
    """§5.2 rule 3: everything needed to trace a result back to the configuration that produced it."""
    return {
        "type": "run",
        "run_id": run_id,
        "mode": args.mode,
        "forced_failure": args.force_llm_failure,
        "via_api": args.via_api,
        "flags": {"judge": args.judge, "pairwise": args.pairwise, "repeat": args.repeat, "category": args.category,
                  "ids": args.ids, "limit": args.limit, "max_calls": args.max_calls},
        "model": config.model,
        "rank_weights": config.rank_weights.model_dump(),
        "llm_candidate_k": config.llm_candidate_k,
        "min_candidates": config.min_candidates,
        "max_chain_outlets": config.max_chain_outlets,
        "ranking_prompt_sha256": PROMPT_SHA256,
        "judge_prompt_sha256": JUDGE_PROMPT_SHA256 if (args.judge or args.pairwise) else None,
        "queries_sha256": file_sha256(args.queries),
        "catalog_rows": len(catalog),
        "git_commit": _git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def read_run(path: Path) -> tuple[dict, list[dict], dict | None]:
    """Header, query records (the last record per query and repeat wins) and the summary, if written."""
    header, summary, latest = None, None, {}
    for record in read_jsonl(path):
        kind = record.get("type")
        if kind == "run":
            header = record
        elif kind == "summary":
            summary = record
        elif kind == "query":
            latest[(record["query_id"], record.get("repeat", 0))] = record
    if header is None:
        raise ValueError(f"{path} has no run header")
    return header, list(latest.values()), summary


# ---------------------------------------------------------------------------
# One query
# ---------------------------------------------------------------------------


def _post(url: str, body: dict) -> dict:
    for _ in range(5):  # the API's per-client limit is a deployment detail; wait it out
        resp = httpx.post(f"{url.rstrip('/')}/recommend", json=body, timeout=httpx.Timeout(300.0, connect=5.0))
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        time.sleep(float(resp.json()["error"].get("retry_after_s") or 5) + 0.5)
    raise StopRun("the API kept answering 429 — raise API_REQUESTS_PER_MINUTE for eval runs")


def run_query(q: Query, repeat: int, *, args: argparse.Namespace, config: Settings, indexed: pd.DataFrame,
              band_edges: set[int]) -> dict:
    started = time.perf_counter()
    listed = shortlist(q.prefs, config=config)
    retrieval_ms = round((time.perf_counter() - started) * 1000, 2)
    use_llm = args.mode == "llm"
    record = {"type": "query", "query_id": q.id, "category": q.category, "repeat": repeat,
              "prefs": q.prefs.model_dump(mode="json"), "retrieval_ms": retrieval_ms,
              "candidate_ids": listed.candidates["restaurant_id"].tolist(), "judge": None, "pairwise": None}

    if args.via_api:
        raw = _post(args.via_api, {**q.prefs.model_dump(mode="json", exclude_none=True), "use_llm": use_llm})
        try:
            response = RecommendationResponse.model_validate(raw)
        except ValidationError as exc:  # phase 4 gate: every response validates against the schema
            return record | {"response": raw, "pick_ids": [],
                             "checks": {"schema": check(False, 0, 1, f"{exc.error_count()} schema errors")}}
    else:
        for attempt in range(MAX_429_RETRIES + 1):
            response = recommend(q.prefs, config=config, use_llm=use_llm,
                                 llm_client=FAILING_CLIENT if args.force_llm_failure else None)
            reason = response.trace.fallback_reason if response.trace else None
            pause = short_pause(reason) if use_llm and response.degraded and not args.force_llm_failure else None
            if pause is None or attempt == MAX_429_RETRIES:
                break
            print(f"  Groq 429 while ranking {q.id}: waiting {pause:.0f} s, then retrying", flush=True)
            time.sleep(pause)

    if use_llm and not args.force_llm_failure and response.degraded and response.trace and _limit_hit(response.trace.fallback_reason):
        raise StopRun(response.trace.fallback_reason)
    checks = {"schema": check(True, 1, 1)}
    checks |= grade_query(q, response, indexed, candidate_ids=set(record["candidate_ids"]), requested=listed.requested,
                          max_chain_outlets=config.max_chain_outlets, band_edges=band_edges, mode=args.mode,
                          forced_failure=args.force_llm_failure)
    return record | {"response": response.model_dump(mode="json", by_alias=True),
                     "pick_ids": [p.restaurant_id for p in response.recommendations], "checks": checks}


def add_judgements(record: dict, q: Query, *, args: argparse.Namespace, config: Settings, indexed: pd.DataFrame,
                   baseline: dict | None) -> bool:
    """Judge and pairwise passes for one record, skipping parts already done. Returns True if it changed."""
    if "schema" in record["checks"] and not record["checks"]["schema"]["passed"]:
        return False
    response = RecommendationResponse.model_validate(record["response"])
    if not response.recommendations:
        return False
    changed = False
    if args.judge and not (record.get("judge") or {}).get("picks"):
        issue = q.notes if q.category in CONFLICT_CATEGORIES else None
        try:
            record["judge"] = _with_429_retry(f"judging {q.id}", lambda: judge_response(
                q.prefs, response, indexed, issue_to_check=issue, config=config))
            record["checks"] |= grade_judge(q, record["judge"])
        except LLMUnavailable as exc:
            if _limit_hit(str(exc)):
                raise StopRun(str(exc)) from exc
            record["judge"] = {"error": str(exc)}
        changed = True
    base = (baseline or {}).get("queries", {}).get(q.id)
    if args.pairwise and base and base["pick_ids"] and not (record.get("pairwise") or {}).get("result"):
        try:
            record["pairwise"] = _with_429_retry(f"pairwise {q.id}", lambda: pairwise_compare(
                q.prefs, record["pick_ids"], base["pick_ids"], indexed, config=config))
            record["checks"] |= grade_pairwise(record["pairwise"])
        except LLMUnavailable as exc:
            if _limit_hit(str(exc)):
                raise StopRun(str(exc)) from exc
            record["pairwise"] = {"error": str(exc)}
        changed = True
    return changed


def regrade(record: dict, q: Query, *, args: argparse.Namespace, config: Settings, indexed: pd.DataFrame,
            band_edges: set[int]) -> None:
    """Recompute every check from the stored response and verdicts. Costs no LLM calls."""
    if not record["checks"].get("schema", {}).get("passed", True):
        return
    response = RecommendationResponse.model_validate(record["response"])
    listed = shortlist(q.prefs, config=config)
    checks = {"schema": check(True, 1, 1)}
    checks |= grade_query(q, response, indexed, candidate_ids=set(record["candidate_ids"]), requested=listed.requested,
                          max_chain_outlets=config.max_chain_outlets, band_edges=band_edges, mode=args.mode,
                          forced_failure=args.force_llm_failure)
    if (record.get("judge") or {}).get("picks"):
        checks |= grade_judge(q, record["judge"])
    if (record.get("pairwise") or {}).get("result"):
        checks |= grade_pairwise(record["pairwise"])
    record["checks"] = checks


def calls_needed(args: argparse.Namespace, repeat: int) -> int:
    ranking = int(args.mode == "llm" and not args.via_api and not args.force_llm_failure)
    return ranking + (int(args.judge) + 2 * int(args.pairwise) if repeat == 0 else 0)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def headline(summary: dict) -> str:
    agg = summary["metrics"]
    if not summary["valid"]:
        return f"INVALID RUN — {agg['M-19']['num']:g} degraded responses"
    if not summary["complete"]:
        return f"INCOMPLETE RUN — {summary['stopped_reason']}"
    failing = [m for m, a in agg.items() if METRIC_BY_ID[m].tier == "Blocking" and a["passed"] is False]
    return f"FAIL — blocking: {', '.join(failing)}" if failing else "PASS — all measured blocking metrics hold"


def render_summary(header: dict, records: list[dict], summary: dict) -> str:
    agg = summary["metrics"]
    first = {r["query_id"]: r for r in records if r.get("repeat", 0) == 0}
    lines = [
        f"# Eval run {header['run_id']} — {headline(summary)}",
        "",
        f"- Mode: `{header['mode']}`{' (forced LLM failure)' if header['forced_failure'] else ''}"
        f"{' via ' + header['via_api'] if header['via_api'] else ''}; model `{header['model']}`; commit `{header['git_commit']}`",
        f"- Queries: {len(first)} of {summary['counts']['selected']} selected; repeats {header['flags']['repeat']}; "
        f"judge {header['flags']['judge']}; pairwise {header['flags']['pairwise']}",
        f"- LLM calls this session: {summary['counts']['llm_calls']}; recorded cost ${summary['counts']['cost_usd']:.4f}",
        f"- Snapshot: weights {header['rank_weights']}, k={header['llm_candidate_k']}, ranking prompt "
        f"`{header['ranking_prompt_sha256'][:12]}`, queries `{header['queries_sha256'][:12]}`, catalog {header['catalog_rows']:,} rows",
        "",
        "## Scorecard",
        "",
        scorecard(agg),
        "",
        "## Failures",
        "",
    ]
    failures = 0
    for metric_id, a in agg.items():
        for qid in a.get("failing", []):
            c = first.get(qid, {}).get("checks", {}).get(metric_id, {})
            lines.append(f"- **{metric_id}** `{qid}`: {c.get('detail') or 'below target'}")
            failures += 1
    schema_errors = [qid for qid, r in first.items() if not r["checks"].get("schema", {}).get("passed", True)]
    lines += [f"- **schema** `{qid}`: {first[qid]['checks']['schema']['detail']}" for qid in schema_errors]
    if not failures and not schema_errors:
        lines.append("None.")
    errors = [f"- `{qid}` {kind}: {r[kind]['error']}" for qid, r in first.items() for kind in ("judge", "pairwise")
              if isinstance(r.get(kind), dict) and "error" in r[kind]]
    if errors:
        lines += ["", "## Judge errors", "", *errors]
    review = [f"- `{qid}` M-09: {r['checks']['M-09']['detail']}" for qid, r in first.items() if r["checks"].get("M-09", {}).get("num")]
    review += [f"- `{qid}` judge ≤ 2: {s['id']} {s}" for qid, r in first.items()
               for s in (r.get("judge") or {}).get("picks", []) if min(s["grounded"], s["preference_specific"], s["concise"]) <= 2]
    review += [f"- `{qid}` ({r['category']}): read the full response" for qid, r in first.items()
               if r["category"] in ("adversarial", "contradictory")]
    lines += ["", "## Manual review (§4.5)", "", *(review or ["Nothing flagged."])]
    return "\n".join(lines) + "\n"


def save_baseline(header: dict, records: list[dict], summary: dict) -> None:
    agg = summary["metrics"]
    BASELINE_PATH.write_text(json.dumps({
        "run_id": header["run_id"],
        "snapshot": header,
        "metrics": {m: agg[m]["value"] for m in ("M-10", "M-11", "M-13", "M-14") if m in agg},
        "queries": {r["query_id"]: {"pick_ids": r["pick_ids"]} for r in records if r.get("repeat", 0) == 0},
    }, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m evals.run_eval", description="Run the eval query set and score it.")
    p.add_argument("--mode", choices=["deterministic", "llm"], default="deterministic")
    p.add_argument("--queries", type=Path, default=QUERIES_PATH)
    p.add_argument("--category")
    p.add_argument("--ids", help="comma-separated query ids, e.g. a live smoke set that fits the daily token budget")
    p.add_argument("--limit", type=int)
    p.add_argument("--judge", action="store_true", help="rubric judge for explanation quality (M-15, M-16)")
    p.add_argument("--pairwise", action="store_true", help="LLM vs deterministic baseline (M-17)")
    p.add_argument("--repeat", type=int, default=1, help="run each query N times (M-18 needs 2)")
    p.add_argument("--via-api", metavar="URL", help="send requests to a running API (phase 4 gate)")
    p.add_argument("--force-llm-failure", action="store_true", help="inject a failing Groq client (M-24)")
    p.add_argument("--check-labels", action="store_true", help="report labels a re-ingest made stale, then exit")
    p.add_argument("--max-calls", type=int, default=150, help="hard cap on LLM calls (ranker + judge) this run")
    p.add_argument("--resume", type=Path, metavar="RUN_JSONL", help="continue an incomplete run")
    p.add_argument("--save-baseline", action="store_true", help=f"write {BASELINE_PATH.name} from a complete deterministic run")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verbose:
        logging.getLogger("src").setLevel(logging.ERROR)
    queries = load_queries(args.queries)
    if args.check_labels:
        problems = check_labels(queries, get_catalog(), get_vocabulary())
        print("\n".join(problems) or f"all labels in {args.queries.name} match the catalog")
        return 1 if problems else 0

    usage_errors = []
    if (args.judge or args.pairwise or args.force_llm_failure) and args.mode != "llm":
        usage_errors.append("--judge, --pairwise and --force-llm-failure need --mode llm")
    if args.force_llm_failure and (args.via_api or args.judge or args.pairwise):
        usage_errors.append("--force-llm-failure runs in-process and without the judge")
    if args.pairwise and not BASELINE_PATH.exists():
        usage_errors.append(f"--pairwise needs {BASELINE_PATH.name}: run --mode deterministic --save-baseline first")
    if args.save_baseline and (args.mode != "deterministic" or args.category or args.limit or args.ids):
        usage_errors.append("--save-baseline needs a full --mode deterministic run")
    wanted = [i.strip() for i in (args.ids or "").split(",") if i.strip()]
    unknown = sorted(set(wanted) - {q.id for q in queries})
    if unknown:
        usage_errors.append(f"unknown query ids: {', '.join(unknown)}")
    config = run_config(args)
    if args.mode == "llm" and not args.force_llm_failure and not args.via_api and not config.llm_enabled:
        usage_errors.append("GROQ_API_KEY is not set: every response would be degraded and the run invalid")
    if usage_errors:
        print("\n".join(f"error: {e}" for e in usage_errors), file=sys.stderr)
        return 2

    selected = [q for q in queries if (not args.category or q.category == args.category)
                and (not wanted or q.id in wanted)][: args.limit]
    catalog = get_catalog()
    indexed = catalog.set_index("restaurant_id", drop=False)
    band_edges = {edge for b in budget_bands(catalog) for edge in (b.min_cost, b.max_cost)}
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8")) if args.pairwise else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.resume:
        header, records, _ = read_run(args.resume)
        current = snapshot(args, config, catalog, header["run_id"])
        drift = [k for k in FINGERPRINT if header.get(k) != current.get(k)]
        if drift:
            print(f"error: can't resume — configuration changed since the run started: {', '.join(drift)}", file=sys.stderr)
            return 2
        header["flags"] = current["flags"]
        path = args.resume
    else:
        run_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}_{args.mode}{'_forced-failure' if args.force_llm_failure else ''}"
        header, records, path = snapshot(args, config, catalog, run_id), [], RESULTS_DIR / f"{run_id}.jsonl"
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
    done = {(r["query_id"], r.get("repeat", 0)): r for r in records}

    call_budget.set_cap(args.max_calls)
    stopped = None
    total = len(selected) * args.repeat
    try:
        with path.open("a", encoding="utf-8") as out:
            for i, (q, repeat) in enumerate(((q, n) for q in selected for n in range(args.repeat)), start=1):
                record = done.get((q.id, repeat))
                fresh = record is None
                if call_budget.used + (calls_needed(args, repeat) if fresh else 0) > args.max_calls:
                    raise StopRun(f"--max-calls {args.max_calls} would be exceeded")
                if fresh:
                    record = run_query(q, repeat, args=args, config=config, indexed=indexed, band_edges=band_edges)
                    done[(q.id, repeat)] = record
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                if repeat == 0 and (args.judge or args.pairwise):
                    if add_judgements(record, q, args=args, config=config, indexed=indexed, baseline=baseline):
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out.flush()
                if fresh or args.judge or args.pairwise:
                    r = record["response"]
                    trace = r.get("trace") or {}
                    print(f"[{i}/{total}] {q.id:<9} {r.get('outcome', 'schema-error'):<18} {len(record['pick_ids'])} picks  "
                          f"{r.get('latency_ms', 0):>6} ms  {trace.get('ranker') or '-':<13}"
                          f"{'  DEGRADED: ' + str(trace.get('fallback_reason')) if r.get('degraded') and args.mode == 'llm' else ''}",
                          flush=True)
    except StopRun as exc:
        stopped = str(exc)
    except KeyboardInterrupt:
        stopped = "interrupted"

    order = {q.id: n for n, q in enumerate(queries)}
    records = sorted(done.values(), key=lambda r: (order.get(r["query_id"], 10**6), r.get("repeat", 0)))
    by_id = {q.id: q for q in queries}
    for record in records:  # a resumed run is scored with today's graders, not the ones it started with
        regrade(record, by_id[record["query_id"]], args=args, config=config, indexed=indexed, band_edges=band_edges)
    agg = aggregate(records)
    cost = sum(((r["response"].get("trace") or {}).get("cost_usd") or 0) + sum(
        (r.get(k) or {}).get("cost_usd", 0) for k in ("judge", "pairwise")) for r in records)
    complete = stopped is None and all((q.id, n) in done for q in selected for n in range(args.repeat))
    valid = not (args.mode == "llm" and not args.force_llm_failure and agg.get("M-19", {}).get("value", 0) > 0)
    summary = {
        "type": "summary",
        "complete": complete,
        "valid": valid,
        "stopped_reason": stopped,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": {"selected": len(selected), "records": len(records), "llm_calls": call_budget.used, "cost_usd": round(cost, 6)},
        "metrics": agg,
    }
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in (header, *records, summary)), encoding="utf-8")
    summary_path = path.with_suffix(".summary.md")
    summary_path.write_text(render_summary(header, records, summary), encoding="utf-8")
    if args.save_baseline and complete:
        save_baseline(header, records, summary)
        print(f"baseline: {BASELINE_PATH}")

    print(f"\n{headline(summary)}\n\n{scorecard(agg)}\n\nresults: {path}\nsummary: {summary_path}")
    if stopped:
        print(f"\nresume with: python -m evals.run_eval {' '.join(a for a in (argv or sys.argv[1:]) if a)} --resume {path}"
              if not args.resume else f"\nresume again with the same command later")
    blocking_ok = all(a["passed"] is not False for m, a in agg.items() if METRIC_BY_ID[m].tier == "Blocking")
    return 0 if complete and valid and blocking_ok else 1


if __name__ == "__main__":
    sys.exit(main())
