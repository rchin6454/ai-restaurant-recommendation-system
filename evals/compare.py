"""Compare two eval runs (docs/eval.md §5.4, §6.1): metric deltas, then per-query flips.

    python -m evals.compare evals/results/<before>.jsonl evals/results/<after>.jsonl

A +3% average that hides two newly failing adversarial queries is a regression, so the flips matter
more than the deltas. Invalid runs (M-19 > 0) are refused. Exit code 1 when any query flips from pass
to fail.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evals.metrics import METRICS, aggregate, format_value, query_passes
from evals.run_eval import read_run


def load(path: Path) -> tuple[dict, dict[str, dict], dict]:
    header, records, summary = read_run(path)
    if summary is None:  # a run that crashed before writing its summary
        summary = {"metrics": aggregate(records), "valid": True, "complete": False}
    first = {r["query_id"]: r for r in records if r.get("repeat", 0) == 0}
    return header, first, summary


def compare(before_path: Path, after_path: Path) -> tuple[int, str]:
    (hb, before, sb), (ha, after, sa) = load(before_path), load(after_path)
    for path, summary in ((before_path, sb), (after_path, sa)):
        if not summary["valid"]:
            return 2, f"refusing to compare: {path.name} is an INVALID run (degraded responses in an LLM run)"

    lines = [f"# {before_path.name} → {after_path.name}", ""]
    if hb["queries_sha256"] != ha["queries_sha256"]:
        lines.append("> Warning: the query set changed between runs; only shared query ids are compared.")
    for path, summary in ((before_path, sb), (after_path, sa)):
        if not summary.get("complete", True):
            lines.append(f"> Warning: {path.name} is incomplete ({summary.get('stopped_reason')}).")
    changed = [k for k in ("model", "rank_weights", "llm_candidate_k", "ranking_prompt_sha256", "judge_prompt_sha256")
               if hb.get(k) != ha.get(k)]
    lines += [f"Changed: {', '.join(changed) or 'no tracked configuration'}", "", "| Metric | Before | After | Δ |",
              "| --- | --- | --- | --- |"]
    for metric in METRICS:
        a, b = sb["metrics"].get(metric.id), sa["metrics"].get(metric.id)
        if a is None and b is None:
            continue
        delta = ""
        if a is not None and b is not None:
            d = b["value"] - a["value"]
            delta = f"{d:+.1%}" if abs(a["value"]) <= 1 and abs(b["value"]) <= 1 and metric.id not in ("M-01", "M-12", "M-22") else f"{d:+.4g}"
        lines.append(f"| {metric.id} {metric.name} | {format_value(metric.id, a) if a else '—'} | "
                     f"{format_value(metric.id, b) if b else '—'} | {delta} |")

    regressions, fixes = [], []
    for qid in sorted(set(before) & set(after)):
        pb, pa = query_passes(before[qid]), query_passes(after[qid])
        for metric_id in sorted(set(pb) & set(pa)):
            if pb[metric_id] and not pa[metric_id]:
                detail = after[qid]["checks"][metric_id].get("detail") or ""
                regressions.append(f"- `{qid}` {metric_id}: pass → **fail** {detail}")
            elif not pb[metric_id] and pa[metric_id]:
                fixes.append(f"- `{qid}` {metric_id}: fail → pass")
    lines += ["", f"## Pass → fail ({len(regressions)})", "", *(regressions or ["None."]),
              "", f"## Fail → pass ({len(fixes)})", "", *(fixes or ["None."])]
    return (1 if regressions else 0), "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.compare", description="Diff two eval runs.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    code, report = compare(args.before, args.after)
    print(report, file=sys.stderr if code == 2 else sys.stdout)
    return code


if __name__ == "__main__":
    sys.exit(main())
