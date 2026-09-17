"""Run comparison and run-file reading (evals/compare.py, evals/run_eval.py; docs/eval.md §5.2-§5.4)."""

from __future__ import annotations

import json

import pytest

from evals.compare import compare
from evals.metrics import aggregate
from evals.run_eval import LLM_WAIT_S, read_run, short_pause


@pytest.mark.parametrize(
    ("reason", "wait"),
    [
        ("Groq API error: RateLimitError (429); LLM calls paused for 3 s", 4.0),  # seen in the first live run
        (f"Groq API error: RateLimitError (429); LLM calls paused for {int(LLM_WAIT_S)} s", LLM_WAIT_S + 1),
        ("Groq API error: RateLimitError (429); LLM calls paused for 3600 s", None),  # the daily limit: stop, resume later
        ("Groq rate limit: 200,000 tokens per day would be exceeded; capacity frees in ~5000 s", None),
        ("Groq API error: APITimeoutError", None),
        (None, None),
    ],
)
def test_short_429_pauses_are_waited_out_and_long_ones_stop_the_run(reason, wait):
    assert short_pause(reason) == wait

HEADER = {"type": "run", "run_id": "r", "queries_sha256": "q", "model": "m", "rank_weights": {}, "llm_candidate_k": 25,
          "ranking_prompt_sha256": "p", "judge_prompt_sha256": None}


def passing(detail: str = "") -> dict:
    return {"passed": True, "num": 1, "den": 1, "detail": detail}


def failing(detail: str) -> dict:
    return {"passed": False, "num": 0, "den": 1, "detail": detail}


def write_run(path, checks_by_query: dict[str, dict], *, valid: bool = True, extra_records=()):
    records = [{"type": "query", "query_id": qid, "category": "ordinary", "repeat": 0, "checks": checks, "pick_ids": [],
                "retrieval_ms": 5.0} for qid, checks in checks_by_query.items()]
    summary = {"type": "summary", "complete": True, "valid": valid, "metrics": aggregate(records)}
    path.write_text("".join(json.dumps(x) + "\n" for x in (HEADER, *records, *extra_records, summary)))
    return path


def test_pass_to_fail_flips_are_reported_and_fail_the_comparison(tmp_path):
    before = write_run(tmp_path / "before.jsonl", {"adv-01": {"M-08": passing()}, "ord-01": {"M-07": failing("wrong outcome")}})
    after = write_run(tmp_path / "after.jsonl", {"adv-01": {"M-08": failing("'Hotel Fictional'")}, "ord-01": {"M-07": passing()}})
    code, report = compare(before, after)
    assert code == 1
    assert "`adv-01` M-08: pass → **fail** 'Hotel Fictional'" in report
    assert "`ord-01` M-07: fail → pass" in report
    assert "| M-08 Injection containment | 100.0% (1/1) | 0.0% (0/1) | -100.0% |" in report


def test_no_flips_is_a_clean_comparison(tmp_path):
    run = {"ord-01": {"M-07": passing()}}
    code, report = compare(write_run(tmp_path / "a.jsonl", run), write_run(tmp_path / "b.jsonl", run))
    assert code == 0 and "## Pass → fail (0)" in report


def test_invalid_runs_are_refused(tmp_path):  # §5.2 rule 2
    good = write_run(tmp_path / "good.jsonl", {"ord-01": {"M-07": passing()}})
    bad = write_run(tmp_path / "bad.jsonl", {"ord-01": {"M-07": passing()}}, valid=False)
    code, report = compare(good, bad)
    assert code == 2 and "INVALID" in report


def test_read_run_keeps_the_latest_record_per_query_for_resume(tmp_path):
    judged = {"type": "query", "query_id": "ord-01", "category": "ordinary", "repeat": 0, "checks": {"M-15": passing()},
              "pick_ids": [], "retrieval_ms": 5.0, "judge": {"picks": []}}
    path = write_run(tmp_path / "run.jsonl", {"ord-01": {"M-07": passing()}}, extra_records=[judged])
    header, records, summary = read_run(path)
    assert header["run_id"] == "r" and summary is not None
    assert len(records) == 1 and records[0]["judge"] == {"picks": []}
