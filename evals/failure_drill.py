"""Failure-mode drill (plan 5.8): break things on purpose and confirm each one degrades with a clear message.

    python -m evals.failure_drill

Runs in-process against the real catalog. Groq is replaced by clients that fail in each way, so the
drill spends no tokens. Writes `evals/results/drill_<timestamp>.md`; exits 1 if any scenario fails.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import groq
import httpx
from fastapi.testclient import TestClient
from pydantic import SecretStr

from src.api.main import create_app
from src.api.routes import get_recommender
from src.config import PROJECT_ROOT, settings
from src.core.models import Preferences
from src.core.recommender import recommend
from src.data.catalog import CatalogError, load_catalog
from src.llm import rate_limit

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_UNLIMITED = {"llm_rate_limit_max_wait_s": 0, "llm_requests_per_minute": 10**9, "llm_requests_per_day": 10**9,
              "llm_tokens_per_minute": 10**12, "llm_tokens_per_day": 10**12}
NO_KEY = settings.model_copy(update={"groq_api_key": None, "response_cache_enabled": False})
FAKE_KEY = settings.model_copy(update={"groq_api_key": SecretStr("gsk_drill"), "response_cache_enabled": False, **_UNLIMITED})
PREFS = Preferences(location="Indiranagar", free_text="family-friendly")


class FakeGroq:
    """Answers every call with `outcome`: an exception to raise, or a completion to return."""

    def __init__(self, outcome) -> None:
        self.calls = 0

        def create(**_kwargs):
            self.calls += 1
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=3900, completion_tokens=900, prompt_tokens_details=SimpleNamespace(cached_tokens=0)),
    )


def _degrades(result, reason: str) -> tuple[bool, str]:
    t = result.trace
    ok = result.degraded and len(result.recommendations) == 5 and t is not None and reason in (t.fallback_reason or "")
    return ok, f"5 template-ranked picks, `degraded=true`; reason: {t.fallback_reason if t else None}"


# --- scenarios: each returns (passed, what the user sees) --------------------------------------------


def key_unset():
    return _degrades(recommend(PREFS, config=NO_KEY), "GROQ_API_KEY is not set")


def groq_timeout():
    client = FakeGroq(groq.APITimeoutError(request=httpx.Request("POST", GROQ_URL)))
    return _degrades(recommend(PREFS, config=FAKE_KEY, llm_client=client), "APITimeoutError")


def groq_rate_limited():
    limited = groq.RateLimitError("rate limited", body=None,
                                  response=httpx.Response(429, headers={"retry-after": "30"}, request=httpx.Request("POST", GROQ_URL)))
    client = FakeGroq(limited)
    first_ok, first = _degrades(recommend(PREFS, config=FAKE_KEY, llm_client=client), "paused for 30 s")
    second_ok, second = _degrades(recommend(Preferences(location="Jayanagar"), config=FAKE_KEY, llm_client=client), "429")
    return first_ok and second_ok and client.calls == 1, f"{first}; the next request degraded without calling Groq ({client.calls} call total)"


def groq_malformed_output():
    return _degrades(recommend(PREFS, config=FAKE_KEY, llm_client=FakeGroq(_completion("I recommend Toit!"))), "schema validation")


def daily_token_budget_spent():
    config = FAKE_KEY.model_copy(update={"llm_tokens_per_day": 1000})
    client = FakeGroq(_completion("{}"))
    ok, seen = _degrades(recommend(PREFS, config=config, llm_client=client), "tokens per day")
    return ok and client.calls == 0, seen + "; Groq was never called"


def catalog_missing():
    missing = Path("/nonexistent/restaurants.parquet")
    app = create_app(catalog_loader=lambda: load_catalog(missing))
    try:
        with TestClient(app):
            return False, "the API started without a catalog"
    except CatalogError as exc:
        api_message = str(exc)
    cli = subprocess.run([sys.executable, "-m", "src.cli", "--no-llm"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                         env={**os.environ, "CATALOG_PATH": str(missing)})
    ok = "python -m src.data.ingest" in api_message and cli.returncode == 1 and "python -m src.data.ingest" in cli.stderr
    return ok, f"API refuses to start: {api_message} CLI exits {cli.returncode} with the same message"


def _api(**app_kwargs) -> TestClient:
    app = create_app(**app_kwargs)
    app.dependency_overrides[get_recommender] = lambda: (
        lambda prefs, top_n=None, *, use_llm=True: recommend(prefs, top_n, config=NO_KEY, use_llm=use_llm)
    )
    return TestClient(app)


def huge_free_text():
    with _api() as client:
        resp = client.post("/recommend", json={"free_text": "x" * 10_000})
    error = resp.json()["error"]
    detail = "; ".join(f"{d['field']}: {d['message']}" for d in error["details"])
    return resp.status_code == 422 and "free_text" in detail, f"HTTP 422: {error['message']} ({detail})"


def unknown_location():
    with _api() as client:
        body = client.post("/recommend", json={"location": "Delhi", "cuisines": ["Italian"]}).json()
    return body["outcome"] == "coverage_error" and not body["recommendations"] and "Bengaluru only" in body["summary"], body["summary"]


def malformed_request():
    with _api() as client:
        resp = client.post("/recommend", content=b'{"location": ', headers={"content-type": "application/json"})
    return resp.status_code == 422 and "Traceback" not in resp.text, f"HTTP {resp.status_code}: {resp.json()['error']['message']}"


def request_flood():
    with _api(requests_per_minute=3) as client:
        statuses = [client.post("/recommend", json={"location": "Jayanagar"}).status_code for _ in range(4)]
        error = client.post("/recommend", json={}).json()["error"]
    return statuses == [200, 200, 200, 429] and error["retry_after_s"], f"4th request: HTTP 429 — {error['message']}"


def ui_api_down():
    from streamlit.testing.v1 import AppTest

    real = httpx.request

    def refuse(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    httpx.request = refuse
    try:
        at = AppTest.from_file(str(PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=30).run()
    finally:
        httpx.request = real
    message = at.error[0].value if at.error else ""
    return "Can't reach the recommendation service" in message and not at.exception, message


SCENARIOS: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
    ("Groq API key unset", key_unset),
    ("Groq call times out", groq_timeout),
    ("Groq answers 429", groq_rate_limited),
    ("Groq returns output that isn't the schema", groq_malformed_output),
    ("Daily token budget already spent", daily_token_budget_spent),
    ("Catalog file missing", catalog_missing),
    ("10,000-character free_text", huge_free_text),
    ("Unknown location (Delhi)", unknown_location),
    ("Malformed JSON body", malformed_request),
    ("Request flood from one client", request_flood),
    ("UI with the API down", ui_api_down),
]


def main() -> int:
    import logging

    logging.getLogger("src").setLevel(logging.CRITICAL)
    rows, failures = [], 0
    for name, scenario in SCENARIOS:
        rate_limit.reset_limiters()  # a 429 pause in one scenario must not leak into the next
        try:
            passed, seen = scenario()
        except Exception as exc:  # a crash is exactly what the drill looks for
            passed, seen = False, f"crashed: {type(exc).__name__}: {exc}"
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}: {seen}", flush=True)
        rows.append(f"| {name} | {'PASS' if passed else '**FAIL**'} | {seen.replace('|', '/')} |")

    stamp = f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S}"
    report = RESULTS_DIR / f"drill_{stamp}.md"
    report.write_text("\n".join([
        f"# Failure drill {stamp} — {'PASS' if not failures else f'{failures} FAILING'}",
        "",
        "Plan task 5.8, run in-process against the real catalog with failing Groq stand-ins (no tokens spent).",
        "",
        "| Scenario | Result | What the user sees |",
        "| --- | --- | --- |",
        *rows,
    ]) + "\n", encoding="utf-8")
    print(f"\nreport: {report}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
