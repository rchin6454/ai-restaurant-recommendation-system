"""Shared fixtures: no test can reach Groq, and no call cap, rate-limit usage or cached response leaks between tests."""

import pytest

from src.core import recommender
from src.llm import client, rate_limit


@pytest.fixture(autouse=True)
def _no_llm_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to build a real Groq client — pass a stub via `client=` / `llm_client=`")

    def no_sleep(seconds):
        raise AssertionError(f"the LLM rate limiter tried to sleep {seconds:.1f} s — use a fake clock or max wait 0")

    monkeypatch.setattr(client, "_client", refuse)
    monkeypatch.setattr(rate_limit, "_sleep", no_sleep)
    client.call_budget.set_cap(None)
    rate_limit.reset_limiters()
    recommender.response_cache.clear()
    yield
    client.call_budget.set_cap(None)
    rate_limit.reset_limiters()
    recommender.response_cache.clear()
