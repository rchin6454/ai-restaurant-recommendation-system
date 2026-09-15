"""Shared fixtures: no test can reach Groq, and no LLM call cap leaks between tests (plan 3.8)."""

import pytest

from src.llm import client


@pytest.fixture(autouse=True)
def _no_llm_network(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("a test tried to build a real Groq client — pass a stub via `client=` / `llm_client=`")

    monkeypatch.setattr(client, "_client", refuse)
    client.call_budget.set_cap(None)
    yield
    client.call_budget.set_cap(None)
