"""Response cache (plan 5.1; edge cases O-05, A-06)."""

from __future__ import annotations

import pytest

from src.config import load_settings
from src.core import recommender
from src.core.cache import ResponseCache
from src.core.models import Preferences, RecommendationResponse
from src.data.catalog import build_vocabulary
from src.llm.client import LLMUnavailable
from tests.factories import make_catalog, restaurant

NO_KEY = load_settings(_env_file=None, groq_api_key=None)


def response(summary: str = "x") -> RecommendationResponse:
    return RecommendationResponse(outcome="results", summary=summary, degraded=False)


def test_entries_expire_after_the_ttl():
    now = [0.0]
    cache = ResponseCache(ttl_s=10, max_entries=5, clock=lambda: now[0])
    cache.put("k", response())
    now[0] = 9.9
    assert cache.get("k") is not None
    now[0] = 10.1
    assert cache.get("k") is None and len(cache) == 0


def test_least_recently_used_entry_is_evicted():  # O-05
    cache = ResponseCache(ttl_s=60, max_entries=2)
    cache.put("a", response("a"))
    cache.put("b", response("b"))
    cache.get("a")
    cache.put("c", response("c"))
    assert cache.get("b") is None and cache.get("a").summary == "a" and cache.get("c").summary == "c"


def test_cache_key_ignores_casing_order_and_spacing_but_not_what_changes_the_answer():
    a = Preferences(location="Koramangala", cuisines=["Chinese", "Thai"], free_text="quiet  place")
    b = Preferences(location="koramangala", cuisines=["thai", "CHINESE"], free_text=" quiet place ")
    key = recommender.cache_key(a, 5, True, NO_KEY)
    assert key == recommender.cache_key(b, 5, True, NO_KEY)
    assert key != recommender.cache_key(a, 3, True, NO_KEY)
    assert key != recommender.cache_key(a.model_copy(update={"budget": "low"}), 5, True, NO_KEY)
    assert key != recommender.cache_key(a, 5, True, load_settings(_env_file=None, groq_api_key=None, llm_candidate_k=20))
    assert key != recommender.cache_key(a, 5, True, load_settings(_env_file=None, groq_api_key="gsk_test"))


@pytest.fixture
def process_catalog(monkeypatch):
    """Stands in for the process-wide catalog, which is the only path the cache serves."""
    cat = make_catalog([restaurant(i, votes=1000 - i) for i in range(12)])
    vocab = build_vocabulary(cat)
    monkeypatch.setattr(recommender, "get_catalog", lambda: cat)
    monkeypatch.setattr(recommender, "get_vocabulary", lambda: vocab)
    return cat


@pytest.fixture
def computed(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = recommender._recommend

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(recommender, "_recommend", counting)
    return calls


def test_repeat_request_is_served_from_the_cache(process_catalog, computed):  # A-06
    first = recommender.recommend(Preferences(location="Indiranagar"), config=NO_KEY)
    second = recommender.recommend(Preferences(location="  indiranagar "), config=NO_KEY)
    assert len(computed) == 1
    assert (first.cached, second.cached) == (False, True)
    assert second.recommendations == first.recommendations and second.summary == first.summary


def test_cache_can_be_disabled(process_catalog, computed):
    config = load_settings(_env_file=None, groq_api_key=None, response_cache_enabled=False)
    for _ in range(2):
        assert recommender.recommend(Preferences(), config=config).cached is False
    assert len(computed) == 2


def test_transient_llm_failures_are_not_cached(process_catalog, computed, monkeypatch):
    def rate_limited(*_args, **_kwargs):
        raise LLMUnavailable("Groq rate limit: 8,000 tokens per minute would be exceeded")

    monkeypatch.setattr(recommender, "llm_rank", rate_limited)
    config = load_settings(_env_file=None, groq_api_key="gsk_test")
    for _ in range(2):
        r = recommender.recommend(Preferences(), config=config)
        assert r.degraded and not r.cached
    assert len(computed) == 2 and len(recommender.response_cache) == 0


def test_coverage_errors_are_cached(process_catalog, computed):
    for _ in range(2):
        r = recommender.recommend(Preferences(location="Delhi"), config=NO_KEY)
    assert r.outcome == "coverage_error" and r.cached and len(computed) == 1


def test_injected_catalogs_are_never_cached(computed):
    cat = make_catalog([restaurant(i) for i in range(6)])
    for _ in range(2):
        recommender.recommend(Preferences(), catalog=cat, config=NO_KEY)
    assert len(computed) == 2 and len(recommender.response_cache) == 0
