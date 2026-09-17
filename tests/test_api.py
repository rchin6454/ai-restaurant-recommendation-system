"""API contract (plan 4.8): edge cases A-01…A-12, I-11, I-12, I-14, I-19.

No Parquet file and no network: the app gets a hand-built catalog through its loader, and the
recommender dependency runs the real `recommend()` over that catalog with the LLM unavailable.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import MAX_BODY_BYTES, ClientRateLimiter, create_app
from src.api.routes import CatalogInfo, budget_bands, get_recommender
from src.config import load_settings, settings
from src.core.models import RecommendationResponse
from src.core.recommender import recommend
from src.data.catalog import CatalogError, load_catalog
from tests.factories import make_catalog, restaurant

CONFIG = load_settings(_env_file=None, groq_api_key=None)


@pytest.fixture
def catalog():
    rows = []
    for i in range(12):
        band, cost = [("low", 200), ("medium", 450), ("high", 1200)][i % 3]
        rows.append(
            restaurant(i, location="Koramangala" if i % 2 else "Indiranagar", budget_band=band, cost_for_two=cost,
                       rating=3.6 + i * 0.05, cuisines=["North Indian", "Chinese"] if i % 4 == 0 else ["Café"])
        )
    return make_catalog(rows)


def info_for(df) -> CatalogInfo:
    return CatalogInfo(
        rows=len(df),
        locations=sorted(set(df["location"])),
        cuisines=sorted(set(df["cuisines"].explode())),
        budgets=budget_bands(df),
    )


@pytest.fixture
def calls() -> list:
    return []


def app_with(catalog, calls, recommender=None, **app_kwargs):
    def real(prefs, top_n=None, *, use_llm=True):
        calls.append((prefs, top_n, use_llm))
        return recommend(prefs, top_n, catalog=catalog, config=CONFIG, use_llm=use_llm)

    app = create_app(catalog_loader=lambda: info_for(catalog), **app_kwargs)
    app.dependency_overrides[get_recommender] = lambda: recommender or real
    return app


@pytest.fixture
def client(catalog, calls):
    with TestClient(app_with(catalog, calls), raise_server_exceptions=False) as c:
        yield c


def assert_envelope(resp, status: int, code: str) -> dict:
    assert resp.status_code == status
    error = resp.json()["error"]
    assert error["code"] == code and error["message"]
    assert error["request_id"] == resp.headers["x-request-id"]
    assert "Traceback" not in resp.text
    return error


# --- health and metadata (4.1, 4.3) ------------------------------------------------------------


def test_health_reports_catalog_and_model(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "catalog_loaded": True, "rows": 12, "model": settings.model,
                    "llm_configured": settings.llm_enabled}


def test_meta_endpoints_serve_the_loaded_vocabularies(client):
    assert client.get("/meta/locations").json() == ["Indiranagar", "Koramangala"]
    assert client.get("/meta/cuisines").json() == ["Café", "Chinese", "North Indian"]
    assert client.get("/meta/budgets").json() == [
        {"band": "low", "min_cost": 200, "max_cost": 200, "restaurants": 4},
        {"band": "medium", "min_cost": 450, "max_cost": 450, "restaurants": 4},
        {"band": "high", "min_cost": 1200, "max_cost": 1200, "restaurants": 4},
    ]


def test_budget_bands_skip_rows_without_a_cost():
    df = make_catalog([restaurant(1, budget_band="low", cost_for_two=150), restaurant(2, budget_band="None", cost_for_two=None)])
    assert [b.model_dump() for b in budget_bands(df)] == [{"band": "low", "min_cost": 150, "max_cost": 150, "restaurants": 1}]


def test_catalog_not_loaded_returns_503_not_empty_lists(catalog, calls):  # A-08
    client = TestClient(app_with(catalog, calls))  # no `with`: the lifespan never ran
    for path in ("/meta/locations", "/meta/cuisines", "/meta/budgets"):
        assert_envelope(client.get(path), 503, "unavailable")
    assert_envelope(client.post("/recommend", json={}), 503, "unavailable")
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["status"] == "starting"
    assert calls == []


def test_missing_catalog_aborts_startup_naming_the_fix(tmp_path):  # A-01
    app = create_app(catalog_loader=lambda: load_catalog(tmp_path / "missing.parquet"))
    with pytest.raises(CatalogError, match=r"python -m src\.data\.ingest"):
        with TestClient(app):
            pass


# --- POST /recommend (4.2) ---------------------------------------------------------------------


def test_recommend_passes_preferences_through_and_matches_the_schema(client, calls):
    resp = client.post("/recommend", json={"location": "Koramangala", "cuisines": ["Café"], "top_n": 3})
    assert resp.status_code == 200
    body = RecommendationResponse.model_validate(resp.json())
    prefs, top_n, use_llm = calls[0]
    assert (prefs.location, prefs.cuisines, top_n, use_llm) == ("Koramangala", ["Café"], 3, True)
    assert body.degraded is True and body.trace.ranker == "deterministic"  # no key in CONFIG
    assert 1 <= len(body.recommendations) <= 3
    for card in body.recommendations:  # problem statement §5: every field on every card
        assert card.name and card.cuisines and card.explanation and card.cost_for_two and card.rating


def test_relaxations_serialize_with_the_from_key(client):  # §7 shape
    body = client.post("/recommend", json={"min_rating": 4.2}).json()  # catalog tops out at 4.15★
    assert body["outcome"] == "relaxed_results"
    first = body["relaxations"][0]
    assert first["field"] == "min_rating" and first["from"] == 4.2 and "from_" not in first


def test_use_llm_false_is_honoured(client, calls):
    client.post("/recommend", json={"use_llm": False})
    assert calls[0][2] is False


def test_empty_result_names_the_blocking_constraint(client):  # F-02, U-05
    body = client.post("/recommend", json={"book_table": True}).json()
    assert body["outcome"] == "empty_with_reason" and body["recommendations"] == []
    assert body["blocking_constraints"] == ["book_table"]


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({"min_rating": 7}, "min_rating"),  # I-11
        ({"budget": "cheap"}, "budget"),  # I-12
        ({"free_text": "x" * 501}, "free_text"),  # I-14
        ({"top_n": 0}, "top_n"),  # I-19
        ({"top_n": 26}, "top_n"),
        ({"cuisine": "Thai"}, "cuisine"),  # misspelled field is rejected, not ignored
        ({"cuisines": [1]}, "cuisines.0"),
    ],
)
def test_invalid_preferences_get_a_422_envelope(client, calls, payload, field):
    error = assert_envelope(client.post("/recommend", json=payload), 422, "invalid_request")
    assert field in [d["field"] for d in error["details"]]
    assert calls == []


@pytest.mark.parametrize(
    ("content", "content_type"),
    [(b'{"location": ', "application/json"), (b"location=Koramangala", "text/plain"), (b"[1, 2]", "application/json")],
)
def test_malformed_body_gets_a_422_envelope(client, calls, content, content_type):  # A-03
    assert_envelope(client.post("/recommend", content=content, headers={"content-type": content_type}), 422, "invalid_request")
    assert calls == []


def test_unhandled_error_returns_a_generic_500(catalog, calls, caplog):  # A-04
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom: internal detail gsk_notARealKey")

    with TestClient(app_with(catalog, calls, recommender=explode), raise_server_exceptions=False) as client:
        with caplog.at_level("ERROR", logger="src.api.main"):
            resp = client.post("/recommend", json={})
    error = assert_envelope(resp, 500, "internal_error")
    assert "boom" not in resp.text and "gsk_" not in resp.text
    assert error["details"] == []
    assert any(r.exc_info and "boom" in str(r.exc_info[1]) for r in caplog.records)  # detail stays in the logs


def test_unicode_round_trips(client, calls):  # A-11
    text = "Café vibes, ಬೆಂಗಳೂರು, 😋"
    resp = client.post("/recommend", json={"free_text": text, "cuisines": ["café"]})
    assert resp.status_code == 200
    assert calls[0][0].free_text == text
    assert all("Café" in c["cuisines"] for c in resp.json()["recommendations"])


def test_oversized_body_is_rejected_before_parsing(client, calls):  # A-12
    resp = client.post("/recommend", content=b"x" * (MAX_BODY_BYTES + 1), headers={"content-type": "application/json"})
    assert_envelope(resp, 413, "payload_too_large")
    assert calls == []


def test_unknown_route_uses_the_envelope(client):
    assert_envelope(client.get("/nope"), 404, "not_found")
    assert_envelope(client.get("/recommend"), 405, "method_not_allowed")


def test_recommend_is_rate_limited_per_client(catalog, calls):  # A-06, A-07, S-04
    with TestClient(app_with(catalog, calls, requests_per_minute=2)) as client:
        assert [client.post("/recommend", json={}).status_code for _ in range(2)] == [200, 200]
        resp = client.post("/recommend", json={})
        error = assert_envelope(resp, 429, "rate_limited")
        assert 1 <= error["retry_after_s"] <= 60 and resp.headers["retry-after"] == str(error["retry_after_s"])
        assert len(calls) == 2  # the limited request never reached the recommender
        assert client.get("/meta/locations").status_code == 200  # only /recommend is limited


def test_client_rate_limiter_window_slides():
    now = [0.0]
    limiter = ClientRateLimiter(2, clock=lambda: now[0], max_clients=2)
    assert limiter.check("a") == 0 and limiter.check("a") == 0
    now[0] = 20.0
    assert limiter.check("a") == pytest.approx(40.0)
    assert limiter.check("b") == 0  # clients are independent
    now[0] = 60.5
    assert limiter.check("a") == 0  # the t=0 calls have left the window
    limiter.check("c")  # a third client evicts the least recently seen ("b")
    assert set(limiter._calls) == {"a", "c"}


def test_request_id_is_echoed_only_when_well_formed(client):
    assert client.get("/health", headers={"x-request-id": "ui-1234"}).headers["x-request-id"] == "ui-1234"
    generated = client.get("/health", headers={"x-request-id": "bad id\n<script>"}).headers["x-request-id"]
    assert generated != "bad id\n<script>" and len(generated) == 16
