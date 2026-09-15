"""Phase 1 catalog loader, atomic write, and the catalog gate's failure detection (plan 1.7, 1.9)."""

import pandas as pd
import pytest

from evals.check_catalog import run_checks
from src.config import load_settings
from src.data import catalog as catalog_module
from src.data.catalog import CatalogError, load_catalog
from src.data.ingest import build_catalog, meta_path, write_catalog
from tests.test_cleaning import raw_frame


@pytest.fixture
def built(tmp_path):
    raw = raw_frame(
        [
            {"name": "Jalsa", "cuisines": "North Indian, Chinese", "location": "BTM", "votes": 775},
            {"name": "Spice", "address": "2 Church St", "cuisines": "Thai, ", "location": "Church Street"},
            {"name": "Hole", "address": "3 Brigade Rd", "cuisines": None, "location": None, "rate": "NEW"},
        ]
    )
    cat, stats = build_catalog(raw, min_rows=1)
    path = tmp_path / "processed" / "restaurants.parquet"
    write_catalog(cat, stats, path)
    return cat, stats, path


def test_round_trip_restores_lists(built):
    cat, _, path = built
    loaded = load_catalog(path)
    assert loaded["restaurant_id"].tolist() == cat["restaurant_id"].tolist()
    assert all(isinstance(v, list) for v in loaded["cuisines"])
    assert meta_path(path).exists()
    assert not list(path.parent.glob("*.tmp"))


def test_missing_catalog_error_is_actionable(tmp_path):
    with pytest.raises(CatalogError, match="python -m src.data.ingest"):
        load_catalog(tmp_path / "nope.parquet")


def test_schema_drift_is_rejected(built, tmp_path):
    cat, _, _ = built
    path = tmp_path / "old.parquet"
    cat.drop(columns="budget_band").to_parquet(path)
    with pytest.raises(CatalogError, match="budget_band"):
        load_catalog(path)


def test_crash_mid_write_keeps_previous_catalog(built, monkeypatch):  # D-28
    cat, stats, path = built
    before = path.read_bytes()

    def explode(self, target, **kwargs):
        open(target, "wb").write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError):
        write_catalog(cat, stats, path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob("*.tmp"))


def test_vocabularies_are_sorted_distinct_and_non_empty(built, monkeypatch):
    _, _, path = built
    monkeypatch.setattr(catalog_module, "settings", load_settings(_env_file=None, catalog_path=path))
    catalog_module.clear_cache()
    try:
        assert catalog_module.get_area_vocabulary() == ["BTM Layout", "Church Street"]
        assert catalog_module.get_cuisine_vocabulary() == ["Chinese", "North Indian", "Thai"]
        assert catalog_module.get_catalog() is catalog_module.get_catalog()
    finally:
        catalog_module.clear_cache()


def test_gate_flags_imputed_ratings_and_bad_dedup(built):
    cat, stats, path = built
    meta = {"raw_rows": 12, "rows": len(cat), "unparsed_rate_values": [], "unparsed_cost_values": []}
    names = {c.name: c.passed for c in run_checks(cat, meta)}
    assert names["dedup ratio 0.20-0.30"] is True  # 3 / 12
    assert names["no rating imputation"] is True
    assert names["restaurant_id stable across ingests"] is None

    imputed = cat.copy()
    imputed["rating"] = imputed["rating"].fillna(3.5)
    names = {c.name: c.passed for c in run_checks(imputed, {**meta, "raw_rows": 3})}
    assert names["no rating imputation"] is False
    assert names["dedup ratio 0.20-0.30"] is False
