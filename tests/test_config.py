"""Phase 0 smoke tests: settings load, defaults match architecture §10, ops edge cases O-09..O-18."""

import io
import logging
import os
import re
from pathlib import Path

import pytest

from src.config import (
    PROJECT_ROOT,
    RankWeights,
    Settings,
    SettingsError,
    configure_logging,
    load_settings,
    settings,
)

ENV_VARS = [name.upper() for name in Settings.model_fields]


@pytest.fixture
def clean_env(monkeypatch):
    """Isolate from the developer's shell and .env so defaults are really defaults."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for var in list(os.environ):
        if var.startswith("RANK_WEIGHTS__"):
            monkeypatch.delenv(var)
    return monkeypatch


def make(**overrides) -> Settings:
    return load_settings(_env_file=None, **overrides)


def test_module_settings_loads():
    assert isinstance(settings, Settings)


def test_defaults_match_architecture(clean_env):
    s = make()
    assert s.anthropic_api_key is None
    assert s.model == "claude-opus-5"
    assert s.llm_candidate_k == 25
    assert s.default_top_n == 5
    assert s.min_candidates == 10
    assert s.max_chain_outlets == 2
    assert s.rank_weights == RankWeights(rating=0.45, votes=0.20, cuisine=0.20, budget=0.15)
    assert s.llm_timeout_s == 30
    assert s.enable_semantic_search is False
    assert s.catalog_path == PROJECT_ROOT / "data" / "processed" / "restaurants.parquet"
    assert s.log_level == "INFO"
    assert s.llm_enabled is False


def test_env_overrides(clean_env):
    clean_env.setenv("MIN_CANDIDATES", "7")
    clean_env.setenv("ENABLE_SEMANTIC_SEARCH", "true")
    clean_env.setenv("RANK_WEIGHTS__RATING", "0.55")
    clean_env.setenv("RANK_WEIGHTS__VOTES", "0.10")
    s = make()
    assert s.min_candidates == 7
    assert s.enable_semantic_search is True
    assert s.rank_weights.rating == 0.55
    assert s.rank_weights.budget == 0.15  # untouched nested field keeps its default


def test_empty_api_key_means_degraded_mode(clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "")
    assert make().llm_enabled is False


@pytest.mark.parametrize(
    ("var", "value"),
    [("MIN_CANDIDATES", "ten"), ("LLM_TIMEOUT_S", "-1"), ("LOG_LEVEL", "LOUD"), ("RANK_WEIGHTS__RATING", "0.9")],
)
def test_invalid_values_fail_at_startup_naming_the_setting(clean_env, var, value):  # O-12
    clean_env.setenv(var, value)
    with pytest.raises(SettingsError, match=var.split("__")[0].lower()):
        make()


def test_misspelled_env_file_key_is_rejected(clean_env, tmp_path):  # O-11
    env_file = tmp_path / ".env"
    env_file.write_text("MIN_CANDIDATE=3\n")
    with pytest.raises(SettingsError, match="min_candidate"):
        load_settings(_env_file=env_file)


def test_missing_env_file_is_fine(clean_env, tmp_path):  # O-01
    assert load_settings(_env_file=tmp_path / "does-not-exist.env").model == "claude-opus-5"


def test_secret_never_appears_in_repr_or_errors(clean_env, tmp_path):  # O-18
    fake = "sk-ant-test-SECRET123"
    s = make(anthropic_api_key=fake)
    assert fake not in repr(s) and fake not in str(s.model_dump())
    env_file = tmp_path / ".env"
    env_file.write_text(f"ANTHROPIC_APIKEY={fake}\n")  # misspelled → validation error
    with pytest.raises(SettingsError) as excinfo:
        load_settings(_env_file=env_file)
    assert fake not in str(excinfo.value)


def test_relative_catalog_path_anchored_at_repo_root(clean_env, tmp_path):  # O-13
    clean_env.chdir(tmp_path)
    assert make(catalog_path="fixtures/x.parquet").catalog_path == PROJECT_ROOT / "fixtures" / "x.parquet"
    assert make(catalog_path=tmp_path / "abs.parquet").catalog_path == tmp_path / "abs.parquet"


def test_env_example_keys_match_settings_fields():  # O-11
    text = (PROJECT_ROOT / ".env.example").read_text()
    keys = {m.group(1) for m in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.MULTILINE)}
    top_level = {k.split("__")[0].lower() for k in keys}
    assert top_level == set(Settings.model_fields)
    assert re.search(r"^ANTHROPIC_API_KEY=\s*$", text, re.MULTILINE), "committed key must be empty"


def test_logging_configuration_is_idempotent():  # O-15, O-18
    configure_logging("INFO")
    configure_logging("INFO")
    handlers = logging.getLogger("src").handlers
    assert len(handlers) == 1
    # The handler bound sys.stderr at import time, so capture its stream directly.
    buf = io.StringIO()
    old_stream = handlers[0].setStream(buf)
    try:
        logging.getLogger("src.test").info("hello key=%s", "sk-ant-abc123")
    finally:
        handlers[0].setStream(old_stream)
    out = buf.getvalue()
    assert out.count('"msg"') == 1
    assert "sk-ant-abc123" not in out and "sk-ant-***" in out


def test_python_version_guard():  # O-09
    source = (PROJECT_ROOT / "src" / "__init__.py").read_text()
    assert "(3, 11)" in source
    assert 'requires-python = ">=3.11"' in Path(PROJECT_ROOT / "pyproject.toml").read_text()
