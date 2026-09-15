"""Settings and logging — every tunable from architecture §10.

Usage:
    from src.config import settings, get_logger
    logger = get_logger(__name__)

Every field is overridable by an environment variable of the same name
(case-insensitive) or a `.env` file at the repository root. Nested fields use
`__`, e.g. `RANK_WEIGHTS__RATING=0.5`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class RankWeights(BaseModel):
    """Pre-ranking blend from §4.4. A starting point to tune against the eval set."""

    model_config = {"extra": "forbid", "frozen": True}

    rating: float = Field(0.45, ge=0, le=1)
    votes: float = Field(0.20, ge=0, le=1)
    cuisine: float = Field(0.20, ge=0, le=1)
    budget: float = Field(0.15, ge=0, le=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> RankWeights:
        total = self.rating + self.votes + self.cuisine + self.budget
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"rank weights must sum to 1.0, got {total:.4f}")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        # `ANTHROPIC_API_KEY=` (empty) means "no key" → degraded mode, not an empty-string key.
        env_ignore_empty=True,
        # A misspelled key in .env fails loudly instead of silently using the default (O-11).
        extra="forbid",
        frozen=True,
    )

    # --- §10 tunables ---
    anthropic_api_key: SecretStr | None = None
    model: str = "claude-opus-5"
    llm_candidate_k: PositiveInt = 25
    default_top_n: PositiveInt = 5
    min_candidates: PositiveInt = 10
    max_chain_outlets: PositiveInt = 2
    rank_weights: RankWeights = RankWeights()
    llm_timeout_s: PositiveFloat = 30.0
    enable_semantic_search: bool = False

    # --- paths & ops ---
    catalog_path: Path = Path("data/processed/restaurants.parquet")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("catalog_path")
    @classmethod
    def _resolve_catalog_path(cls, v: Path) -> Path:
        # Relative paths are anchored at the repo root, not the CWD (O-13).
        return v if v.is_absolute() else (PROJECT_ROOT / v).resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, v: object) -> object:
        return v.upper() if isinstance(v, str) else v

    @property
    def llm_enabled(self) -> bool:
        return self.anthropic_api_key is not None


class SettingsError(RuntimeError):
    """Invalid configuration, with input values stripped so secrets can't leak (O-18)."""


def load_settings(**overrides: object) -> Settings:
    """Build a Settings instance, turning validation failures into a readable, redacted error.

    Tests should call this (or `Settings(_env_file=None, ...)`) rather than mutating
    the module-level `settings` (O-16).
    """
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        # Deliberately omit `input` from each error: it may contain an API key.
        lines = [
            f"  {'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise SettingsError(
            "Invalid configuration (check environment variables and .env):\n" + "\n".join(lines)
        ) from None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_SECRET_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")
_HANDLER_NAME = "src-json"
_STD_RECORD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


def redact(text: str) -> str:
    return _SECRET_PATTERN.sub("sk-ant-***", text)


class JsonFormatter(logging.Formatter):
    """One JSON object per line; `extra={...}` fields are included as top-level keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STD_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


def configure_logging(level: str = "INFO") -> None:
    """Attach a JSON handler to the `src` logger. Safe to call repeatedly (O-15)."""
    pkg_logger = logging.getLogger("src")
    pkg_logger.setLevel(level)
    if any(h.get_name() == _HANDLER_NAME for h in pkg_logger.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(JsonFormatter())
    pkg_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """One logger per module: call as `get_logger(__name__)`."""
    return logging.getLogger(name)


settings = load_settings()
configure_logging(settings.log_level)
