"""Eval query labels (docs/eval.md §3): record schema, loading, and catalog label checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.models import Budget, Outcome, Preferences
from src.data.catalog import Vocabulary

QUERIES_PATH = Path(__file__).resolve().parent / "queries.jsonl"

Category = Literal[
    "ordinary", "over_constrained", "thin", "contradictory", "free_text",
    "out_of_coverage", "fuzzy", "adversarial", "empty", "nulls",
]
CATEGORY_COUNTS: dict[str, int] = {  # §3.1
    "ordinary": 8, "over_constrained": 4, "thin": 2, "contradictory": 3, "free_text": 4,
    "out_of_coverage": 2, "fuzzy": 2, "adversarial": 3, "empty": 1, "nulls": 1,
}
CONFLICT_CATEGORIES = frozenset({"contradictory", "thin"})  # M-16


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Predicates(_Strict):
    """Checked against a pick's catalog row (M-10). Every predicate that is set must hold."""

    area_in: list[str] | None = None  # an area covers its blocks: "Koramangala" accepts "Koramangala 5th Block"
    cuisines_any: list[str] | None = None
    budget_band_in: list[Budget] | None = None
    rating_gte: float | None = None
    rest_type_any: list[str] | None = None
    book_table: bool | None = None
    online_order: bool | None = None

    def is_empty(self) -> bool:
        return all(v is None for v in self.model_dump().values())


class Signals(_Strict):
    """Catalog-checkable stand-ins for free-text intent (M-14). A pick matches if ANY set signal holds."""

    rest_type_any: list[str] | None = None
    cuisines_any: list[str] | None = None
    book_table: bool | None = None
    online_order: bool | None = None


class RelaxationExpect(_Strict):
    expected: bool
    # The first *ladder* step (step >= 1). The §4.2 budget stretch (step 0) isn't a ladder step and is ignored.
    first_field: Literal["min_rating", "budget", "location", "cuisines"] | None = None


class Expect(_Strict):
    outcome: Outcome
    min_picks: int = Field(0, ge=0)
    acceptable: Predicates = Field(default_factory=Predicates)
    free_text_signals: Signals | None = None
    relaxation: RelaxationExpect
    caveat_required: bool = False
    interpretations: dict[Literal["location", "cuisines"], str] = Field(default_factory=dict)
    gold_ids: list[str] = Field(default_factory=list)
    forbidden_ids: list[str] = Field(default_factory=list)
    forbidden_strings: list[str] = Field(default_factory=list)


class Query(_Strict):
    id: str
    category: Category
    prefs: Preferences
    expect: Expect
    notes: str


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    queries = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            queries.append(Query.model_validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path.name}:{n}: {exc}") from None
    ids = [q.id for q in queries]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"{path.name}: duplicate query ids {duplicates}")
    return queries


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_labels(queries: list[Query], catalog: pd.DataFrame, vocab: Vocabulary) -> list[str]:
    """Labels a re-ingest may have made stale (§3.3 rule 5): unknown IDs, areas or cuisines."""
    ids = set(catalog["restaurant_id"])
    problems = []
    for q in queries:
        e = q.expect
        problems += [f"{q.id}: {rid} is not in the catalog" for rid in e.gold_ids + e.forbidden_ids if rid not in ids]
        problems += [f"{q.id}: area {a!r} is not in the catalog" for a in e.acceptable.area_in or []
                     if a.casefold() not in vocab.areas]
        cuisines = (e.acceptable.cuisines_any or []) + ((e.free_text_signals.cuisines_any or []) if e.free_text_signals else [])
        problems += [f"{q.id}: cuisine {c!r} is not in the catalog" for c in cuisines if c.casefold() not in vocab.cuisines]
    return problems
