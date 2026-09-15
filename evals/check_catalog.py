"""Phase 1 gate (docs/eval.md §2): assert catalog quality and exit non-zero on any failure.

    python -m evals.check_catalog                   # full gate, incl. a second ingest in a fresh process
    python -m evals.check_catalog --skip-reingest   # everything except ID stability (fast)

Rerun after every ingest: a changed row count means the eval gold IDs need re-checking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.data import cleaning
from src.data.catalog import CATALOG_COLUMNS, CatalogError, load_catalog
from src.data.ingest import meta_path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None  # None = skipped
    detail: str


def run_checks(catalog: pd.DataFrame, meta: dict, reingested_ids: pd.Series | None = None) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        checks.append(Check(name, None if passed is None else bool(passed), detail))

    raw_rows = meta["raw_rows"]
    ratio = len(catalog) / raw_rows if raw_rows else 0.0
    add("dedup ratio 0.20-0.30", 0.20 <= ratio <= 0.30, f"{len(catalog):,} / {raw_rows:,} = {ratio:.3f}")
    add("meta sidecar matches catalog", meta.get("rows") == len(catalog), f"meta rows {meta.get('rows')}")

    ids = catalog["restaurant_id"]
    add("restaurant_id unique", ids.is_unique, f"{int(ids.duplicated().sum())} duplicates")
    if reingested_ids is None:
        add("restaurant_id stable across ingests", None, "skipped (--skip-reingest)")
    else:
        same = ids.tolist() == reingested_ids.tolist()
        detail = "identical ID column from a second ingest in a fresh process" if same else (
            f"{len(set(ids) ^ set(reingested_ids))} IDs differ between ingests"
        )
        add("restaurant_id stable across ingests", same, detail)

    imputed = int((catalog["is_unrated"] & catalog["rating"].notna()).sum())
    unflagged = int((~catalog["is_unrated"] & catalog["rating"].isna()).sum())
    add("no rating imputation", imputed == 0 and unflagged == 0, f"unrated-with-rating {imputed}, null-not-flagged {unflagged}")

    rating = catalog["rating"].dropna()
    in_range = bool(rating.between(0, 5).all())
    add(
        "rating parse coverage",
        in_range and not meta["unparsed_rate_values"],
        f"unparsed raw values {meta['unparsed_rate_values'] or 0}; range [{rating.min()}, {rating.max()}]",
    )
    positive = bool((catalog["cost_for_two"].dropna() > 0).all())
    add(
        "cost parse coverage",
        positive and not meta["unparsed_cost_values"],
        f"unparsed raw values {meta['unparsed_cost_values'] or 0}",
    )

    bands = cleaning.describe_budget_bands(catalog["cost_for_two"], catalog["budget_band"])
    ordered = [b["band"] for b in bands] == list(cleaning.BUDGET_BANDS)
    monotonic = ordered and all(lo["max"] < hi["min"] for lo, hi in zip(bands, bands[1:]))
    balanced = all(0.25 <= b["share"] <= 0.40 for b in bands)
    null_aligned = bool((catalog["cost_for_two"].isna() == catalog["budget_band"].isna()).all())
    add(
        "budget bands monotonic, 25-40% each",
        monotonic and balanced and null_aligned,
        "; ".join(f"{b['band']} ₹{b['min']}-{b['max']} {b['share']:.1%}" for b in bands)
        + ("" if null_aligned else "; null cost/band mismatch"),
    )

    problems = (
        _vocab_problems("area", catalog["location"], cleaning.AREA_ALIASES)
        + _vocab_problems("cuisine", catalog["cuisines"].explode(), cleaning.CUISINE_ALIASES)
        + _vocab_problems("rest_type", catalog["rest_type"].explode(), cleaning.REST_TYPE_ALIASES)
    )
    add("vocabulary clean", not problems, "; ".join(problems) or "no empty, case-variant, or unmerged-alias entries")

    add(
        "PII dropped, schema exact",
        "phone" not in catalog.columns and list(catalog.columns) == list(CATALOG_COLUMNS),
        f"{len(catalog.columns)} columns, phone {'present' if 'phone' in catalog.columns else 'absent'}",
    )
    return checks


def _vocab_problems(label: str, values: pd.Series, aliases: dict[str, str]) -> list[str]:
    distinct = {v for v in values if isinstance(v, str)}
    problems = []
    if any(v != v.strip() or not v for v in distinct):
        problems.append(f"{label}: empty or untrimmed entry")
    folded = pd.Series(sorted(distinct), dtype="str").str.casefold()
    if folded.duplicated().any():
        problems.append(f"{label}: case variants {sorted(set(folded[folded.duplicated()]))}")
    stale = sorted(alias for alias, canon in aliases.items() if alias != canon.casefold() and alias in set(folded))
    if stale:
        problems.append(f"{label}: unmerged aliases {stale}")
    return problems


def reingest_ids() -> pd.Series:
    """Ingest again in a separate process with a different hash seed (D-23) and return its IDs."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "restaurants.parquet"
        seed = "1" if os.environ.get("PYTHONHASHSEED") == "0" else "0"
        proc = subprocess.run(
            [sys.executable, "-m", "src.data.ingest", "--output", str(out), "--no-report-file", "--quiet"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"second ingest failed:\n{proc.stderr[-2000:]}")
        return pd.read_parquet(out, columns=["restaurant_id"])["restaurant_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.check_catalog", description="Phase 1 catalog gate.")
    parser.add_argument("--catalog", type=Path, default=settings.catalog_path)
    parser.add_argument("--skip-reingest", action="store_true", help="skip the ID-stability re-ingest (~1 min)")
    args = parser.parse_args(argv)

    try:
        catalog = load_catalog(args.catalog)
        meta = json.loads(meta_path(args.catalog).read_text(encoding="utf-8"))
    except (CatalogError, FileNotFoundError) as exc:
        print(f"FAIL  cannot load catalog: {exc}", file=sys.stderr)
        return 1

    reingested = None
    if not args.skip_reingest:
        print("re-ingesting in a fresh process to check ID stability…", flush=True)
        try:
            reingested = reingest_ids()
        except RuntimeError as exc:
            print(f"FAIL  {exc}", file=sys.stderr)
            return 1

    checks = run_checks(catalog, meta, reingested)
    for check in checks:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[check.passed]
        print(f"{status}  {check.name:<38} {check.detail}")
    print(f"\nMANUAL  coverage: {catalog['location'].nunique()} areas — confirm all are Bengaluru (recorded in README)")

    failed = [c for c in checks if c.passed is False]
    print(f"\n{'PHASE 1 GATE: FAIL' if failed else 'PHASE 1 GATE: PASS'} ({len(failed)} failing)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
