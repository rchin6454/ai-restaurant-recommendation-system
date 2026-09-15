"""Hugging Face dump → clean → dedupe → derive → validate → Parquet (architecture §3.2, plan 1.5-1.6).

    python -m src.data.ingest                        # writes CATALOG_PATH and prints the quality report
    python -m src.data.ingest --output x.parquet --no-report-file --quiet

Offline and one-shot: the service only ever reads the Parquet this writes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, get_logger, settings
from src.data import cleaning as c
from src.data.catalog import CATALOG_COLUMNS, LIST_COLUMNS

logger = get_logger(__name__)

DATASET_ID = "ManikaSaini/zomato-restaurant-recommendation"
RAW_CACHE_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "evals" / "results"
MIN_CATALOG_ROWS = 1_000  # D-31: a rule that filters everything out must fail loudly


class IngestError(RuntimeError):
    """Ingest can't produce a trustworthy catalog; the message says what to do."""


@dataclass
class IngestStats:
    raw_rows: int
    dropped_empty_names: int
    deduped_rows: int
    rate_status: dict[str, int]
    unparsed_rate_values: list[str]
    cost_status: dict[str, int]
    unparsed_cost_values: list[str]
    unrecognized_yes_no: int
    budget_edges: list[int]
    rating_prior_mean: float | None
    bayes_vote_prior: int = c.BAYES_VOTE_PRIOR
    dataset: str = DATASET_ID
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def dedup_ratio(self) -> float:
        return self.deduped_rows / self.raw_rows if self.raw_rows else 0.0


def meta_path(catalog_path: Path) -> Path:
    """Sidecar holding the ingest stats the phase-1 gate needs (raw counts, parse outcomes)."""
    return catalog_path.with_name(catalog_path.stem + ".meta.json")


def download_raw() -> pd.DataFrame:
    try:
        from datasets import load_dataset

        ds = load_dataset(DATASET_ID, split="train", cache_dir=str(RAW_CACHE_DIR))
    except Exception as exc:  # offline, rate-limited, repo moved — all end the same way (D-26)
        raise IngestError(
            f"could not load {DATASET_ID} from Hugging Face ({type(exc).__name__}: {exc}). "
            "Check the network; if the dataset was downloaded before, retry with HF_HUB_OFFLINE=1."
        ) from exc
    # Drop the heavy review/menu text before pandas materializes it (D-27).
    ds = ds.remove_columns([col for col in c.HEAVY_COLUMNS if col in ds.column_names])
    return ds.to_pandas()


def build_catalog(raw: pd.DataFrame, *, min_rows: int = MIN_CATALOG_ROWS) -> tuple[pd.DataFrame, IngestStats]:
    """Pure transform from the raw frame to the catalog. Raises IngestError on any gate-breaking state."""
    missing = [col for col in c.RAW_COLUMNS if col not in raw.columns]
    if missing:  # D-29
        raise IngestError(f"raw dataset is missing expected columns {missing}; has the upstream schema changed?")

    rate_status = c.rate_status(raw["rate"])
    cost_status = c.cost_status(raw["approx_cost(for two people)"])
    df = pd.DataFrame(
        {
            "name": c.clean_display_text(raw["name"]),
            "address": c.clean_display_text(raw["address"]),
            "url": raw["url"].astype("str"),
            "location": c.clean_area(raw["location"]),
            "listed_area": c.clean_area(raw["listed_in(city)"]),
            "listed_type": c.clean_display_text(raw["listed_in(type)"]),
            "rating": c.clean_rate(raw["rate"]),
            "is_new": rate_status.eq("new").astype(bool),
            "votes": c.clean_votes(raw["votes"]),
            "cost_for_two": c.clean_cost(raw["approx_cost(for two people)"]),
            "cuisines": c.split_list_field(raw["cuisines"], c.CUISINE_ALIASES),
            "rest_type": c.split_list_field(raw["rest_type"], c.REST_TYPE_ALIASES),
            "dish_liked": c.split_list_field(raw["dish_liked"]),
            "online_order": c.clean_yes_no(raw["online_order"]),
            "book_table": c.clean_yes_no(raw["book_table"]),
        },
        index=raw.index,
    )  # `phone` is never copied across (S-06)

    unrecognized_yes_no = int(
        sum((raw[col].notna() & df[col].isna()).sum() for col in ("online_order", "book_table"))
    )
    empty_name = df["name"].isna()  # D-25: blank after repair
    if empty_name.any():
        logger.warning("dropping rows with an empty name", extra={"rows": int(empty_name.sum())})
    df = df[~empty_name]

    cat = c.deduplicate(df)
    cat["is_unrated"] = cat["rating"].isna()
    cat["name_norm"] = c.normalize_text(cat["name"])
    cat["location_norm"] = cat["location"].str.casefold()
    cat["listed_areas_norm"] = pd.Series([[a.casefold() for a in v] for v in cat["listed_areas"]], dtype=object)
    cat["cuisines_norm"] = pd.Series([[x.casefold() for x in v] for v in cat["cuisines"]], dtype=object)
    edges = c.budget_band_edges(cat["cost_for_two"])
    cat["budget_band"] = c.assign_budget_band(cat["cost_for_two"], edges)
    cat["bayesian_rating"] = c.bayesian_rating(cat["rating"], cat["votes"])
    cat["popularity_pct"] = c.popularity_pct(cat["votes"])
    cat["text_blob"] = c.text_blob(cat["name"], cat["cuisines"], cat["rest_type"], cat["dish_liked"], cat["location"])
    cat = cat[list(CATALOG_COLUMNS)]

    stats = IngestStats(
        raw_rows=len(raw),
        dropped_empty_names=int(empty_name.sum()),
        deduped_rows=len(cat),
        rate_status=_counts(rate_status),
        unparsed_rate_values=_distinct_raw(raw["rate"], rate_status.isin(["unparsed", "out_of_range"])),
        cost_status=_counts(cost_status),
        unparsed_cost_values=_distinct_raw(raw["approx_cost(for two people)"], cost_status.eq("unparsed")),
        unrecognized_yes_no=unrecognized_yes_no,
        budget_edges=edges,
        rating_prior_mean=c.rating_prior_mean(cat["rating"]),
    )
    for label, values in (("rate", stats.unparsed_rate_values), ("cost", stats.unparsed_cost_values)):
        if values:
            logger.warning("unparsed raw values nulled", extra={"field": label, "values": values[:20]})

    if len(cat) < min_rows:  # D-31
        raise IngestError(f"only {len(cat)} restaurants survived cleaning (minimum {min_rows}); a rule is too aggressive")
    if not cat["restaurant_id"].is_unique:  # D-24
        dupes = cat.loc[cat["restaurant_id"].duplicated(), "restaurant_id"].tolist()[:5]
        raise IngestError(f"restaurant_id collision, e.g. {dupes}")
    return cat, stats


def _counts(status: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in status.value_counts().sort_index().items()}


def _distinct_raw(values: pd.Series, mask: pd.Series) -> list[str]:
    return sorted({repr(v) if not isinstance(v, str) else v for v in values[mask]})


def write_catalog(catalog: pd.DataFrame, stats: IngestStats, path: Path) -> None:
    """Temp file + atomic rename: a crash mid-write leaves the previous catalog intact (D-28)."""
    meta = {**asdict(stats), "dedup_ratio": round(stats.dedup_ratio, 4), "rows": len(catalog)}
    _atomic_write(path, lambda tmp: catalog.to_parquet(tmp, index=False))
    _atomic_write(meta_path(path), lambda tmp: tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8"))


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Quality report (1.6)
# ---------------------------------------------------------------------------


def format_report(catalog: pd.DataFrame, stats: IngestStats) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"CATALOG QUALITY REPORT  {stats.generated_at}")
    add(f"source: {stats.dataset}")

    add("\n== Rows ==")
    add(f"raw rows                 {stats.raw_rows:>7,}")
    add(f"dropped (empty name)     {stats.dropped_empty_names:>7,}")
    add(f"deduplicated restaurants {stats.deduped_rows:>7,}   ratio {stats.dedup_ratio:.3f} (expect 0.20-0.30)")

    add("\n== Raw parse outcomes ==")
    add(f"rate: {_fmt_counts(stats.rate_status)}")
    add(f"      unparsed/out-of-range values: {stats.unparsed_rate_values or 'none'}")
    add(f"cost: {_fmt_counts(stats.cost_status)}")
    add(f"      unparsed values: {stats.unparsed_cost_values or 'none'}")
    add(f"online_order/book_table unrecognized: {stats.unrecognized_yes_no}")

    add("\n== Null rate per column (list columns: share of empty lists) ==")
    for col in CATALOG_COLUMNS:
        s = catalog[col]
        rate = s.map(len).eq(0).mean() if col in LIST_COLUMNS else s.isna().mean()
        add(f"{col:<20} {rate:6.1%}")

    add("\n== Ratings ==")
    rated = catalog["rating"].dropna().astype(float)
    add(
        f"rated {len(rated):,} · unrated {int(catalog['is_unrated'].sum()):,} "
        f"(NEW {int(catalog['is_new'].sum()):,}) · imputed 0 (never imputed by construction)"
    )
    if len(rated):
        add(f"min {rated.min():.1f} · mean {rated.mean():.2f} · median {rated.median():.1f} · max {rated.max():.1f}")
        hist = pd.cut(rated, bins=[0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], include_lowest=True).value_counts(sort=False)
        for interval, n in hist.items():
            add(f"  {str(interval):<13} {n:>6,}  {'#' * round(60 * n / len(rated))}")
        add(f"bayesian prior: C = {stats.rating_prior_mean}, m = {stats.bayes_vote_prior} votes")

    add("\n== Budget bands (₹ for two) ==")
    cost = catalog["cost_for_two"].dropna().astype(int)
    add(f"costed {len(cost):,} · null cost {int(catalog['cost_for_two'].isna().sum()):,} (band=None, kept for unbanded queries)")
    if len(cost):
        q = cost.quantile([0.01, 0.5, 0.99])
        add(f"P1 ₹{q[0.01]:,.0f} · median ₹{q[0.5]:,.0f} · P99 ₹{q[0.99]:,.0f} · max ₹{cost.max():,}")
    for b in c.describe_budget_bands(catalog["cost_for_two"], catalog["budget_band"]):
        add(f"  {b['band']:<7} ₹{b['min']:>5,} - ₹{b['max']:>5,}   {b['rows']:>6,} rows  {b['share']:5.1%}")

    areas = catalog["location"].value_counts()
    add(f"\n== Top 20 areas (of {len(areas)}) ==")
    add(_fmt_top(areas.head(20)))
    cuisines = catalog["cuisines"].explode().dropna().value_counts()
    add(f"\n== Top 20 cuisines (of {len(cuisines)}) ==")
    add(_fmt_top(cuisines.head(20)))
    add("\n== Listing types ==")
    add(_fmt_top(catalog["listed_types"].explode().dropna().value_counts()))

    add("\n== Vocabulary review ==")
    add(f"aliases applied — cuisine {c.CUISINE_ALIASES}, rest_type {c.REST_TYPE_ALIASES}, area {c.AREA_ALIASES}")
    add(f"all cuisines: {', '.join(sorted(cuisines.index, key=str.casefold))}")
    add(f"all rest types: {', '.join(sorted(catalog['rest_type'].explode().dropna().unique(), key=str.casefold))}")
    add(f"all areas (confirm Bengaluru-only coverage): {', '.join(sorted(areas.index, key=str.casefold))}")
    return "\n".join(lines)


def _fmt_counts(counts: dict[str, int]) -> str:
    return " · ".join(f"{k} {v:,}" for k, v in counts.items())


def _fmt_top(counts: pd.Series) -> str:
    return "\n".join(f"  {name:<28} {n:>6,}" for name, n in counts.items())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.data.ingest", description="Build the restaurant catalog.")
    parser.add_argument("--output", type=Path, default=settings.catalog_path, help="catalog Parquet path")
    parser.add_argument("--no-report-file", action="store_true", help="don't save the report under evals/results/")
    parser.add_argument("--quiet", action="store_true", help="don't print the report")
    args = parser.parse_args(argv)

    try:
        catalog, stats = build_catalog(download_raw())
        write_catalog(catalog, stats, args.output)
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    report = format_report(catalog, stats)
    if not args.quiet:
        print(report)
    if not args.no_report_file:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report_path = REPORT_DIR / f"catalog_{stamp}.txt"
        report_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nreport saved to {report_path}")
    print(f"wrote {len(catalog):,} restaurants to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
