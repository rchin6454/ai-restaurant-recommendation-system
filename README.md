# AI-Powered Restaurant Recommendation System

Restaurant recommendations over the Zomato Bangalore dataset, filtered deterministically and ranked and explained by an LLM served by [Groq](https://console.groq.com). See [docs/architecture.md](docs/architecture.md) and [docs/implementation-plan.md](docs/implementation-plan.md).

## Setup

Requires Python 3.11+.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add GROQ_API_KEY; empty is allowed (degraded mode)
```

If `pip` crashes with `invalid literal for int() with base 10: ''` (a Homebrew Python whose `pyexpat` can't load against the system libexpat), use a uv-managed interpreter instead:

```bash
brew install uv
uv venv --python 3.12 --python-preference only-managed .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

## Data pipeline (phase 1)

```bash
python -m src.data.ingest          # download → clean → dedupe → data/processed/restaurants.parquet
python -m evals.check_catalog      # phase-1 gate; add --skip-reingest to skip the ID-stability re-ingest
```

The first ingest downloads ~550 MB from Hugging Face (cached afterwards; set `HF_HUB_OFFLINE=1` to reuse the cache without network). Peak memory is ~0.9 GB. Each run prints a quality report and saves it to `evals/results/catalog_<timestamp>.txt`, plus a `restaurants.meta.json` sidecar the gate reads. A running API keeps the catalog it loaded — restart it after a re-ingest.

**Coverage: Bengaluru only.** The dataset is the Zomato Bangalore dump. All 93 `location` values in the catalog are Bengaluru neighbourhoods, roads, or regions (Koramangala 5th Block, Whitefield, Hosur Road, South Bangalore, …) — confirmed against the 2026-09-15 ingest. The UI labels the field **"Area (Bengaluru)"**; a query for another city is told the dataset doesn't cover it.

Numbers from that ingest, for reference:

| | |
| --- | --- |
| Rows | 51,717 raw → 12,449 restaurants (ratio 0.241) |
| Budget bands (₹ for two) | low ₹40-250 · medium ₹300-469 · high ₹500-6,000 |
| `rating` null 24.2% | unrated (`NEW`, `-`, blank) — never imputed; `is_unrated` flags them |
| `dish_liked` empty 62.8% | missing in the source (54% of raw rows; dedup keeps the most-voted listing) |

Source `BTM`/`HSR` are shown as `BTM Layout`/`HSR Layout`; `listed_in(city)` becomes the `listed_areas` list because it varies between listings of the same restaurant.

## Recommendations without the LLM (phase 2)

```bash
python -m src.cli --location Koramangala --budget medium --cuisine "North Indian" --min-rating 4.0
python -m src.cli --location koramangla --cuisine Chinese --cuisine Thai --online-order --top-n 10
python -m src.cli --location Jakkur --budget low --min-rating 4.9 --json     # over-constrained: relaxation fires
```

Flags: `--location`, `--budget low|medium|high`, `--cuisine` (repeatable, OR-matched), `--min-rating`, `--online-order/--no-online-order`, `--book-table/--no-book-table`, `--party-size`, `--free-text` (weighed only by the LLM ranker), `--top-n`, `--json`.

How a query is answered ([src/core/](src/core/)):

- **Typos and aliases** are fuzzy-matched against the catalog (≥85) and reported: "koramangla" → Koramangala, "HSR" → HSR Layout. An area outside the dataset (e.g. "Delhi") returns a Bengaluru-only coverage error, never Bengaluru results. "Koramangala" covers all its blocks; "Whitefield" includes "ITPL Main Road, Whitefield".
- **Relaxation** runs only while fewer than `MIN_CANDIDATES` restaurants match, in a fixed order: rating −0.3 (floor 3.0) → budget one band → nearby areas → all of Bengaluru → drop cuisine. A requested budget with fewer than 15 exact matches first stretches one band up. Every step is listed in `relaxations` and `caveats`. "Nearby" is derived from Zomato's listing zones, not a hand-written map.
- **Ranking** scores rating, votes, cuisine overlap and budget fit (weights in `RANK_WEIGHTS__*`), but restaurants meeting the whole original request always rank ahead of ones admitted by a relaxation. At most 2 outlets of a chain appear; unrated restaurants rank on the corpus-average rating and are shown as "New — not yet rated".
- **Empty results** name the blocking constraint ("book table = yes — without it there would be 36 matches").

## LLM ranking on Groq (phase 3)

Set `GROQ_API_KEY` in `.env` ([get a key](https://console.groq.com/keys)). The filtered top 25 candidates go to the model, which picks and explains up to 5; every card's facts (name, rating, cost, cuisines) are still joined from the catalog.

```bash
python -m src.cli --location Indiranagar --free-text "family-friendly, quiet" --verbose   # LLM-ranked
python -m src.cli --location Indiranagar --free-text "family-friendly, quiet" --no-llm    # phase-2 reference
MODEL=qwen/qwen3.6-27b python -m src.cli --location Koramangala --cuisine Italian
```

| Model | Why | Price per 1M tokens (in / out) |
| --- | --- | --- |
| `openai/gpt-oss-120b` (default) | Strict JSON schema, automatic prompt caching | $0.15 / $0.60 (≈ $0.002 per query) |
| `qwen/qwen3.6-27b` | Alternative; best-effort JSON schema, validated locally | ≈ $0.60 / $3.00 |

- **Grounding:** IDs the model returns that aren't in the candidate set are dropped, logged and backfilled; duplicates, odd ranks and ID casing are normalized.
- **Degraded mode:** with no key, a Groq error, a truncated or malformed response, or no valid pick, the deterministic ranker answers and the response says `degraded: true` with the reason in `trace.fallback_reason`.
- **Cost and caching:** the CLI header shows tokens, cached tokens and estimated cost. Run the same query twice; the second should report cached tokens. `--max-llm-calls` (default 3) caps paid calls per run.

## Tests

Run from the repository root:

```bash
pytest
```

## Configuration

All tunables live in [src/config.py](src/config.py) and can be overridden by environment variables or `.env` (see [.env.example](.env.example)). Relative paths such as `CATALOG_PATH` resolve against the repository root.
