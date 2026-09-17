# AI-Powered Restaurant Recommendation System

Restaurant recommendations over the Zomato Bangalore dataset, filtered deterministically and ranked and explained by an LLM served by [Groq](https://console.groq.com). See [docs/architecture.md](docs/architecture.md) and [docs/implementation-plan.md](docs/implementation-plan.md).

## Quick start (clean clone → working UI)

1. [Setup](#setup): create the venv, `pip install -e ".[dev]"`, `cp .env.example .env`, and add `GROQ_API_KEY` (optional; without it results are ranked without AI).
2. Build the catalog: `python -m src.data.ingest` (downloads ~550 MB once).
3. Terminal 1: `uvicorn src.api.main:app`
4. Terminal 2: `streamlit run app/streamlit_app.py`, then open http://localhost:8501.

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
- **Rate limits:** LLM calls are paced against the Groq account's limits for the model: `LLM_REQUESTS_PER_MINUTE=30`, `LLM_REQUESTS_PER_DAY=1000`, `LLM_TOKENS_PER_MINUTE=8000`, `LLM_TOKENS_PER_DAY=200000`.
  - **Tokens bind first.** A ranking call is ~6.3K tokens, so about one LLM-ranked answer fits per minute and ~31 per day.
  - **Before each call** the tokens are estimated and checked against all four limits. A call that doesn't fit waits up to `LLM_RATE_LIMIT_MAX_WAIT_S` (10 s) for capacity. Otherwise it's answered by the deterministic ranker with `degraded: true`, and `trace.fallback_reason` says which limit applied.
  - **A 429 from Groq isn't retried.** It pauses LLM calls for its `retry-after`.
  - **Usage is tracked per process.** Running the CLI while the API is up, or several API workers, on one key isn't coordinated; those collisions show up as 429s and degrade the same way.

## API and UI (phase 4)

Run each in its own terminal from the repository root, with the venv active:

```bash
uvicorn src.api.main:app --reload       # API on http://localhost:8000 (interactive docs at /docs)
streamlit run app/streamlit_app.py      # UI on http://localhost:8501
```

| Endpoint | Returns |
| --- | --- |
| `POST /recommend` | Ranked recommendations. The body is the preferences (`location`, `budget`, `cuisines`, `min_rating`, `online_order`, `book_table`, `party_size`, `free_text`) plus optional `top_n` (1-25) and `use_llm` (default `true`) |
| `GET /meta/locations` | Area names for the dropdown |
| `GET /meta/cuisines` | Cuisine names |
| `GET /meta/budgets` | Each budget band's real rupee range and restaurant count |
| `GET /health` | `catalog_loaded`, `rows`, `model`, `llm_configured` (503 until the catalog is loaded) |

```bash
curl -s localhost:8000/recommend -H 'content-type: application/json' \
  -d '{"location": "Indiranagar", "budget": "high", "free_text": "family-friendly, quiet"}'
```

- **Startup:** the API loads the catalog before accepting requests. If the Parquet is missing it refuses to start and prints the ingest command. Restart it after a re-ingest. Each `--workers` process holds its own copy of the catalog.
- **Errors:** every non-2xx response has the same shape, and no stack trace is ever returned; details go to the logs, matched by `request_id` (also sent as the `X-Request-ID` header):
  ```json
  {"error": {"code": "invalid_request", "message": "The request is invalid. See `details`.",
             "details": [{"field": "min_rating", "message": "Input should be less than or equal to 5"}],
             "request_id": "5fde33c1d8b74563"}}
  ```
- **UI:** reads `API_URL` (default `http://localhost:8000`). On each card the catalog facts (rating, cost, cuisines) are shown separately from the explanation. The explanation is labelled "AI explanation" only when the model wrote it, and "Why it matches" when it's a template. A widened search is listed above the results, a no-results page offers to rerun without the blocking filter, and a banner appears when AI ranking is unavailable.

## Hardening and evaluation (phase 5)

- **Response cache:** identical requests within an hour (`RESPONSE_CACHE_TTL_S`) are answered without a new LLM call; responses say `cached: true`. Answers that fell back because of a timeout, a 429 or the rate limiter aren't cached, so they can't pin template explanations.
- **API rate limit:** `POST /recommend` allows `API_REQUESTS_PER_MINUTE` (10) per client address, then returns 429 with `retry_after_s` and a `Retry-After` header. Behind a reverse proxy every user shares one address, so size the limit for that.
- **Input limits:** free text ≤ 500 characters, cuisine names ≤ 60, party size ≤ 100, `LLM_CANDIDATE_K` ≤ 100; control characters are stripped.

The eval ([docs/eval.md](docs/eval.md)) scores 30 labelled queries in [evals/queries.jsonl](evals/queries.jsonl). Results land in `evals/results/`: a JSONL file per run plus a Markdown scorecard.

```bash
python -m evals.run_eval --mode deterministic --save-baseline      # free: all 30 queries, writes the baseline
python -m evals.run_eval --mode llm --force-llm-failure            # free: proves the fallback path (M-24)
python -m evals.run_eval --mode llm --ids ft-01,ft-02,ft-03,ft-04,con-01,thin-01,adv-01 --judge --pairwise   # live smoke
python -m evals.compare evals/results/<before>.jsonl evals/results/<after>.jsonl   # metric deltas + per-query flips
python -m evals.failure_drill                                      # 5.8: breaks things on purpose, no tokens spent
python -m evals.run_eval --check-labels                            # after a re-ingest
```

- **Live runs cost tokens, not much money.** Groq limits the account to 8K tokens/min and 200K/day; a ranking call is ~6.3K and a judge call ~3-4K. A run paces itself to about one ranking call a minute, a full 30-query `--mode llm` run uses about a day's tokens, and the full `--judge --pairwise` run takes ~2.5 days of budget. If a limit stops a run, it's saved as incomplete; rerun with `--resume <file>`.
- **Judge calibration:** before trusting M-15, export a blind sheet, score its 15 picks by hand, then compare: `python -m evals.judge export-sheet --run <llm run> --queries ft-02,con-01,adv-01`, fill in the scores, and `python -m evals.judge calibrate --run <llm run> --sheet evals/results/calibration_sheet.csv`.
- **Tuning log:** accepted weight and prompt changes are recorded, with the run that justified them, in [evals/results/CHANGELOG.md](evals/results/CHANGELOG.md).

## Tests

Run from the repository root:

```bash
pytest
```

## Configuration

All tunables live in [src/config.py](src/config.py) and can be overridden by environment variables or `.env` (see [.env.example](.env.example)). Relative paths such as `CATALOG_PATH` resolve against the repository root.
