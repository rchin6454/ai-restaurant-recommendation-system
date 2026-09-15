# Implementation Plan

Phase-by-phase build plan for the system specified in [problemStatement.md](problemStatement.md) and designed in [architecture.md](architecture.md). Section references like §3.2 point into the architecture doc.

**How to use this:** phases are sequential — each one ends in something you can run and check. Tasks inside a phase are ordered but several are independent (noted). Every task lists the files it touches and a concrete "done when" you can verify, not a feeling of completion.

---

## Plan at a Glance

| Phase | Goal | Output you can run | Est. |
| --- | --- | --- | --- |
| **0** | Scaffolding | `pytest` runs, config loads | 2-3h |
| **1** | Data pipeline | `restaurants.parquet` + a data-quality report | 6-8h |
| **2** | Retrieval engine | CLI: preferences → top-25, no LLM | 6-8h |
| **3** | LLM ranking | CLI: preferences → 5 ranked picks with explanations | 6-8h |
| **4** | API + UI | Browser → recommendations | 5-7h |
| **5** | Hardening + eval | Eval suite green, degraded path proven | 6-8h |
| | | **Total** | **~4-5 focused days** |

Estimates assume one developer familiar with Python and pandas. Phase 1 is the one that usually overruns — real-world data cleaning always has one more surprise.

**Two rules that hold across every phase:**

1. **Phase 2 must be a complete working system without the LLM.** That gives you a reference output to diff against when phase 3 misbehaves, and it *is* the fallback path from §5.6 — you get it for free instead of writing it twice.
2. **Never commit a secret.** `.env` is gitignored from phase 0, before there is anything in it to leak.

---

## Phase 0 — Project Scaffolding

**Goal:** an empty but correct skeleton, so no later phase stops to invent structure.

| # | Task | Files | Notes |
| --- | --- | --- | --- |
| 0.1 | Init repo + `.gitignore` | `.gitignore` | Must include `.env`, `data/raw/`, `data/processed/`, `__pycache__/`, `.venv/` |
| 0.2 | Virtualenv + `pyproject.toml` | `pyproject.toml` | Deps from §9; pin Python ≥3.11 (the `str \| None` syntax is used throughout) |
| 0.3 | Package skeleton | `src/{data,core,llm,api}/__init__.py`, `app/`, `tests/`, `evals/` | Mirror §9 exactly |
| 0.4 | Settings module | `src/config.py` | `pydantic-settings`; every tunable from §10 with its default |
| 0.5 | `.env.example` + local `.env` | `.env.example` | Committed file has `ANTHROPIC_API_KEY=` with an empty value |
| 0.6 | pytest + one smoke test | `tests/test_config.py` | Asserts settings load and defaults match §10 |
| 0.7 | Logging setup | `src/config.py` | Structured logging; one logger per module |

**Done when:** `pytest` passes, and `python -c "from src.config import settings; print(settings.model)"` prints `claude-opus-5`.

**Pitfall:** don't hardcode paths. `catalog_path` comes from settings so tests can point at a fixture Parquet instead of the real one.

---

## Phase 1 — Data Pipeline

**Goal:** the raw 51,717-row Hugging Face dump becomes a clean, deduplicated, enriched Parquet catalog — and you have evidence it's correct.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 1.1 | Download + profile the raw dataset | `notebooks/01_explore.ipynb` (scratch) | 0.2 |
| 1.2 | Per-field cleaning functions | `src/data/cleaning.py` | 1.1 |
| 1.3 | Deduplication | `src/data/cleaning.py` | 1.2 |
| 1.4 | Derived fields | `src/data/cleaning.py` | 1.3 |
| 1.5 | Ingestion script wiring | `src/data/ingest.py` | 1.4 |
| 1.6 | Data-quality report | `src/data/ingest.py` | 1.5 |
| 1.7 | Catalog loader + vocabularies | `src/data/catalog.py` | 1.5 |
| 1.8 | Cleaning unit tests | `tests/test_cleaning.py` | 1.2-1.4 |
| 1.9 | Catalog quality checks | `evals/check_catalog.py` | 1.5 |

### 1.1 Profile first, clean second

Before writing a cleaning rule, look at the data:

```python
from datasets import load_dataset
df = load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train").to_pandas()
df["rate"].value_counts(dropna=False).head(30)      # find every junk form
df["approx_cost(for two people)"].unique()[:50]
df.isna().mean().sort_values(ascending=False)       # null rates per column
df.duplicated(subset=["name", "address"]).sum()     # confirm the dup scale
```

Write down what you find. The §3.2 table lists the known offenders (`"NEW"`, `"-"`, `"1,200"`), but this dataset will have at least one the table doesn't name — find it now rather than in phase 3.

### 1.2 Cleaning functions

One pure function per field in `cleaning.py`, each taking and returning a Series. Pure functions are trivially testable and each becomes one test case in 1.8.

```python
def clean_rate(s: pd.Series) -> pd.Series:          # "4.1/5" → 4.1; "NEW"/"-"/NaN → None
def clean_cost(s: pd.Series) -> pd.Series:          # "1,200" → 1200 (Int64)
def split_list_field(s: pd.Series) -> pd.Series:    # "A, B" → ["A", "B"]
def normalize_text(s: pd.Series) -> pd.Series:      # ftfy + strip + casefold
```

**Critical:** `clean_rate` returns `None` for unrated rows and sets an `is_unrated` flag. Do not impute a mean — §3.2 explains why, and it is the kind of mistake that stays invisible until your "4.5+ only" filter quietly returns unrated restaurants.

### 1.3 Deduplication

Collapse on `(name_norm, address_norm)`, keep the max-`votes` row, aggregate `listed_in(type)` into `listed_types: list[str]`. Print before/after counts. Expect ~51.7K → ~12-13K; if you see 51.7K → 51.6K, your normalization isn't matching and the dedup silently did nothing.

### 1.4 Derived fields

`restaurant_id` (stable hash — same input must give the same ID across runs, so use `hashlib`, **not** Python's `hash()`, which is salted per process), `budget_band` (tertile cut), `bayesian_rating`, `popularity_pct`, `text_blob`.

### 1.6 Data-quality report

`ingest.py` prints, every run: raw row count → deduped count, null rate per retained column, rating distribution, the three budget-band boundaries in rupees, top-20 areas and cuisines by count. This is how you check phase 1 — eyeballing `df.head()` is not verification.

### 1.7 Catalog loader

`get_catalog()` with `@lru_cache(maxsize=1)`, plus `get_area_vocabulary()` and `get_cuisine_vocabulary()` — the sorted distinct values the UI dropdowns and the fuzzy matcher both need.

### 1.9 Catalog quality checks

The 1.6 report is for reading; `check_catalog.py` is for asserting. It turns the phase 1 gate in [eval.md](eval.md) §2 into checks that exit non-zero on failure: dedup ratio 0.20-0.30, `restaurant_id` unique and stable across two ingests, zero imputed ratings, zero unparsed `rate`/cost values, monotonic budget bands each holding 25-40% of rows, no empty vocabulary entries, `phone` absent. Rerun it after every ingest — it is also what tells you the eval labels may need re-checking.

**Done when:**
- `python -m src.data.ingest` writes `data/processed/restaurants.parquet` and prints the quality report
- Deduped row count is in the expected ~12-13K range
- No column has an unexplained null rate above ~60%
- Budget-band boundaries look like real rupee amounts
- `pytest tests/test_cleaning.py` passes, covering every junk value found in 1.1
- `python -m evals.check_catalog` passes

**Decision to lock here:** Bangalore-only (§3.1, §15). Once the catalog is built, confirm `location` really does hold only Bengaluru areas and write the answer into the README so phase 4 labels the UI field honestly.

---

## Phase 2 — Retrieval Engine (no LLM)

**Goal:** a complete, demonstrable recommender with zero API calls. This is the fallback path and the reference implementation.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 2.1 | Pydantic models | `src/core/models.py` | 0.4 |
| 2.2 | Preference normalization (fuzzy match) | `src/core/filters.py` | 1.7, 2.1 |
| 2.3 | Filter chain | `src/core/filters.py` | 2.2 |
| 2.4 | Relaxation ladder | `src/core/filters.py` | 2.3 |
| 2.5 | Pre-ranking score + diversity trim | `src/core/ranking.py` | 2.3 |
| 2.6 | Deterministic ranker (template explanations) | `src/core/ranking.py` | 2.5 |
| 2.7 | CLI harness | `src/cli.py` | 2.1-2.6 |
| 2.8 | Tests | `tests/test_filters.py`, `tests/test_ranking.py` | 2.3-2.6 |

### 2.1 Models

`Preferences` (§4.1), `Pick`, `Recommendation`, `RecommendationResponse` (§7 shape), `Relaxation`. Define the full response shape now, including `applied_filters`, `relaxations`, `degraded`, `candidates_considered` — retrofitting transparency fields after the UI exists means touching every layer twice.

### 2.2 Normalization

`rapidfuzz` against the area/cuisine vocabularies at ≥85 ratio, so "koramangla" and "north indian" both resolve. Return the matched canonical value *and* record it — the response should be able to say "interpreted 'koramangla' as 'Koramangala'".

### 2.3-2.4 Filters and relaxation

Build each predicate as a separate function returning a boolean mask, then `AND` them. Separate masks make it possible to report *which* constraint eliminated everything.

The relaxation ladder (rating → budget → location → cuisine) loops until `len(pool) >= min_candidates` or steps are exhausted, appending a `Relaxation` record each step. **Test the order explicitly** — a ladder that drops cuisine first is a different product, and nothing will fail loudly if you get it backwards.

### 2.5-2.6 Ranking

Implement the §4.4 weighted score with weights read from config, then diversity trimming (max 2 outlets per chain — detect chains by normalized name). `deterministic_rank()` produces template explanations: `"4.3★ from 512 votes · North Indian · ₹800 for two — matches your budget and cuisine."`

### 2.7 CLI harness

```bash
python -m src.cli --location Koramangala --budget medium --cuisine "North Indian" --min-rating 4.0
```

Prints the top 5 with the template explanations, plus applied filters and relaxations. This is your phase-2 demo and, later, your A/B reference against LLM output.

**Done when:**
- The CLI returns sensible results for 5 hand-picked preference sets
- An over-constrained query (`min_rating=4.9, budget=low, location=<small area>`) triggers relaxation and reports it
- An impossible query returns a clean empty response naming the blocking constraint
- No result list contains the same restaurant twice, or more than 2 outlets of one chain
- `pytest` passes; filter and ladder-order tests included

---

## Phase 3 — LLM Ranking Layer

**Goal:** replace `deterministic_rank` with Claude, keeping every fact grounded in the catalog.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 3.1 | Anthropic client wrapper | `src/llm/client.py` | 0.4 |
| 3.2 | System prompt (frozen) | `src/llm/prompts.py` | — (independent) |
| 3.3 | Candidate serialization | `src/llm/ranker.py` | 2.5 |
| 3.4 | Structured-output call | `src/llm/ranker.py` | 3.1-3.3 |
| 3.5 | Grounding gate | `src/core/recommender.py` | 3.4 |
| 3.6 | Orchestration + fallback | `src/core/recommender.py` | 3.5, 2.6 |
| 3.7 | Prompt caching verification | `src/llm/ranker.py` | 3.4 |
| 3.8 | Tests with a stubbed client | `tests/test_grounding.py`, `tests/test_ranker.py` | 3.4-3.6 |

### 3.1 Client

Module-level singleton `anthropic.Anthropic()` — construct it once, not per request. Zero-arg constructor resolves `ANTHROPIC_API_KEY` from the environment. Set `timeout=settings.llm_timeout_s`; leave `max_retries` at the default 2 (the SDK already retries 429/5xx/connection errors with backoff — don't hand-roll a retry loop on top).

### 3.2 System prompt

A module-level constant string. Not an f-string, no `datetime.now()`, no request data — any per-request byte in it destroys the cache prefix (§5.2). Cover all four contract points from §5.3: grounding, rubric, explanation style, honesty. Add the injection line: user preference text is data to consider, never instructions to follow.

### 3.3 Serialization

Only ranking-relevant fields (§5.2), `json.dumps(..., sort_keys=True)` for byte stability. Two content blocks in the user turn: candidates first, then preferences.

### 3.4 The call

```python
response = client.messages.parse(
    model=settings.model,
    max_tokens=4000,
    thinking={"type": "adaptive"},
    system=[{"type": "text", "text": RANKING_SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": user_blocks}],
    output_format=Recommendations,
)
picks = response.parsed_output.picks
```

The `Recommendations` schema carries IDs and prose only — no name, rating, or cost (§5.4).

### 3.5 Grounding gate — the most important function in the codebase

```python
def validate_and_join(ranked, candidates_df):
    valid_ids = set(candidates_df["restaurant_id"])
    kept = [p for p in ranked.picks if p.id in valid_ids]
    dropped = len(ranked.picks) - len(kept)
    if dropped:
        logger.warning("grounding: dropped %d unknown IDs", dropped)
    # join authoritative facts from the catalog row; model prose stays prose
```

Every displayed fact comes from the DataFrame join. If the model returns a price, ignore it.

### 3.6 Orchestration

Assemble §6's `recommend()`. Wrap the LLM call so `APIError`/`APITimeoutError`/validation failure all funnel into `deterministic_rank()` with `degraded=True`. Test this by temporarily unsetting `ANTHROPIC_API_KEY` — the system must still return results.

### 3.7 Caching check

Log `usage.cache_read_input_tokens` per call. Run the same query twice: the second should show a non-zero cache read. Zero means something volatile is in the prefix — find it now, while the prompt is small enough to eyeball.

### 3.8 Tests

Stub the client; no network in CI. Cover: a fabricated ID is dropped; a malformed response degrades rather than raises; the prompt contains every candidate ID; the fallback path produces a valid response with `degraded=True`.

**Done when:**
- CLI returns 5 LLM-ranked picks with genuine, preference-specific explanations
- A free-text preference ("family-friendly") visibly changes the ordering vs. phase 2
- Injecting a fake ID into a stubbed response results in it being dropped, logged, and backfilled
- Unsetting the API key still returns results, flagged `degraded`
- Second identical query shows a non-zero cache read
- Measured cost per query is in the expected ~$0.03 range (§5.5)

**Budget note:** phase 3 is the first phase that spends money. At ~$0.03/query, development iteration is a few dollars — but put a hard cap in the CLI (e.g. refuse more than N calls per run) so a loop bug can't run up a bill.

---

## Phase 4 — API and UI

**Goal:** end-to-end from a browser.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 4.1 | FastAPI app + `/health` | `src/api/main.py` | 3.6 |
| 4.2 | `POST /recommend` | `src/api/routes.py` | 4.1 |
| 4.3 | `/meta/locations`, `/meta/cuisines` | `src/api/routes.py` | 1.7 |
| 4.4 | Error envelope + request logging | `src/api/main.py` | 4.2 |
| 4.5 | Streamlit form | `app/streamlit_app.py` | 4.3 |
| 4.6 | Result cards | `app/streamlit_app.py` | 4.2 |
| 4.7 | Transparency panel + states | `app/streamlit_app.py` | 4.6 |
| 4.8 | API tests | `tests/test_api.py` | 4.2-4.3 |

**4.1** — load the catalog at startup (FastAPI lifespan), not on first request; a 0.5s cold start on a user's first query is avoidable. `/health` reports catalog loaded, row count, and model configured.

**4.2** — thin: validate → `recommend()` → return. All logic stays in `core/`. Never leak a stack trace; return the §7 error envelope.

**4.5-4.7** — sidebar form (area dropdown from `/meta/locations`, budget radio showing the real rupee bands from phase 1, cuisine multiselect, rating slider, free-text box), result cards with the AI explanation **visually distinct** from the catalog facts (§8), a "why these results" expander showing applied filters and relaxations, a spinner during the 2-4s call, and a quiet banner when `degraded` is true.

Label the location field **"Area (Bengaluru)"** — this is where the §3.1 dataset-coverage decision becomes visible to the user.

**Done when:**
- `uvicorn src.api.main:app --reload` + `streamlit run app/streamlit_app.py`, and a full journey works in the browser
- Every field required by the problem statement's §5 output spec appears on the card: name, cuisine, rating, estimated cost, AI explanation
- No-results and degraded states both render correctly (force them to check)
- `pytest tests/test_api.py` passes with a stubbed recommender

---

## Phase 5 — Hardening and Evaluation

**Goal:** prove it behaves under failure, and get a measurement you can tune against.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 5.1 | Response cache | `src/core/recommender.py` | 3.6 |
| 5.2 | Rate limiting | `src/api/main.py` | 4.1 |
| 5.3 | Input hardening | `src/core/models.py` | 2.1 |
| 5.4 | Eval query set (~30) | `evals/queries.jsonl` | 3.6 |
| 5.5 | Eval runner + metrics + run comparison | `evals/run_eval.py`, `evals/compare.py` | 5.4 |
| 5.9 | LLM judge prompts + calibration | `evals/judge_prompts.py` | 5.5 |
| 5.6 | Weight/prompt tuning against the eval | `src/config.py`, `src/llm/prompts.py`, `evals/results/CHANGELOG.md` | 5.5, 5.9 |
| 5.7 | README + run instructions | `README.md` | all |
| 5.8 | Failure-mode drill | — | 5.1-5.3 |

**5.1** — exact-match cache keyed on normalized preferences, ~1h TTL. The catalog is static, so repeat queries are genuinely free.

**5.3** — `free_text` length cap (~500 chars), `min_rating` bounds, budget enum. Cheap, and it closes the obvious abuse paths.

**5.4** — ~30 labelled preference sets covering: ordinary queries, over-constrained (relaxation must fire), single-match, contradictory ("cheap" + "fine dining"), free-text-only, and unknown location. Hand-label what an acceptable answer looks like.

**5.5** — per run, report: **grounding violations (must be 0)**, constraint satisfaction rate, explanation quality (an LLM-judge pass or manual spot-check), p95 latency, cost per query. Write results to a timestamped file so runs are comparable. `compare.py` diffs two result files: per-metric deltas plus per-query pass→fail flips, and refuses runs marked invalid. Full metric definitions and runner flags are in [eval.md](eval.md) §1 and §5.

**5.9** — the judge's system prompt as a frozen module-level constant (same cache rule as 3.2), with a Pydantic score schema for `messages.parse()`: one rubric prompt for per-pick explanation quality, one pairwise prompt for LLM-vs-deterministic comparison. Calibrate before trusting it: hand-score 15 picks blind, and require ±1 agreement on ≥80% and every ungrounded pick caught ([eval.md](eval.md) §4.3).

**5.6** — only now tune the §4.4 weights and the prompt, measuring each change. Tuning before the eval exists is guesswork (§11). Record each accepted change with its run ID in `evals/results/CHANGELOG.md`, so any result can be traced to the change that produced it.

**5.8** — deliberately break things and confirm graceful behavior: unset the API key, point `catalog_path` at a missing file, send a 10K-character `free_text`, send an unknown location, simulate a timeout.

**Done when:**
- Eval runs clean: 0 grounding violations, constraint satisfaction above your chosen bar
- Every failure drill degrades gracefully with a clear user-facing message
- README lets someone clone the repo and reach a working UI following only its instructions

---

## Dependency Graph

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2 ──▶ Phase 3 ──▶ Phase 4 ──▶ Phase 5
                            │           │
                            └───────────┴──▶ 2.6 deterministic_rank is
                                              reused as 3.6's fallback
```

Genuinely parallelizable if more than one person is building: **3.2** (system prompt) needs nothing but the architecture doc; **5.4** (eval queries) can be written during phase 2; **4.5-4.7** (Streamlit) can be built against a stubbed API while phase 3 is in progress.

---

## Risk Register

| Risk | Phase | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Dedup doesn't match → duplicates in every result | 1 | High | Print before/after counts (1.3); assert the reduction in a test |
| Silent rating imputation corrupts filters | 1 | Medium | `None` + `is_unrated` flag; test asserts no imputation |
| Filters return empty for reasonable queries | 2 | High | Relaxation ladder, built in phase 2 not bolted on later |
| Model invents restaurants or prices | 3 | Medium | Structural grounding gate (3.5) + ID-only schema |
| Prompt cache never hits | 3 | Medium | Log `cache_read_input_tokens` from the first call (3.7) |
| Dataset is Bangalore-only, UI implies otherwise | 4 | Certain | Label the field "Area (Bengaluru)"; decided in phase 1 |
| Dev iteration burns budget | 3-5 | Low | Per-run call cap; response cache; batch eval runs |
| Prompt tuning becomes guesswork | 5 | High | Eval set before tuning, not after |

---

## Definition of Done (whole project)

The problem statement's five workflow stages, each verifiable:

- [ ] **Data ingestion** — HF dataset loaded, cleaned, deduped, persisted; quality report reproducible
- [ ] **User input** — location, budget, cuisine, min rating, and free-text preferences all collected and all affecting results
- [ ] **Integration layer** — filtered candidates serialized into a cache-efficient prompt that instructs the model to reason and rank
- [ ] **Recommendation engine** — LLM ranks, explains per pick, and summarizes; output validated against the candidate set
- [ ] **Output display** — name, cuisine, rating, estimated cost, and AI explanation on every card

Plus the engineering bar: tests pass, eval shows 0 grounding violations, the system degrades gracefully without the LLM, and no secret is committed.
