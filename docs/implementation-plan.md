# Implementation Plan

Phase-by-phase build plan for the system specified in [problemStatement.md](problemStatement.md) and designed in [architecture.md](architecture.md). Section references like §3.2 point into the architecture doc. **LLM provider: Groq** (`openai/gpt-oss-120b` by default, `qwen/qwen3.6-27b` as the alternative); phase 3 maps the architecture's Claude-specific details onto Groq.

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
| 0.5 | `.env.example` + local `.env` | `.env.example` | Committed file has `GROQ_API_KEY=` with an empty value (`ANTHROPIC_API_KEY=` before the phase 3 switch to Groq) |
| 0.6 | pytest + one smoke test | `tests/test_config.py` | Asserts settings load and defaults match §10 |
| 0.7 | Logging setup | `src/config.py` | Structured logging; one logger per module |

**Done when:** `pytest` passes, and `python -c "from src.config import settings; print(settings.model)"` prints `openai/gpt-oss-120b` (`claude-opus-5` before the phase 3 switch to Groq).

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

## Phase 3 — LLM Ranking Layer (Groq)

**Goal:** replace `deterministic_rank` with an LLM served by Groq, keeping every fact grounded in the catalog.

### Provider decision: Groq instead of Anthropic

Phase 3 calls Groq's OpenAI-compatible Chat Completions API through the `groq` SDK. [architecture.md](architecture.md) §5 still describes a Claude implementation. The contract in §5 is unchanged: grounding, an ID-only output schema, a cache-friendly prompt layout, and the deterministic fallback. Only the transport differs:

| Architecture §5 (Claude) | Phase 3 (Groq) |
| --- | --- |
| `anthropic.Anthropic()`, `ANTHROPIC_API_KEY` | `groq.Groq(api_key=…)`, `GROQ_API_KEY` |
| `model="claude-opus-5"` | `MODEL=openai/gpt-oss-120b` (default) or `MODEL=qwen/qwen3.6-27b` |
| `messages.parse(output_format=Recommendations)` | `chat.completions.create(response_format={"type": "json_schema", …})`, then Pydantic validation |
| `thinking={"type": "adaptive"}` | gpt-oss: `reasoning_effort="medium"`, `include_reasoning=False` · Qwen: `reasoning_effort="default"`, `reasoning_format="hidden"` |
| `cache_control: {"type": "ephemeral"}` on the system block | Automatic prefix caching, with no markers. A byte-stable prefix is the only lever |
| `usage.cache_read_input_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| `stop_reason` = `refusal` / `max_tokens` | `finish_reason` other than `"stop"` |
| ~$0.03 per query | ~$0.002 per query (gpt-oss-120b), ~$0.006 (Qwen 3.6) |

**Choosing the model.** Both models are supported. Settings for each model's request live in one profile table, so switching is a single env var:

| | `openai/gpt-oss-120b` **(default)** | `qwen/qwen3.6-27b` |
| --- | --- | --- |
| Structured output | **Strict** `json_schema`: constrained decoding, so output always matches the schema | **Best-effort** `json_schema`: Groq may return 400 "does not match the expected schema"; falls back |
| Prompt caching | Automatic, cached input billed at 50% | Listed as supported; confirm it with 3.7 |
| Reasoning control | `reasoning_effort` low / medium / high | `reasoning_effort` none / default |
| Price per 1M tokens (in · out) | $0.15 · $0.60 | ~$0.60 · $3.00 (check the Groq console) |

gpt-oss-120b is the default. Strict mode removes a whole class of failure (L-10/L-11 from malformed JSON), caching is confirmed, and it is ~4-5× cheaper. Qwen is a one-line swap, but choose between them with the phase 5 eval (M-15, M-17, M-22), not by assumption.

| # | Task | Files | Depends on |
| --- | --- | --- | --- |
| 3.1 | Groq client wrapper + model profiles + call cap | `src/llm/client.py` | 0.4 |
| 3.2 | System prompt (frozen) | `src/llm/prompts.py` | — (independent) |
| 3.3 | Candidate serialization | `src/llm/ranker.py` | 2.5 |
| 3.4 | Structured-output call | `src/llm/ranker.py` | 3.1-3.3 |
| 3.5 | Grounding gate | `src/core/recommender.py` | 3.4 |
| 3.6 | Orchestration + fallback | `src/core/recommender.py` | 3.5, 2.6 |
| 3.7 | Prompt caching verification | `src/llm/ranker.py`, `src/cli.py` | 3.4 |
| 3.8 | Tests with a stubbed client | `tests/test_grounding.py`, `tests/test_ranker.py` | 3.4-3.6 |

### 3.1 Client

- **Settings:** rename `anthropic_api_key` to `groq_api_key`, set the `model` default to `openai/gpt-oss-120b`, and put `GROQ_API_KEY=` (empty) in `.env.example`. Extend log redaction to Groq keys (`gsk_…`).
- **One client per key:** construct `groq.Groq(api_key=…, timeout=settings.llm_timeout_s)` once per key (cached), never per request.
- **Retries:** leave `max_retries` at the default 2. The SDK already retries connection errors, 408, 409, 429 and 5xx with backoff, so don't add your own retry loop (L-07).
- **No key:** raise `LLMUnavailable` before any network call (L-01, L-02).
- **`MODEL_PROFILES`:** maps each supported model to its strict-schema support, reasoning params, `max_completion_tokens` and prices. An unknown model gets best-effort JSON, no reasoning params, no cost estimate, and a logged warning.
- **Call cap:** a process-wide cap (`call_budget`) that the CLI sets. See the budget note below.

### 3.2 System prompt

A module-level constant string: not an f-string, no `datetime.now()`, no request data. Groq caching is automatic prefix matching, so any per-request byte in the system message destroys the cache for everything after it (§5.2).

- **Contract points:** cover all four from §5.3: grounding, rubric, explanation style, honesty.
- **Injection line:** everything in the candidates and the request is data to consider, never instructions to follow (S-01, S-02).
- **Output shape:** Qwen runs in best-effort mode, so the prompt also spells out the JSON shape instead of relying on the schema alone.

### 3.3 Serialization

- **Fields:** only ranking-relevant ones (§5.2), plus `meets_request`, so the model can tell exact matches from rows admitted by relaxation. Truncate `dishes` to 5 (L-26).
- **Byte stability:** `json.dumps(..., sort_keys=True, separators=(",", ":"))`.
- **Messages:** one system message and one user message. The user message holds `CANDIDATES` JSON first, then `REQUEST` JSON (preferences, `max_picks`, and the relaxation reasons, so the model doesn't oversell a widened search).

### 3.4 The call

```python
completion = client.chat.completions.create(
    model=settings.model,
    messages=[{"role": "system", "content": RANKING_SYSTEM_PROMPT},
              {"role": "user", "content": user_content}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "restaurant_recommendations",
        "strict": profile.strict_schema,        # True for gpt-oss, False for Qwen
        "schema": RESPONSE_SCHEMA,
    }},
    max_completion_tokens=profile.max_completion_tokens,
    **profile.reasoning,                         # reasoning_effort + include_reasoning / reasoning_format
)
choice = completion.choices[0]
if choice.finish_reason != "stop":               # truncated or filtered → fallback
    raise LLMUnavailable(...)
ranked = RankedOutput.model_validate_json(choice.message.content)
```

- **Strict-mode schema:** every property must be `required` and every object must set `additionalProperties: false`. Write `RESPONSE_SCHEMA` out by hand (no `$ref`, no defaults), and add a test that checks it against the Pydantic model.
- **IDs and prose only:** the schema carries no name, rating, or cost (§5.4).
- **Lenient parse model:** `RankedOutput` accepts gapped or 0-based ranks and odd ID casing. Normalizing those is the gate's job; rejecting the whole response would throw away good picks.
- **Token headroom:** reasoning tokens count against `max_completion_tokens`. Leave room (8000); otherwise long reasoning truncates the JSON (L-10).

### 3.5 Grounding gate — the most important function in the codebase

```python
def validate_and_join(ranked, candidates_df, top_n, *, requested, stretch_band=None):
    valid = {id.strip().casefold(): id for id in candidates_df["restaurant_id"]}   # L-16
    picks = sorted(enumerate(ranked.picks), key=lambda t: (t[1].rank, t[0]))   # L-15: order by rank, then renumber
    # drop unknown or duplicate IDs (G-01, G-03, L-13, L-14) and log them
    # backfill only the slots dropped picks vacated, from pre-ranked order (G-04); never pad a short list
    # blank explanation → template explanation (L-17)
    # join authoritative facts from the catalog row; model prose stays prose (G-05)
```

- **Facts:** every displayed fact comes from the DataFrame join. If the model's prose quotes a price, it stays prose and never fills a field.
- **Single path:** the deterministic ranker's output goes through the same gate, so nothing reaches a response without passing through it.

### 3.6 Orchestration

Assemble §6's `recommend()`. These all become `LLMUnavailable` and end in `deterministic_rank()` with `degraded=True`:
- `groq.APIError`, which covers `APITimeoutError`, `APIConnectionError`, and 401/429/5xx after the SDK's retries
- a `finish_reason` other than `"stop"`
- empty content
- a schema-validation failure
- zero picks surviving the gate (G-02)

**Response trace:** add a `trace` to the response with the ranker used, model, prompt/cached/completion tokens, estimated cost, dropped IDs, backfill count, and fallback reason. The CLI prints it, and eval M-02, M-22 and M-23 read it.

**Degraded-path check:** leave `GROQ_API_KEY` empty; the system must still return results. Also add `--no-llm` to the CLI for the phase-2 reference output.

### 3.7 Caching check

- **Log:** `usage.prompt_tokens_details.cached_tokens` on every call.
- **Run the same query twice:** the second run should show non-zero cached tokens.
- **How Groq caches:** only prefixes above a model-specific minimum (128-1024 tokens) are cached, and entries expire after 2 hours unused.
- **If gpt-oss shows zero on an identical rerun:** something volatile is in the prefix. Find it now, while the prompt is small enough to read by eye.
- **If Qwen shows zero once the prefix is proven stable:** record that caching isn't active for that model, and don't chase it further.

### 3.8 Tests

Stub the client; no network in CI. A shared fixture replaces the real client constructor with one that fails the test. Cover:
- **Prompt:** contains every candidate ID; the system prompt is identical across requests; serialization is byte-stable; hostile catalog text stays inside a JSON string (S-02).
- **Grounding:** a fabricated ID is dropped and backfilled; a real ID outside the candidate set is rejected; duplicates, rank gaps, ID casing and blank explanations are normalized.
- **Model and API failures:** truncated, empty, and malformed responses degrade rather than raise; 401/429/timeout/5xx degrade; the call cap stops a runaway loop.
- **End to end:** with a key missing, the pool empty, or `use_llm=False`, the LLM is never called; the fallback path returns a valid response with `degraded=True`.

**Done when:**
- [x] CLI returns 5 LLM-ranked picks with genuine, preference-specific explanations
- [x] A free-text preference ("family-friendly") visibly changes the ordering vs. `--no-llm`
- [x] Injecting a fake ID into a stubbed response results in it being dropped, logged, and backfilled
- [x] Leaving `GROQ_API_KEY` empty still returns results, flagged `degraded`
- [x] Second identical query shows non-zero `cached_tokens` (gpt-oss-120b)
- [x] Measured cost per query is in the expected ~$0.002 range for gpt-oss-120b (~$0.006 for Qwen). Re-baseline eval M-22's target to the chosen model

**Status: complete (2026-09-15)**, verified live against `openai/gpt-oss-120b`:
- **Free text reorders results.** "family-friendly" moved a cafe and a bar out of Koramangala's top 5 in favour of casual dining with table booking, and put Whitefield's microbreweries behind family restaurants. "date night with drinks" produced a different list again. Every run had `degraded=false` and 0 dropped IDs.
- **Cost and latency.** $0.0012-0.0019 per query (3.8-4.1K tokens in, 1.0-2.1K out), with the model call taking ~4-5 s when the account isn't rate-limited.
- **Caching.** An identical rerun served 3,584 of 3,828 prompt tokens from cache. Groq caching is best-effort, so some reruns report none.
- **Tests.** 164 pass with a stubbed client. The ID gate cases (G-01…G-05), response-shape failures (L-01…L-17), the call cap (L-24) and injection (S-01, S-02) are all covered.
- **Grounding rule tightened.** After the model described restaurants as "spacious" and offering a "buffet", with no field to support either, the prompt now forbids atmosphere, size, service-style and offer claims unless a candidate field supports them.

**Carried into phase 5:**
- **Candidate recall for free text (M-13, M-14).** The top-25 shortlist ignores `free_text`. Where the pool lacks the right kind of restaurant (Indiranagar at a medium budget: 5 casual-dining places among 141, ranked 23rd and 93rd onward), the LLM has nothing better to promote. Capping how many of one restaurant type make the shortlist was simulated and rejected: it didn't help there, and it cut Koramangala's casual-dining candidates from 11 to 5. Candidates for tuning: the §4.4 free-text blend.
- **Ungrounded prose (L-19/L-20, M-19).** One "buffet" claim still appeared after the rule was added; it comes from the model's general knowledge. Measure the rate in the eval, and consider the optional explanation audit.
- **Rate limit (pacing added 2026-09-15).** The account's `openai/gpt-oss-120b` limits are 30 requests/min, 1,000 requests/day, 8,000 tokens/min and 200,000 tokens/day. One ranking call measured 6.3K tokens, so tokens bind: ~1 call/min, ~31/day.
  - **Pacing.** [src/llm/rate_limit.py](../src/llm/rate_limit.py) reserves an estimate against all four limits before each call and corrects it to real usage afterwards. It waits up to `LLM_RATE_LIMIT_MAX_WAIT_S` (10 s) for capacity, then degrades with the reason in `trace.fallback_reason`.
  - **429s.** `GroqClient` no longer retries a 429, which caused the old 13-38 s stalls. A 429 pauses calls for its `retry-after`.
  - **For the eval.** Live runs must still be paced about one call a minute (for example `LLM_RATE_LIMIT_MAX_WAIT_S=70`, so calls queue instead of degrading), and the ~31-calls/day budget caps how many live queries a day can run. Otherwise latency (M-20) measures throttling, not the model.
  - **Not covered.** The eval runner and LLM judge (5.5, 5.9) must share the limiter, and limits aren't coordinated across processes.

**Budget note:** Groq is roughly 15× cheaper per query than the §5.5 Claude estimate, but a loop bug still spends real money. The CLI enforces a hard cap on LLM calls per run (`--max-llm-calls`, default 3); past it, the call degrades instead of spending.

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
- [x] `uvicorn src.api.main:app --reload` + `streamlit run app/streamlit_app.py`, and a full journey works in the browser
- [x] Every field required by the problem statement's §5 output spec appears on the card: name, cuisine, rating, estimated cost, AI explanation
- [x] No-results and degraded states both render correctly (force them to check)
- [x] `pytest tests/test_api.py` passes with a stubbed recommender

**Status: complete (2026-09-15)**
- **Browser journey.** Headless Chrome ran against the real API and catalog: area and budget chosen, "family-friendly, quiet" typed, submitted. The submit button was disabled during the call (U-07), and 5 LLM-ranked cards arrived in ~6 s with `degraded=false`. Cards render correctly in light and dark themes.
- **Live API.** `/meta/*` serves 93 areas, 105 cuisines and the phase 1 rupee bands. Invalid input returns the 422 envelope. A missing `CATALOG_PATH` stops startup with the ingest command in the error (A-01). Request logs carry a `request_id` and never the body.
- **Tests.** 195 pass. `tests/test_api.py` covers A-01, A-03, A-04, A-08, A-11, A-12, I-11, I-12, I-14 and I-19. `tests/test_streamlit_app.py` uses Streamlit's AppTest against a fake API to force the unreachable-API, null-rating, null-cost, no-URL, no-results, degraded, widened-search and HTML-escaping states (U-01…U-06, U-10, U-11). The degraded and no-results states were forced there, not in a live browser.
- **Additions beyond the task list.**
  - `GET /meta/budgets` (the rupee bands for the budget radio).
  - `blocking_constraints` on the response (for one-click relaxation, U-05).
  - The `API_URL` setting.
  - The error envelope, documented in architecture §7.

**Carried into phase 5:**
- **`/health` doesn't probe Groq.** It reports `llm_configured` only. An invalid key still shows up per request as `degraded` with `trace.fallback_reason` (A-09).
- **Body size cap reads `Content-Length` only.** A chunked request without one isn't capped (A-12).
- **Not yet done:** rate limiting (5.2, A-07), response caching (5.1), and the `run_eval --via-api` phase 4 eval gate (needs the 5.5 runner).

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

**5.9** — the judge's system prompt as a frozen module-level constant (same cache rule as 3.2), with a Pydantic score schema sent as a strict `json_schema` response format through the same Groq call path as 3.4: one rubric prompt for per-pick explanation quality, one pairwise prompt for LLM-vs-deterministic comparison. Calibrate before trusting it: hand-score 15 picks blind, and require ±1 agreement on ≥80% and every ungrounded pick caught ([eval.md](eval.md) §4.3).

**5.6** — only now tune the §4.4 weights and the prompt, measuring each change. Tuning before the eval exists is guesswork (§11). Record each accepted change with its run ID in `evals/results/CHANGELOG.md`, so any result can be traced to the change that produced it.

**5.8** — deliberately break things and confirm graceful behavior: unset the API key, point `catalog_path` at a missing file, send a 10K-character `free_text`, send an unknown location, simulate a timeout.

**Done when:**
- [ ] Eval runs clean: 0 grounding violations, constraint satisfaction above your chosen bar. *0 grounding violations and 100% constraint satisfaction in every run, but the live smoke run fails M-07 on `ft-01`, and the full 30-query live run hasn't been run (it needs about 2.5 days of Groq token budget)*
- [x] Every failure drill degrades gracefully with a clear user-facing message
- [ ] README lets someone clone the repo and reach a working UI following only its instructions. *Quick start written; not yet tried from a fresh clone*

**Status: built; eval partly green (2026-09-15)**
- **5.1-5.3.**
  - Response cache: 1 h TTL, 512 entries. Transient fallbacks aren't cached, and a response served from the cache says `cached: true`.
  - Per-client 429 on `POST /recommend`, with `retry_after_s`.
  - Input bounds on cuisine length, party size and `LLM_CANDIDATE_K`; control characters stripped.
- **5.4.** 30 labelled queries matching [eval.md](eval.md) §3.1, 15 of them with gold IDs, all labels checked against the catalog (`--check-labels`).
- **5.5.** `run_eval` supports deterministic, LLM, `--via-api` and forced-failure modes, plus `--judge`, `--pairwise`, `--ids`, `--resume`, `--save-baseline` and `--max-calls`. It waits out short Groq 429s and saves the run as incomplete on long ones. `compare` refuses invalid runs and lists per-query flips.
- **5.8.** `python -m evals.failure_drill` covers 11 scenarios, including a key unset, timeout, 429, malformed output, daily budget spent, missing catalog, 10K free text, Delhi, malformed JSON, a request flood, and the UI with the API down. All pass; the report is in `evals/results/drill_*.md`.
- **5.9.** Frozen judge and pairwise prompts go through the ranker's guarded call path. The calibration CLI (`export-sheet`, `calibrate`) is built, but **calibration itself needs 15 hand-scored picks and hasn't been done**, so M-15 isn't trusted yet.
- **5.6.** No weight or prompt change accepted; see [evals/results/CHANGELOG.md](../evals/results/CHANGELOG.md).
- **Runs:**

  | Run | Result |
  | --- | --- |
  | Deterministic, 30 queries (saved as the baseline) | All blocking metrics pass. M-13 100%. M-14 35% (by design) |
  | Forced LLM failure | M-24 100% (30/30) |
  | Live smoke, 7 queries with judge and pairwise | M-01 = 0, M-19 = 0, M-17 lift 4/4, M-14 94%, $0.0015/query. **M-07 fails on `ft-01`**, M-15 3.92 (8 ungrounded-prose picks), M-23 0/7, M-20 p95 6.1 s |

- **Tests:** 286 pass.

**Open:**
- **Free-text candidate recall (M-07 `ft-01`).** The shortlist ignores free text, so a citywide "quick lunch" query shortlists Casual Dining and bars. Candidates: §4.4's embedding blend, or a catalog-field blend, measured deterministically first.
- **Ungrounded prose (M-15).** The model still claims facilities and atmosphere that aren't in the row, including one false "allows table booking". Next: tighten the prompt, validated with `--repeat 2`, and consider a code check for boolean facility claims.
- **Judge calibration.** Hand-score 15 picks. Known judge issues first: it penalises ignoring injected instructions and treats "craft beer" as ungrounded for a Microbrewery.
- **M-23 cross-query caching** stays at 0 while the static prefix is only the system prompt; decide whether to re-baseline the target or restructure the prefix.
- **Full live runs** need `--resume` across days at ~31 calls/day, or a paid Groq tier.

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
| Prompt cache never hits | 3 | Medium | Log `usage.prompt_tokens_details.cached_tokens` from the first call (3.7) |
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
