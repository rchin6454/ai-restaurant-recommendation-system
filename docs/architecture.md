# Architecture — AI-Powered Restaurant Recommendation System

Implementation architecture for the requirements in [problemStatement.md](problemStatement.md).

**Stack assumption:** Python 3.11+, FastAPI for the service, Streamlit for the UI, Claude (`claude-opus-5`) as the LLM. Python is assumed because the dataset ships through the Hugging Face `datasets` library and the data-prep work is pandas-shaped. If you'd rather build the service in TypeScript, the layering below still holds — only §9's code changes.

---

## 1. Design Principles

| Principle | What it means here |
| --- | --- |
| **Deterministic retrieval, generative ranking** | Structured filters (SQL/pandas) decide *what is eligible*. The LLM decides *what is best* and *why*. The LLM never sees the full 51,717-row dataset. |
| **Ground every claim** | The LLM may only return restaurants whose IDs appear in the candidate set it was given. Output is validated against that set before it reaches the user. |
| **Facts from data, prose from the model** | Name, rating, and cost shown to the user come from the DataFrame row, not from the model's text. The model contributes ranking order and explanation only. |
| **Cheap path first** | Filtering is free. Only the ~20-40 survivors cost tokens. A cache-friendly prompt layout keeps repeat queries cheap. |
| **Degrade, never dead-end** | Zero matches triggers progressive constraint relaxation, not an empty screen. LLM failure falls back to a deterministic score ranking. |

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  PRESENTATION            Streamlit UI  /  REST client            │
│  preference form → results cards → "why this fits" explanations  │
└───────────────────────────────┬──────────────────────────────────┘
                                │ POST /recommend  (JSON)
┌───────────────────────────────▼──────────────────────────────────┐
│  API LAYER (FastAPI)                                             │
│  request validation · rate limit · error envelope · tracing      │
└───────────────────────────────┬──────────────────────────────────┘
┌───────────────────────────────▼──────────────────────────────────┐
│  ORCHESTRATION  recommender.py                                   │
│   1 normalize preferences   2 retrieve candidates                │
│   3 pre-rank + truncate     4 LLM rank & explain                 │
│   5 validate + merge facts  6 assemble response                  │
└───────┬───────────────────────────────────────┬──────────────────┘
        │                                       │
┌───────▼─────────────────┐          ┌──────────▼──────────────────┐
│  RETRIEVAL LAYER        │          │  LLM LAYER                  │
│  filters · scoring      │          │  prompt builder             │
│  relaxation ladder      │          │  Anthropic client + retries │
│  optional semantic sim  │          │  structured output parsing  │
└───────┬─────────────────┘          └─────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────────────────────┐
│  DATA LAYER                                                      │
│  HF dataset → cleaning pipeline → restaurants.parquet            │
│  in-memory pandas index (+ optional DuckDB / FAISS)              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Layer

### 3.1 Source

`ManikaSaini/zomato-restaurant-recommendation` — 51,717 rows, single `train` split, CSV (auto-converted to Parquet on the Hub), ~574 MB. Raw columns:

`url`, `address`, `name`, `online_order`, `book_table`, `rate`, `votes`, `phone`, `location`, `rest_type`, `dish_liked`, `cuisines`, `approx_cost(for two people)`, `reviews_list`, `menu_item`, `listed_in(type)`, `listed_in(city)`

> **Coverage caveat — decide this early.** This is the Bengaluru Zomato dump; `location` and `listed_in(city)` are Bangalore neighbourhoods (Banashankari, Basavanagudi, Koramangala…), not Indian metros. The problem statement's "Delhi, Bangalore" example cannot be served by this dataset alone. Two honest options: (a) scope the product to Bangalore and treat "location" as *neighbourhood*, or (b) add a second dataset for other cities. Option (a) is recommended for a first build — just label the field "Area (Bengaluru)" in the UI so the constraint is visible rather than silently wrong.

### 3.2 Ingestion & Cleaning Pipeline

`src/data/ingest.py` — runs once, offline, writes a Parquet artifact. The service never touches Hugging Face at request time.

```
load_dataset("ManikaSaini/zomato-restaurant-recommendation", split="train")
  → to_pandas()
  → drop heavy columns (reviews_list, menu_item) unless the review-snippet feature is on
  → clean each field (table below)
  → deduplicate
  → derive fields
  → validate
  → write data/processed/restaurants.parquet
```

| Raw field | Problem in the raw data | Cleaning rule |
| --- | --- | --- |
| `rate` | `"4.1/5"`, `"NEW"`, `"-"`, `NaN` | Strip `/5` → float. `NEW`/`-`/`NaN` → `None`, plus `is_new` / `is_unrated` boolean. **Never impute a rating** — an invented 3.5 corrupts every filter downstream. |
| `approx_cost(for two people)` | `"1,200"` with comma separator, strings, nulls | Strip commas → `Int64` (nullable). Rename `cost_for_two`. |
| `votes` | `0` on many rows | `int`. Used as a confidence weight, not a filter. |
| `cuisines` | `"North Indian, Mughlai, Chinese"` | Split on `,`, strip, title-case, canonicalize aliases → `list[str]` + a lowercase `cuisines_norm` set for matching. |
| `rest_type` | `"Casual Dining, Bar"` | Same treatment → `list[str]`. Drives "family-friendly", "quick bite" inference. |
| `name` | Encoding artifacts (`Ã©`), stray whitespace | `ftfy.fix_text` → `.strip()`. |
| `location` / `listed_in(city)` | Near-duplicate area names, casing drift | Lowercase-normalize, map aliases (`"btm"` ≡ `"BTM Layout"`), keep the display form. |
| `online_order`, `book_table` | `"Yes"` / `"No"` | → `bool`. |
| `dish_liked` | Comma list, ~54% null | → `list[str]`. High-signal for cuisine/dish intent; keep. |
| `address`, `url`, `phone` | Free text | Keep `address` + `url` for display. **Drop `phone`** — personal/business contact data with no role in recommending. |

**Deduplication.** The dataset repeats one restaurant across `listed_in(type)` categories (Delivery, Dine-out, Buffet…). Collapse on `(name_norm, address_norm)`; keep the max `votes` row as canonical and aggregate the categories into a `listed_types: list[str]`. Expect a large reduction — roughly 51.7K rows to ~12-13K distinct restaurants. Skipping this is the single most common bug in builds on this dataset: the same restaurant appears five times in one result list.

### 3.3 Derived Fields

| Field | Derivation | Used by |
| --- | --- | --- |
| `restaurant_id` | Stable hash of `(name_norm, address_norm)` | Grounding contract with the LLM |
| `budget_band` | Quantile cut of `cost_for_two` over the corpus → `low` / `medium` / `high` | Budget filter |
| `bayesian_rating` | `(v·R + m·C) / (v + m)`, `v`=votes, `R`=rate, `C`=corpus mean, `m`=vote prior (~50) | Pre-ranking |
| `popularity_pct` | Percentile rank of `votes` | Pre-ranking tiebreak |
| `text_blob` | `name + cuisines + rest_type + dish_liked + location` | Optional embedding / keyword match |

**Budget bands** are corpus-relative quantiles, not hardcoded rupee thresholds — that keeps "medium" meaningful if the dataset is ever swapped. Document the resulting boundaries in the UI ("medium ≈ ₹400-800 for two") so the user knows what they picked.

### 3.4 Storage & Access

The cleaned Parquet is ~10-20 MB and loads into memory in under a second. Start there.

```python
@lru_cache(maxsize=1)
def get_catalog() -> pd.DataFrame:          # loaded once per process
    return pd.read_parquet(settings.catalog_path)
```

Escalate only when measurement demands it: **DuckDB** over the same Parquet if filter latency grows or you want real SQL; **FAISS/Chroma** only if you add semantic free-text search (§4.4). A vector database is not required for this system — the filters are all structured.

---

## 4. Retrieval Layer

### 4.1 Normalized Preferences

```python
class Preferences(BaseModel):
    location: str | None = None            # Bengaluru area
    budget: Literal["low", "medium", "high"] | None = None
    cuisines: list[str] = []               # OR-matched
    min_rating: float | None = Field(None, ge=0, le=5)
    party_size: int | None = None
    free_text: str | None = None           # "family-friendly, quick service"
    online_order: bool | None = None
    book_table: bool | None = None
```

`free_text` is deliberately **not** converted into filters. It is the LLM's job (§5) — that is precisely the part of the problem structured predicates handle badly.

### 4.2 Filter Chain

Applied as a conjunction of vectorized pandas masks:

| Preference | Predicate | Notes |
| --- | --- | --- |
| `location` | `location_norm == x` or `city_norm == x` | Fuzzy-match user input to the known area vocabulary (`rapidfuzz`, ≥85 ratio) before filtering, so "koramangla" still works. |
| `budget` | `budget_band == x` | Soft edge: also admit one band up if fewer than 15 rows match, flagged as `stretch: true`. |
| `cuisines` | `cuisines_norm ∩ requested ≠ ∅` | OR, not AND — AND over 3 cuisines empties the set. Rank by overlap size later. |
| `min_rating` | `rating >= x` | Unrated rows excluded when this is set; surfaced as a "N new restaurants hidden" note. |
| `online_order` / `book_table` | equality | Only applied when explicitly requested. |

### 4.3 Relaxation Ladder

If the survivor count is below `MIN_CANDIDATES` (default 10), relax in this order and record every step in `relaxations[]` so the response can say what it did:

1. `min_rating` − 0.3 (floor 3.0)
2. Budget widened one band
3. Location widened to adjacent areas / whole city
4. Cuisine constraint dropped last

Cuisine goes last because it is usually the preference the user cares most about; rating goes first because a 0.3 delta is rarely a real quality difference on a 5-point scale.

### 4.4 Pre-Ranking (the token gate)

The filter can return thousands of rows; the LLM should see ~25. Score and truncate:

```
score = 0.45 · norm(bayesian_rating)
      + 0.20 · norm(log1p(votes))
      + 0.20 · cuisine_overlap_ratio
      + 0.15 · budget_fit          # 1.0 exact band, 0.5 adjacent
```

Take the top `LLM_CANDIDATE_K` (default 25), then apply **diversity trimming** — cap 2 outlets per restaurant chain, so the list isn't five Domino's. Weights live in config; they are a starting point to tune against the eval set (§11), not settled truth.

*Optional:* when `free_text` is present, blend in cosine similarity between the query embedding and `text_blob`. Worth adding only after the deterministic path is measured — it introduces an index to build and keep in sync.

---

## 5. LLM Layer

### 5.1 Responsibility Boundary

The model gets a small, clean, pre-filtered candidate list and does the three things structured code does badly:

1. **Rank** — weigh "highly rated" against "cheap" against "the cuisine I asked for" the way a person would.
2. **Interpret free-text preferences** — map "family-friendly" onto Casual Dining + table booking + higher party capacity; "quick service" onto Quick Bites / Takeaway.
3. **Explain** — one grounded sentence per pick, referencing the user's stated preferences.

### 5.2 Prompt Structure (cache-aware)

Prompt caching is a **prefix match** — order blocks from most stable to most volatile so repeat traffic hits the cache:

```
system (static, cached)     role, ranking rubric, grounding rules, output contract
   ↓
user turn:
  [block 1] candidate JSON  ← varies per query
  [block 2] user preferences + free text
```

The system block is frozen at build time — no timestamps, no request IDs, no f-strings interpolating per-request data — and marked cacheable:

```python
system=[{
    "type": "text",
    "text": RANKING_SYSTEM_PROMPT,          # constant module-level string
    "cache_control": {"type": "ephemeral"},
}]
```

Verify with `response.usage.cache_read_input_tokens`. If that is 0 across identical-prefix requests, something volatile crept into the prefix.

Candidates are serialized compactly — only ranking-relevant fields, `sort_keys=True` for byte stability:

```json
{"id":"r_8f21","name":"Jalsa","cuisines":["North Indian","Mughlai"],
 "rating":4.1,"votes":775,"cost_for_two":800,"area":"Banashankari",
 "type":["Casual Dining"],"dishes":["Pasta","Lunch Buffet"],"book_table":true}
```

### 5.3 System Prompt Contract

The system prompt must state, explicitly:

- **Grounding:** "Recommend only restaurants from the provided list. Use the exact `id` values. Never invent a restaurant, rating, or price."
- **Rubric:** preference match first, then rating quality (weighing vote count as confidence), then budget fit, then variety across the list.
- **Explanation style:** one or two sentences, concrete, tied to what the user asked for. No marketing language. No claims not supported by the candidate fields — if `dish_liked` is empty, don't describe the food.
- **Honesty:** if no candidate genuinely fits a stated preference, say so in the summary rather than overselling.

### 5.4 Structured Output

Use `messages.parse()` with a Pydantic schema — it removes JSON-parsing failure as a category of bug:

```python
class Pick(BaseModel):
    id: str
    rank: int
    explanation: str
    match_highlights: list[str]

class Recommendations(BaseModel):
    picks: list[Pick]
    summary: str
    caveats: list[str] = []

response = client.messages.parse(
    model="claude-opus-5",
    max_tokens=4000,
    system=[{"type": "text", "text": RANKING_SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}],
    messages=[{"role": "user", "content": user_block}],
    output_format=Recommendations,
)
result = response.parsed_output
```

Note the schema carries **no** name/rating/cost fields. The model returns IDs and prose; every displayed fact is joined back from the DataFrame. This makes a hallucinated price structurally impossible rather than merely discouraged.

### 5.5 Model & Cost

| | |
| --- | --- |
| Model | `claude-opus-5` — $5/M input, $25/M output |
| Thinking | `{"type": "adaptive"}` for ranking; it is a genuine multi-criteria judgment |
| Typical request | ~2.5K input (25 candidates + system), ~800 output ≈ **$0.03/query** uncached |
| With prefix caching | System block (~1K tokens) served at ~0.1× on repeat traffic |
| Cheaper tier | `claude-sonnet-5` ($2/$10) is a reasonable swap once the eval set shows quality holds — measure before switching, don't assume |

### 5.6 Failure Handling

| Failure | Handling |
| --- | --- |
| `APITimeoutError`, 429, 5xx | SDK auto-retries (2×, exponential). On exhaustion → deterministic fallback |
| Deterministic fallback | Return the §4.4 pre-ranked top N with template explanations ("4.3★, North Indian, ₹800 for two — matches your budget and cuisine"). Flag `degraded: true` so the UI can say the AI explanations are unavailable |
| Hallucinated ID | Drop that pick, log it, backfill from pre-ranked order |
| Fewer picks than requested | Accept it — the model declining to pad a thin list is correct behavior |
| Empty candidate set after full relaxation | Skip the LLM call entirely; return a "no matches" response naming which constraint was impossible |

---

## 6. Orchestration Layer

`src/core/recommender.py` — the only module that knows the full sequence:

```python
def recommend(prefs: Preferences, top_n: int = 5) -> RecommendationResponse:
    catalog     = get_catalog()
    prefs       = normalize(prefs, catalog.vocabulary)     # fuzzy-match area/cuisine
    pool, relax = retrieve(catalog, prefs)                 # filter + relaxation ladder
    if pool.empty:
        return empty_response(prefs, relax)
    candidates  = pre_rank(pool, prefs, k=settings.llm_candidate_k)
    try:
        ranked  = llm_rank(candidates, prefs)              # structured output
    except LLMUnavailable:
        ranked  = deterministic_rank(candidates)           # degraded path
    picks       = validate_and_join(ranked, candidates)    # grounding enforcement
    return assemble(picks[:top_n], relax, prefs)
```

`validate_and_join` is the grounding gate — it drops unknown IDs, re-sorts by the model's ranks, and attaches authoritative facts from the catalog row. Nothing reaches the response without passing through it.

---

## 7. API Layer

```
POST /recommend        preferences → ranked recommendations
GET  /meta/locations   area vocabulary (populates the UI dropdown)
GET  /meta/cuisines    cuisine vocabulary
GET  /meta/budgets     each budget band's rupee range (labels the UI's budget radio)
GET  /health           catalog loaded, row count, model, whether an LLM key is configured
```

The `/recommend` body is `Preferences` (§4.1) plus optional `top_n` and `use_llm`. Unknown fields are rejected.

**Response shape:**

```json
{
  "recommendations": [{
    "rank": 1, "restaurant_id": "r_8f21", "name": "Jalsa",
    "cuisines": ["North Indian", "Mughlai"], "rating": 4.1, "votes": 775,
    "cost_for_two": 800, "area": "Banashankari", "url": "https://...",
    "explanation": "Strong 4.1 rating across 775 votes, and at ₹800 for two it sits inside your medium budget while covering the North Indian cuisine you asked for.",
    "match_highlights": ["North Indian", "within budget", "well-reviewed"]
  }],
  "summary": "Five North Indian options in Banashankari, all rated 4.0+.",
  "caveats": ["Rating filter relaxed from 4.5 to 4.2 — only 3 restaurants met 4.5."],
  "applied_filters": {"location": "Banashankari", "budget": "medium", "min_rating": 4.2},
  "relaxations": [{"field": "min_rating", "from": 4.5, "to": 4.2}],
  "degraded": false,
  "candidates_considered": 25,
  "latency_ms": 2840
}
```

`applied_filters` and `relaxations` are not optional polish — they are what makes the result legible when it isn't what the user expected. The full model is `RecommendationResponse` in `src/core/models.py`. Beyond the fields above, it carries `outcome`, `interpretations` (how typed values were matched), `suggestions`, `trace` (ranker, model, tokens, fallback reason), and, when nothing matches, `blocking_constraints`: the filters whose removal alone would yield results, which the UI turns into one-click relaxation.

**Error envelope.** Every non-2xx response, whether validation (422), unknown route (404), oversized body (413), rate limited (429, with `retry_after_s` and a `Retry-After` header), catalog not loaded (503) or unhandled error (500), has one shape. Stack traces and internal messages go to the logs only, matched by `request_id`:

```json
{"error": {"code": "invalid_request", "message": "The request is invalid. See `details`.",
           "details": [{"field": "min_rating", "message": "Input should be less than or equal to 5"}],
           "request_id": "5fde33c1d8b74563"}}
```

---

## 8. Presentation Layer

**Streamlit** (`app/streamlit_app.py`) is the recommended first UI: sidebar form → result cards → done, no frontend build. Swap in React later if it outgrows that.

- **Inputs:** area dropdown (from `/meta/locations`, searchable), budget radio with rupee ranges shown, cuisine multiselect, rating slider, free-text box for everything else.
- **Results:** one card per pick — rank badge, name, cuisine chips, ★ rating with vote count, ₹ cost for two, the AI explanation in a visually distinct block, Zomato link.
- **Transparency:** a "why these results" expander showing `applied_filters`, any relaxations, and how many restaurants were considered.
- **States:** loading skeleton during the LLM call (2-4s is normal), a no-results state that names the blocking constraint and offers one-click relaxation, and a quiet banner when `degraded` is true.

Keep the AI explanation visually separated from the data fields. Users should be able to tell at a glance which parts are facts from the catalog and which are the model's reasoning.

---

## 9. Repository Layout

```
├── data/
│   ├── raw/                      # HF cache (gitignored)
│   └── processed/restaurants.parquet
├── docs/
│   ├── problemStatement.md
│   └── architecture.md
├── src/
│   ├── config.py                 # pydantic-settings; all tunables
│   ├── data/
│   │   ├── ingest.py             # HF download → clean → parquet
│   │   ├── cleaning.py           # per-field rules from §3.2
│   │   └── catalog.py            # cached loader + vocabularies
│   ├── core/
│   │   ├── models.py             # Preferences, Pick, RecommendationResponse
│   │   ├── filters.py            # filter chain + relaxation ladder
│   │   ├── ranking.py            # pre-ranking score, diversity trim
│   │   └── recommender.py        # orchestration
│   ├── llm/
│   │   ├── client.py             # Anthropic client, retries, timeouts
│   │   ├── prompts.py            # frozen system prompt (cache prefix)
│   │   └── ranker.py             # serialize → parse → validate
│   └── api/
│       ├── main.py               # FastAPI app
│       └── routes.py
├── app/streamlit_app.py
├── tests/
│   ├── test_cleaning.py          # the messy-value table, case by case
│   ├── test_filters.py
│   ├── test_grounding.py         # hallucinated IDs are dropped
│   └── fixtures/candidates.json
├── evals/
│   ├── queries.jsonl             # ~30 labelled preference sets
│   ├── run_eval.py               # runs the suite, scores metrics
│   ├── compare.py                # diffs two runs, per-query flips
│   ├── check_catalog.py          # phase-1 data-quality assertions
│   ├── judge_prompts.py          # frozen LLM-judge prompts + schemas
│   └── results/                  # timestamped runs + CHANGELOG.md
├── .env.example                  # ANTHROPIC_API_KEY=
└── pyproject.toml
```

**Dependencies:** `anthropic`, `datasets`, `pandas`, `pyarrow`, `pydantic`, `pydantic-settings`, `fastapi`, `uvicorn`, `streamlit`, `rapidfuzz`, `ftfy`. Optional: `duckdb`, `sentence-transformers`.

---

## 10. Configuration

All tunables in `src/config.py` via `pydantic-settings`, overridable by env var:

| Setting | Default | Purpose |
| --- | --- | --- |
| `anthropic_api_key` | — | From env; never committed |
| `model` | `claude-opus-5` | |
| `llm_candidate_k` | 25 | Candidates sent to the model |
| `default_top_n` | 5 | Recommendations returned |
| `min_candidates` | 10 | Relaxation trigger |
| `max_chain_outlets` | 2 | Diversity cap |
| `rank_weights` | see §4.4 | Pre-ranking blend |
| `llm_timeout_s` | 30 | Per-request timeout |
| `enable_semantic_search` | `false` | Embedding path toggle |
| `llm_requests_per_minute` / `_per_day` | 30 / 1,000 | Groq account limits for the model; calls are paced against them |
| `llm_tokens_per_minute` / `_per_day` | 8,000 / 200,000 | The binding limits: a ranking call is ~6.3K tokens |
| `llm_rate_limit_max_wait_s` | 10 | How long a call waits for capacity before degrading |
| `response_cache_enabled` / `_ttl_s` / `_max_entries` | `true` / 3600 / 512 | Exact-match response cache (plan 5.1) |
| `api_requests_per_minute` | 10 | Per-client limit on `POST /recommend` (plan 5.2) |

---

## 11. Testing & Evaluation

**Unit tests** cover the deterministic half, which is where silent correctness bugs live: every messy value in the §3.2 table (`"NEW"`, `"-"`, `"1,200"`, `NaN`), the dedup collapse, each filter predicate, the relaxation ladder order, and the grounding gate (feed it a fabricated ID; assert it is dropped).

**LLM tests** use a stubbed client — no API calls in CI. Assert the prompt contains all candidate IDs, that a malformed response degrades rather than raises, and that the schema round-trips.

**Eval set** (`evals/queries.jsonl`): ~30 preference sets with hand-labelled acceptable answers, covering the ordinary cases plus the awkward ones — over-constrained (relaxation must fire), single-match, conflicting ("cheap" + "fine dining"), and free-text-only. Track per run: grounding violations (must be 0), constraint satisfaction rate, explanation-quality score, p95 latency, cost per query. Run it before and after every prompt or weight change; without it, prompt tuning is guesswork.

---

## 12. Performance & Cost

| Stage | Latency |
| --- | --- |
| Catalog load | ~0.5s, once per process (not per request) |
| Filter + pre-rank | <50ms |
| LLM ranking | 2-4s (dominant) |
| **Total p95** | **~4s** |

Levers, in order of value: prefix caching on the system block (§5.2); an exact-match response cache keyed on normalized preferences with a ~1h TTL (repeat queries are common and the catalog is static); streaming the explanation text so the UI fills progressively; and only then, `claude-sonnet-5` at lower effort — validated against the eval set, not assumed.

---

## 13. Security & Data Handling

- **API key** from environment only. `.env` gitignored, `.env.example` committed with an empty value.
- **Prompt injection:** `free_text` is user-controlled and enters the prompt. It sits in the user turn, never the system block; the system prompt states that user preferences are data, not instructions; and the grounding gate means a successful injection still cannot fabricate a restaurant.
- **Dropped data:** `phone` is removed at ingestion. `reviews_list` contains user-authored review text — keep it out of the default pipeline; if a review-snippet feature is added later, treat those strings as untrusted content in the prompt for the same reason as `free_text`.
- **Rate limiting** on `/recommend` (every call costs money), plus a request cap per session in the UI.
- **Input validation:** Pydantic bounds on `min_rating`, a length cap on `free_text` (~500 chars), and an enum for budget.

---

## 14. Build Order

| Phase | Deliverable | Done when |
| --- | --- | --- |
| **1 — Data** | `ingest.py` + cleaning + Parquet artifact | Row count, null rates, and budget-band boundaries are printed and sane; dedup verified |
| **2 — Retrieval** | Filters, relaxation, pre-ranking, unit tests | Preference set → sensible top-25, no LLM involved |
| **3 — LLM** | Prompt, structured output, grounding gate | Ranked picks with explanations; fabricated IDs provably dropped |
| **4 — Interface** | FastAPI + Streamlit | End-to-end from browser |
| **5 — Hardening** | Fallbacks, caching, rate limiting, eval set | Eval passes with 0 grounding violations; degraded path verified by killing the API key |

Phase 2 is a complete, demonstrable system without any LLM — build it that way. If phase 3 goes wrong, you can see it immediately by comparing against phase 2's output, and the fallback path in §5.6 is already written.

---

## 15. Open Decisions

| Decision | Recommendation |
| --- | --- |
| Bangalore-only vs multi-city | Ship Bangalore-only, label it clearly in the UI (§3.1) |
| Semantic search | Defer — structured filters cover the stated requirements; add only if free-text queries measurably underperform |
| Review snippets in prompts | Defer — large token cost, untrusted content, marginal ranking gain |
| User accounts / history | Out of scope; the problem statement describes a stateless single-query flow |
