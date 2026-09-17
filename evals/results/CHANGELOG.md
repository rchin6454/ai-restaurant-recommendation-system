# Eval changelog

Accepted changes to rank weights, prompts, `llm_candidate_k` or the model, newest first ([docs/eval.md](../../docs/eval.md) §6.1). Each entry names the run that justified it; check a new change with `python -m evals.compare <previous run> <new run>`.

## 2026-09-15 — First live smoke run, no tuning change

- **Run:** `2026-09-15T17-25-27_llm`: 7 queries (`thin-01`, `con-01`, `ft-01`…`ft-04`, `adv-01`) with `--judge --pairwise`. It used 31 LLM calls, $0.025 total, and about 30 minutes. It needed one `--resume` after a 3-second Groq 429; the runner now waits out short 429s.
- **Blocking:** M-01, M-03, M-05, M-06, M-08, M-12 and M-19 hold (0 grounding violations, 0 degraded, injection contained).
  - **M-07 fails on `ft-01`** ("quick vegetarian-friendly lunch near the office"): 2 picks where the label expects 5. The model behaved correctly: only 2 of the 25 shortlisted candidates are Quick Bites (15 are Casual Dining, 6 Bars), because the shortlist ignores free text. This is the candidate-recall gap carried from phase 3, not a ranker bug.
- **Quality:**
  - M-17 lift over baseline is 100% (4/4, all free-text queries).
  - M-14 free-text alignment is 94%, against 35% for the deterministic baseline.
  - M-16 honesty on conflict is 100%; `thin-01` returned only the 2 genuine Sri Lankan matches.
  - M-22 is $0.0015 per ranking call.
- **Misses:**
  - **M-15 is 3.92, with 8 picks scored grounded < 3.** Real ungrounded prose: "spacious … buffet" and "calm atmosphere" (`ft-02`), "late-night delivery" and "highest vote count" (`ft-04`), and **Corner House Ice Cream described as "allows table booking" when `book_table` is false** (`ft-02`). Some flags look too strict and need calibration before being trusted: "offering craft beer" for Microbrewery-type rows (`ft-03`), and preference-specific = 1 on `adv-01`, where ignoring the injected instruction is correct.
  - **M-23 is 0/7 cross-query prompt cache hits.** The shared prefix is only the system prompt, which likely sits under Groq's minimum cacheable length; identical reruns do hit the cache (phase 3).
  - **M-20 p95 is 6.1 s** against a 6 s target (p50 4.4 s).
- **Proposed next, one change at a time:**
  1. **Free-text candidate recall.** The §4.4 optional embedding blend (or a catalog-field blend) to fix `ft-01`, measured first with `--mode deterministic` using candidate-level signal matches.
  2. **Grounding prompt tightening** for facility and atmosphere claims, validated with `--repeat 2` live runs.
  3. **Judge prompt fix** so it doesn't penalise ignoring injected instructions, then calibration against human scores.

  Each needs a measured run before it's accepted, and live runs are limited to about 31 calls a day.

## 2026-09-15 — Baseline, no tuning change

- **Runs:** `2026-09-15T17-23-23_deterministic` (saved as `baseline_deterministic.json`) and `2026-09-15T17-23-26_llm_forced-failure`.
- **Configuration:** rank weights rating 0.45 / votes 0.20 / cuisine 0.20 / budget 0.15, `llm_candidate_k` 25, `min_candidates` 10, model `openai/gpt-oss-120b`.
- **Blocking metrics:** all hold in both runs. M-24 (degraded-path validity) is 100% (30/30).
- **Weights left unchanged.** §6.3 tunes weights first because they decide what the model sees. Candidate recall (M-13) is already 100% (36/36 gold IDs shortlisted), so no weight change has a measured problem to fix.
- **Label caveat.** Gold IDs were chosen by reading the catalog sorted by rating and votes, which is close to what the pre-ranking score rewards. That makes M-11 (gold hit, 100%) and M-13 easy to pass for the deterministic ranker; treat them as regression guards, not as evidence of quality.
- **Known gap.** M-14 (free-text alignment) is 35% (7/20): the deterministic ranker ignores free text by design. It is measured on LLM runs.
