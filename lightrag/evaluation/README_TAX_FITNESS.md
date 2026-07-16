# Tax retrieval fitness (local API)

HTTP harness that scores tax legislation retrieval against a live LightRAG API
(default `http://localhost:9621`, same port as the WebUI backend).

**Cloud / remote agents cannot reach your laptop's `:9621`.** Live runs must be
executed on a machine that can open the API (or the WebUI stack). Do not treat
a skipped/unreachable live check in CI or a cloud agent as a fitness pass.

## Layers

| Layer | What it checks | Endpoint |
| --- | --- | --- |
| **Chunk integrity** | Chunk boundaries / paragraph-semantic quality (separate audit) | offline |
| **Retrieval** | Section Hit@K, provision string Hit@K, keyword-trap suppress, MRR | `POST /query/data` |
| **Answers / rules** | Expected citations present; trap citations absent | `POST /query` |

Failing **live retrieval/answer gates** often means the Act was chunked poorly —
see [Recommended ingest remediations](#recommended-ingest-remediations-if-gates-fail)
and [Is the DB fit?](#is-the-db-fit). Chunk audit failures should be fixed before
trusting retrieval scores.

## Prerequisites

1. API (or WebUI stack) listening on `:9621` **on the machine running the harness**
2. Tax source documents ingested into that workspace
3. Oracle/rules expanded under `lightrag/evaluation/tax_fixtures/`
   - `tax_retrieval_oracle.json` (synthetic / fixture cases)
   - `tax_rules_dataset.json`
   - Synthetic instruments: `mini_ita_deductions.md`, `td_general_deductions.md`,
     `pg_general_deductions.md`, `pr_general_deductions.md`
   - Optional live pack: copy `live_oracle_template.json` → a local file
     (e.g. `live_oracle.local.json`), reconcile strings via WebUI `/query/data`,
     set `"enabled": true` on cases you want scored, then pass `--oracle`

## Environment

| Variable | Meaning |
| --- | --- |
| `LIGHTRAG_API_URL` | Base URL (default `http://localhost:9621`) |
| `LIGHTRAG_API_KEY` | Optional `X-API-Key` |
| `TAX_EVAL_LIVE=true` | Enable live CLI/integration (or use `LIGHTRAG_RUN_INTEGRATION=true`) |
| `TAX_EVAL_MIN_SECTION_HIT` | Gate default `0.5` |
| `TAX_EVAL_MIN_PROVISION_HIT` | Gate default `0.5` |
| `TAX_EVAL_MIN_TRAP_SUPPRESS` | Gate default `1.0` |
| `TAX_EVAL_MIN_MRR` | Gate default `0.3` |
| `TAX_EVAL_MIN_RULES_PASS_RATE` | Gate default `1.0` |

## Run

```bash
# Offline unit tests (mocked HTTP; no server required)
./scripts/test.sh tests/evaluation/test_tax_retrieval_fitness.py

# Chunk integrity (offline; no API)
python -m lightrag.evaluation.chunk_integrity_audit

# Live fitness against local API (must reach :9621 from this host)
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness

# Custom endpoint / JSON report / retrieval-only
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \
  --api-url http://localhost:9621 \
  --output /tmp/tax_fitness.json \
  --retrieval-only

# Live oracle curated from the template (all shells start enabled=false)
cp lightrag/evaluation/tax_fixtures/live_oracle_template.json /tmp/live_oracle.local.json
# edit /tmp/live_oracle.local.json — fill REPLACE markers from /query/data, set enabled=true
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \
  --oracle /tmp/live_oracle.local.json --retrieval-only

# Mode matrix override
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \
  --modes naive,local,hybrid,mix
```

### Live loop on real corpus

Use this when the synthetic oracle scores look misleading on a real Podman /
local index (e.g. high provision Hit@K, low section Hit@K). The loop recons
seed questions, auto-drafts a live oracle from observed chunks, then sweeps
modes × `top_k` under **current** LightRAG settings (no re-ingest).

```bash
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_live_loop \
  --api-url http://localhost:9621 \
  --top-k-values 10,20 \
  --retrieval-only
```

PowerShell (Windows / Podman host that can reach `:9621`):

```powershell
$env:TAX_EVAL_LIVE = "true"
python -m lightrag.evaluation.tax_live_loop `
  --api-url http://localhost:9621 `
  --top-k-values 10,20 `
  --modes mix,hybrid,naive,local `
  --output-dir .\live_runs\manual
```

Writes under `--output-dir` (default `lightrag/evaluation/tax_fixtures/live_runs/<timestamp>`):

| File | Contents |
| --- | --- |
| `recon.json` | Per seed × mode top chunks (path, headings, instrument, snippet) |
| `live_oracle.generated.json` | Auto-drafted retrieval cases (`enabled` only when anchors hit) |
| `live_rules.generated.json` | Light rules cases from `rules` seeds |
| `sweep.json` | All mode/top_k configs + metrics |
| `recommendation.json` | Best mode(s)/top_k, metric table, diagnosis bullets |

Exit codes: `0` if recon+sweep completed (gate failures are informational),
`1` when the API is down / hard errors / live gate off.

Seeds live in `tax_fixtures/tax_live_seed_questions.json` (override with `--seeds`).
Offline unit tests: `./scripts/test.sh tests/evaluation/test_tax_live_loop.py`.

### Model A/B (tag reports with QUERY LLM)

After settling mode/`top_k`, compare cheap vs ZDR query models without mixing
JSON dumps. `tax_model_ab` reads `GET /health` →
`configuration.role_llm_config.QUERY.model` (fallback `configuration.llm_model`),
runs fitness, and writes a tagged report.

```powershell
# Peek at what the server thinks QUERY is
$env:TAX_EVAL_LIVE = "true"
uv run python -m lightrag.evaluation.tax_model_ab `
  --api-url http://localhost:9621 `
  --print-model-only

# Baseline (cheap QUERY_LLM_MODEL)
uv run python -m lightrag.evaluation.tax_model_ab `
  --api-url http://localhost:9621 `
  --label cheap `
  --modes hybrid `
  --oracle .\live_runs\topk_hybrid\live_oracle.generated.json `
  --rules .\live_runs\topk_hybrid\live_rules.generated.json `
  --output-dir .\live_runs\model_ab

# Switch QUERY_LLM_MODEL to ZDR model, restart Podman, then:
uv run python -m lightrag.evaluation.tax_model_ab `
  --label zdr `
  --modes hybrid `
  --oracle .\live_runs\topk_hybrid\live_oracle.generated.json `
  --rules .\live_runs\topk_hybrid\live_rules.generated.json `
  --output-dir .\live_runs\model_ab
```

Each report includes `model_ab.query_llm_model`, `label`, and a filename like
`fitness_zdr_vendor__model_….json`.

### Scenario + WHAT-IF eval (Part 2 first slice)

Larger live query tests for **client scenarios** and **WHAT-IF** variants that
target Act/TD/PG/PR reference packs with issues/benefits. This is the eval
harness only — not agent orchestration or LLM-as-judge.

Seeds: `tax_fixtures/tax_scenario_seeds.json` (employee home office + sole-trader
WHAT-IF, business expense, apportionment, keyword trap, Act↔TD/PG crosswalk,
penalty negative, etc.).

```powershell
# Sonnet (or other ZDR) QUERY model — answer-first scenario pack
$env:TAX_EVAL_LIVE = "true"
uv run python -m lightrag.evaluation.tax_scenario_eval `
  --api-url http://localhost:9621 `
  --modes hybrid `
  --top-k 50 `
  --chunk-top-k 50 `
  --label sonnet `
  --output-dir .\live_runs\scenario_sonnet

# Switch QUERY_LLM_MODEL to a cheaper OSS model, restart Podman, then:
uv run python -m lightrag.evaluation.tax_scenario_eval `
  --api-url http://localhost:9621 `
  --modes hybrid `
  --top-k 50 `
  --chunk-top-k 50 `
  --label oss `
  --output-dir .\live_runs\scenario_oss
```

Writes under `--output-dir`:

| File | Contents |
| --- | --- |
| `recon.json` | Per seed × mode chunks + `graph_context_present` |
| `scenario_oracle.generated.json` | Drafted expected_* from hints that hit recon |
| `answers/<case_id>.json` | Question, mode, model tags, answer text, refs, scores |
| `scenario_report.json` | Full packs + WHAT-IF diffs + `model_ab` |
| `recommendation.json` | Headline **answer** metrics + diagnosis |

**What to compare across models (`--label sonnet` vs `oss`):**

1. `recommendation.json` → `headline_answer_metrics`
   (`answer_pass_rate`, `issues_hit_rate`, `benefits_hit_rate`,
   `citation_source_hit_rate`) — primary QUERY-model signal
2. Side-by-side `answers/` folders — same `case_id`, different model tags
3. `what_if.changed_rate` / `must_change_hit_rate` — variant sensitivity
4. Retrieval (`mean_source_type_recall`, trap suppress) — secondary; often
   similar across QUERY models when KEYWORD/embeddings are unchanged

Exit codes: `0` run completed (gates informational), `1` API/hard error or
`--strict` answer pass-rate miss, `2` gated off / bad flags.

Offline unit tests: `./scripts/test.sh tests/evaluation/test_tax_scenario_eval.py`.

Hand-curated optional path: `tax_scenario_oracle.json` (disabled example shells).
Generated oracles stay under `live_runs/`.

Exit codes for `tax_retrieval_fitness`: `0` gates passed, `1` gates failed,
`2` gated-off / server down / bad config.

Cases with `"enabled": false` are skipped by the runner (useful for templates and
WIP live oracles). Omitting `enabled` means the case is active.

## Is the DB fit?

| Signal | Verdict | Next step |
| --- | --- | --- |
| Chunk integrity audit: P beats F on boundary/trap; high heading coverage | Ingest shape looks sound | Run retrieval fitness |
| Retrieval gates pass (section / provision / trap / MRR) on synthetic fixtures | Workspace fit for fixture corpus | Safe to expand live oracle |
| Trap suppress fails (penalty Part outranks Deduction Division) | Chunk / index layout unfit | Re-ingest with **P** chunker + headings (below) |
| Section / provision Hit@K low; trap OK | Index missing text or oracle strings drift | Reconcile oracle vs `/query/data`; re-ingest if chunks lack markers |
| Live check skipped / connect refused | **No fitness verdict** | Run on a host that can reach the API; cloud agents cannot use your `:9621` |
| Scenario / WHAT-IF gates | Part 2 harness in progress | See [Scenario + WHAT-IF eval](#scenario--what-if-eval-part-2-first-slice) |

## Recommended ingest remediations (if gates fail)

1. **Use the paragraph-semantic `P` chunker** (not fixed-token `F`) for Acts and
   ATO instruments so `s N-N` anchors and Division headings stay intact.
2. **Keep heading metadata / sidecars** — for the synthetic Act, ship
   `mini_ita_deductions.blocks.jsonl` beside the markdown so P does not fall
   back to recursive chunking.
3. **Clear and re-ingest** the tax workspace after changing chunker or source
   files (stale vectors will not match new boundaries). Optionally keep LLM
   response cache if unchanged.
4. **Reconcile oracle strings** against indexed text: open WebUI or
   `POST /query/data`, copy phrases from `content` / `content_headings` /
   `file_path` into `expected_*` / `expected_sources` (never book text that
   never survived ingest).
5. Re-run `chunk_integrity_audit`, then tax fitness with the same modes you care
   about (`mix` recommended when a reranker is configured).

## Interpreting gates

- **Section Hit@K** — fraction of `expected_sections` found in top-K chunk text
  (includes `content` and optional `content_headings`).
- **Provision Hit@K** — fraction of `expected_chunk_contains` substrings in top-K.
- **Trap suppress** — `forbidden_chunk_contains` must not outrank expected hits.
- **MRR** — reciprocal rank of the first expected provision/section hit.
- **Rules pass rate** — deterministic answer/citation judges on `/query`
  (`mix` + `include_references` + `include_chunk_content`). RAGAS is optional later.

Pytest marks the live check `@pytest.mark.integration`; it also skips when the
server is down even with `--run-integration` / `TAX_EVAL_LIVE`.

Tax modules (`tax_retrieval_fitness`, `chunk_integrity_audit`, `tax_fixtures`)
are separate from `lightrag.evaluation.RAGEvaluator` — that import stays lazy
when ragas/datasets are absent.

## Forward-compatible fields (v1 already supports)

Optional oracle / rules fields are reserved for Part 2 multi-instrument packs.
v1 **records or soft-scores** them when present; default gates do **not** fail on them.

| Field | Where | v1 behaviour |
| --- | --- | --- |
| `enabled` | retrieval / rules case | `false` → skipped by CLI runners; omit or `true` → active. |
| `expected_sources` `{acts,tds,pgs,prs}` | retrieval oracle case | Computes `source_type_recall` / `source_type_recall_by_type` on `RetrievalCaseScore`. Soft default `1.0` when absent/empty. Gate only with `--gate-source-type-recall` (default **off**). Empty `pgs`/`prs` is fine. |
| `expected_issues` / `expected_benefits` | rules (or future scenario) case | Soft substring checks → `issues_ok` / `benefits_ok` (True when lists empty). Included in `RulesCaseScore.passed` only when the list is non-empty. |
| `forbidden_sources` | retrieval case | Reserved (fixture schema); not enforced by v1 gates. |
| `scenario_id` / `what_if` / `baseline_case_id` | retrieval / scenario case | Null/omit in v1; wiring hooks for paired WHAT-IF variants. |
| `graph_context_present` | report metadata | Set from `/query/data` when `entities` or `relationships` are non-empty. |

Helpers already landed for later packs: `classify_instrument`, `extract_ranked_chunks`,
`source_type_recall`, `citation_set`, `jaccard`, `set_delta`.

## Not yet

- **Full agent orchestration** that composes packs into advice workflows —
  still deferred (scenario eval harness only).
- **LLM-as-judge / RAGAS** as the primary advisory-tone gate.
- Enforcing `forbidden_sources` or requiring multi-instrument packs in default
  v1 retrieval gates.

The Part 2 **eval harness** (`tax_scenario_eval.py` + `tax_scenario_seeds.json`)
is implemented — see [Scenario + WHAT-IF eval](#scenario--what-if-eval-part-2-first-slice).
`tax_scenario_oracle.json` remains the hand-curated optional path.
