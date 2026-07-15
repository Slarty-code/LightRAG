# Tax retrieval fitness (local API)

HTTP harness that scores tax legislation retrieval against a live LightRAG API
(default `http://localhost:9621`, same port as the WebUI backend).

## Layers

| Layer | What it checks | Endpoint |
| --- | --- | --- |
| **Chunk integrity** | Chunk boundaries / paragraph-semantic quality (separate audit) | offline |
| **Retrieval** | Section Hit@K, provision string Hit@K, keyword-trap suppress, MRR | `POST /query/data` |
| **Answers / rules** | Expected citations present; trap citations absent | `POST /query` |

Failing **live retrieval/answer gates** often means the Act was chunked poorly —
re-ingest with the paragraph-semantic **`P`** chunker, then re-run this suite.
Chunk audit failures should be fixed before trusting retrieval scores.

## Prerequisites

1. API (or WebUI stack) listening on `:9621`
2. Tax source documents ingested into that workspace
3. Oracle/rules expanded under `lightrag/evaluation/tax_fixtures/`
   - `tax_retrieval_oracle.json`
   - `tax_rules_dataset.json`
   - Synthetic instruments: `mini_ita_deductions.md`, `td_general_deductions.md`,
     `pg_general_deductions.md`, `pr_general_deductions.md`

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

# Live fitness against local API
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness

# Custom endpoint / JSON report / retrieval-only
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \
  --api-url http://localhost:9621 \
  --output /tmp/tax_fitness.json \
  --retrieval-only

# Mode matrix override
TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \
  --modes naive,local,hybrid,mix
```

Exit codes: `0` gates passed, `1` gates failed, `2` gated-off / server down / bad config.

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

## Forward-compatible fields (v1 already supports)

Optional oracle / rules fields are reserved for Part 2 multi-instrument packs.
v1 **records or soft-scores** them when present; default gates do **not** fail on them.

| Field | Where | v1 behaviour |
| --- | --- | --- |
| `expected_sources` `{acts,tds,pgs,prs}` | retrieval oracle case | Computes `source_type_recall` / `source_type_recall_by_type` on `RetrievalCaseScore`. Soft default `1.0` when absent/empty. Gate only with `--gate-source-type-recall` (default **off**). |
| `expected_issues` / `expected_benefits` | rules (or future scenario) case | Soft substring checks → `issues_ok` / `benefits_ok` (True when lists empty). Included in `RulesCaseScore.passed` only when the list is non-empty. |
| `forbidden_sources` | retrieval case | Reserved (fixture schema); not enforced by v1 gates. |
| `scenario_id` / `what_if` / `baseline_case_id` | retrieval / scenario case | Null/omit in v1; wiring hooks for paired WHAT-IF variants. |
| `graph_context_present` | report metadata | Set from `/query/data` when `entities` or `relationships` are non-empty. |

Helpers already landed for later packs: `classify_instrument`, `extract_ranked_chunks`,
`source_type_recall`, `citation_set`, `jaccard`, `set_delta`.

## Not yet

- **Client-scenario + WHAT-IF evaluation** — `tax_scenario_oracle.json` is reserved
  (`cases: []`). Intended case shape: Act/TD/PG/PR packs, issues, benefits,
  `variant_must_change` vs `baseline_case_id`.
- **Scenario evaluation CLI** / live scenario oracles — not implemented in v1.
- Enforcing `forbidden_sources` or requiring multi-instrument packs in default gates.
