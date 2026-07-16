#!/usr/bin/env python3
"""Live tax evaluation loop: recon → auto-draft oracle → mode/top_k sweep.

Runs against a reachable LightRAG API (default ``http://localhost:9621``).
Cloud / remote agents cannot reach a laptop API — execute locally.

Phases:
  A) Recon seed questions across modes via ``POST /query/data``
  B) Auto-draft live oracle + light rules from observed chunks
  C) Mode / top_k sweep reusing ``run_retrieval_fitness`` scoring
  D) Write recon/oracle/rules/sweep/recommendation under ``--output-dir``

Usage:
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_live_loop
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_live_loop \\
        --api-url http://localhost:9621 --top-k-values 10,20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lightrag.evaluation.tax_retrieval_fitness import (
    DEFAULT_API_URL,
    DEFAULT_MODES,
    FIXTURES_DIR,
    ServerUnavailableError,
    TaxFitnessClient,
    extract_graph_context_present,
    extract_ranked_chunks,
    gates_passed,
    live_eval_enabled,
    run_retrieval_fitness,
    run_rules_fitness,
    summarize_retrieval,
    summarize_rules,
)

DEFAULT_SEEDS = FIXTURES_DIR / "tax_live_seed_questions.json"
DEFAULT_LIVE_RUNS_DIR = FIXTURES_DIR / "live_runs"

# Section / provision markers commonly surviving AU Act ingest.
_SECTION_MARKER_RE = re.compile(
    r"(?:"
    r"s\s*8-1"
    r"|section\s*8-1"
    r"|Division\s*8"
    r"|Part\s*2"
    r"|8-1"
    r"|TD\s*\d{4}/\d+"
    r"|PG\s*\d{4}/\d+"
    r"|PR\s*\d{4}/\d+"
    r"|Taxation\s+Determination"
    r"|Practice\s+Guide"
    r")",
    re.IGNORECASE,
)

_SNIPPET_CHARS = 400
_RANK_METRICS = (
    "trap_suppress_rate",
    "mean_reciprocal_rank",
    "average_provision_hit_at_k",
    "average_section_hit_at_k",
)


# ---------------------------------------------------------------------------
# Seeds / I/O helpers
# ---------------------------------------------------------------------------


def default_output_dir(now: datetime | None = None) -> Path:
    """Return ``tax_fixtures/live_runs/<UTC timestamp>``."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LIVE_RUNS_DIR / stamp


def load_seed_questions(path: Path) -> list[dict[str, Any]]:
    """Load seed question objects from JSON (``seeds`` list or bare list)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        seeds = payload
    elif isinstance(payload, dict):
        seeds = payload.get("seeds")
    else:
        raise ValueError(f"{path} must be a JSON object or list")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{path} must contain a non-empty seeds list")
    out: list[dict[str, Any]] = []
    for entry in seeds:
        if not isinstance(entry, dict):
            raise ValueError("Each seed must be an object")
        if not str(entry.get("id") or "").strip():
            raise ValueError("Each seed needs an id")
        if not str(entry.get("question") or "").strip():
            raise ValueError(f"Seed {entry.get('id')!r} needs a question")
        intent = str(entry.get("intent") or "provision").strip().casefold()
        if intent not in {"provision", "trap", "crosswalk", "rules"}:
            raise ValueError(
                f"Seed {entry['id']!r} intent must be provision|trap|crosswalk|rules"
            )
        out.append(
            {
                "id": str(entry["id"]),
                "question": str(entry["question"]),
                "intent": intent,
                "anchor_terms": [
                    str(t) for t in (entry.get("anchor_terms") or []) if str(t).strip()
                ],
                "forbidden_hints": [
                    str(t)
                    for t in (entry.get("forbidden_hints") or [])
                    if str(t).strip()
                ],
                "pair_with": (
                    str(entry["pair_with"]).strip()
                    if entry.get("pair_with")
                    else None
                ),
            }
        )
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _chunk_snippet(chunk: dict[str, Any], limit: int = _SNIPPET_CHARS) -> str:
    content = chunk.get("content")
    if isinstance(content, str) and content.strip():
        text = content.strip()
    else:
        headings = chunk.get("content_headings")
        if isinstance(headings, str):
            text = headings.strip()
        elif isinstance(headings, list):
            text = " ".join(str(h) for h in headings if h)
        else:
            text = ""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _headings_as_list(headings: Any) -> list[str]:
    if isinstance(headings, str) and headings.strip():
        return [headings.strip()]
    if isinstance(headings, list):
        return [str(h).strip() for h in headings if str(h).strip()]
    return []


def _searchable_blob(chunk: dict[str, Any]) -> str:
    parts: list[str] = []
    content = chunk.get("content")
    if isinstance(content, str):
        parts.append(content)
    parts.extend(_headings_as_list(chunk.get("content_headings")))
    file_path = chunk.get("file_path")
    if isinstance(file_path, str):
        parts.append(file_path)
    # Recon-normalized rows may only carry a snippet.
    snippet = chunk.get("content_snippet")
    if isinstance(snippet, str):
        parts.append(snippet)
    return "\n".join(parts)


def _term_present(term: str, blob: str) -> bool:
    if not term or not blob:
        return False
    return term.casefold() in blob.casefold()


# ---------------------------------------------------------------------------
# Phase A — Recon
# ---------------------------------------------------------------------------


def summarize_mode_response(
    payload: dict[str, Any],
    *,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    """Normalize one ``/query/data`` response into recon-friendly chunk rows."""
    if payload.get("status") == "failure":
        return {
            "mode": mode,
            "top_k": top_k,
            "status": "failure",
            "message": str(payload.get("message") or ""),
            "graph_context_present": False,
            "chunks": [],
        }
    ranked = extract_ranked_chunks(payload)
    window = ranked[:top_k]
    return {
        "mode": mode,
        "top_k": top_k,
        "status": str(payload.get("status") or "success"),
        "graph_context_present": extract_graph_context_present(payload),
        "chunks": [
            {
                "rank": idx,
                "file_path": chunk.get("file_path"),
                "content_headings": chunk.get("content_headings"),
                "instrument": chunk.get("instrument"),
                "content_snippet": _chunk_snippet(chunk),
            }
            for idx, chunk in enumerate(window, start=1)
        ],
    }


def run_recon(
    client: TaxFitnessClient,
    seeds: Sequence[dict[str, Any]],
    *,
    modes: Sequence[str],
    top_k: int = 10,
) -> dict[str, Any]:
    """Query each seed × mode and return a recon document."""
    queries: list[dict[str, Any]] = []
    for seed in seeds:
        per_mode: list[dict[str, Any]] = []
        for mode in modes:
            payload = client.query_data(
                seed["question"],
                mode=str(mode),
                top_k=top_k,
                chunk_top_k=top_k,
            )
            per_mode.append(
                summarize_mode_response(payload, mode=str(mode), top_k=top_k)
            )
        queries.append(
            {
                "id": seed["id"],
                "question": seed["question"],
                "intent": seed["intent"],
                "anchor_terms": list(seed.get("anchor_terms") or []),
                "forbidden_hints": list(seed.get("forbidden_hints") or []),
                "pair_with": seed.get("pair_with"),
                "modes": per_mode,
            }
        )
    return {
        "api_url": client.api_url,
        "top_k": top_k,
        "modes": list(modes),
        "queries": queries,
    }


# ---------------------------------------------------------------------------
# Phase B — Auto-draft oracle + rules
# ---------------------------------------------------------------------------


def _flat_chunks_for_seed(
    recon: dict[str, Any],
    seed_id: str,
) -> list[dict[str, Any]]:
    """All recon chunks across modes for one seed (preserves mode tag)."""
    out: list[dict[str, Any]] = []
    for query in recon.get("queries") or []:
        if query.get("id") != seed_id:
            continue
        for mode_block in query.get("modes") or []:
            mode = mode_block.get("mode")
            for chunk in mode_block.get("chunks") or []:
                enriched = dict(chunk)
                enriched["_mode"] = mode
                out.append(enriched)
    return out


def collect_anchor_hits(
    anchor_terms: Sequence[str],
    blobs: Sequence[str],
) -> list[str]:
    """Return anchor terms that appear in at least one searchable blob."""
    hits: list[str] = []
    seen: set[str] = set()
    for term in anchor_terms:
        key = term.casefold()
        if key in seen:
            continue
        if any(_term_present(term, blob) for blob in blobs):
            hits.append(term)
            seen.add(key)
    return hits


def collect_section_markers(
    chunks: Sequence[dict[str, Any]],
    *,
    max_markers: int = 8,
) -> list[str]:
    """Derive ``expected_sections`` from headings and regex markers in chunks."""
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(text)

    for chunk in chunks:
        for heading in _headings_as_list(chunk.get("content_headings")):
            if _SECTION_MARKER_RE.search(heading) or len(heading) <= 80:
                _add(heading)
            for match in _SECTION_MARKER_RE.finditer(heading):
                _add(match.group(0))
        blob = _searchable_blob(chunk)
        for match in _SECTION_MARKER_RE.finditer(blob):
            _add(match.group(0))
        if len(found) >= max_markers:
            break
    return found[:max_markers]


def collect_expected_sources(
    chunks: Sequence[dict[str, Any]],
    *,
    max_per_type: int = 3,
) -> dict[str, list[str]]:
    """Build ``expected_sources`` from classified instruments / file paths."""
    buckets: dict[str, list[str]] = {
        "acts": [],
        "tds": [],
        "pgs": [],
        "prs": [],
    }
    type_to_key = {"act": "acts", "td": "tds", "pg": "pgs", "pr": "prs"}
    seen: dict[str, set[str]] = {k: set() for k in buckets}

    for chunk in chunks:
        instrument = str(chunk.get("instrument") or "unknown").casefold()
        key = type_to_key.get(instrument)
        if key is None:
            continue
        if len(buckets[key]) >= max_per_type:
            continue
        candidates: list[str] = []
        file_path = chunk.get("file_path")
        if isinstance(file_path, str) and file_path.strip():
            candidates.append(file_path.strip())
            name = Path(file_path).name
            if name and name not in candidates:
                candidates.append(name)
        for heading in _headings_as_list(chunk.get("content_headings")):
            for match in _SECTION_MARKER_RE.finditer(heading):
                candidates.append(match.group(0))
        for candidate in candidates:
            folded = candidate.casefold()
            if folded in seen[key]:
                continue
            seen[key].add(folded)
            buckets[key].append(candidate)
            if len(buckets[key]) >= max_per_type:
                break
    return buckets


def collect_forbidden_from_trap(
    trap_chunks: Sequence[dict[str, Any]],
    provision_blobs: Sequence[str],
    forbidden_hints: Sequence[str],
    *,
    max_forbidden: int = 6,
) -> list[str]:
    """Forbidden strings: hints present in recon, or trap-only snippet markers."""
    forbidden: list[str] = []
    seen: set[str] = set()

    def _add(term: str) -> None:
        text = term.strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        forbidden.append(text)

    trap_blobs = [_searchable_blob(c) for c in trap_chunks]
    all_trap_blob = "\n".join(trap_blobs)

    for hint in forbidden_hints:
        if _term_present(hint, all_trap_blob) or any(
            _term_present(hint, b) for b in provision_blobs
        ):
            _add(hint)

    for chunk in trap_chunks:
        blob = _searchable_blob(chunk)
        if not blob.strip():
            continue
        for match in _SECTION_MARKER_RE.finditer(blob):
            marker = match.group(0)
            if not any(_term_present(marker, prov) for prov in provision_blobs):
                _add(marker)
        snippet = (chunk.get("content_snippet") or "").strip()
        if snippet and len(snippet) >= 24:
            head = snippet[:48].strip()
            if head and not any(_term_present(head, prov) for prov in provision_blobs):
                _add(head)
        if len(forbidden) >= max_forbidden:
            break
    return forbidden[:max_forbidden]


def draft_oracle_case_from_recon(
    seed: dict[str, Any],
    recon: dict[str, Any],
    *,
    seeds_by_id: dict[str, dict[str, Any]] | None = None,
    default_modes: Sequence[str] | None = None,
    default_top_k: int = 10,
) -> tuple[dict[str, Any], bool, str]:
    """Draft one retrieval oracle case. Returns (case, enabled, skip_reason)."""
    intent = seed["intent"]
    modes = list(default_modes or DEFAULT_MODES)
    seeds_by_id = seeds_by_id or {}

    if intent == "rules":
        case = {
            "id": f"live_{seed['id']}",
            "enabled": False,
            "intent": intent,
            "question": seed["question"],
            "expected_sections": [],
            "expected_chunk_contains": [],
            "forbidden_chunk_contains": [],
            "modes": modes,
            "top_k": default_top_k,
            "draft_note": "rules intent — see live_rules.generated.json",
        }
        return case, False, "rules_intent"

    chunks = _flat_chunks_for_seed(recon, seed["id"])
    blobs = [_searchable_blob(c) for c in chunks]

    expected_contains: list[str] = []
    expected_sections: list[str] = []
    expected_sources: dict[str, list[str]] = {
        "acts": [],
        "tds": [],
        "pgs": [],
        "prs": [],
    }
    forbidden: list[str] = []

    if intent in {"provision", "crosswalk"}:
        expected_contains = collect_anchor_hits(seed.get("anchor_terms") or [], blobs)
        expected_sections = collect_section_markers(chunks)
        expected_sources = collect_expected_sources(chunks)
        forbidden = [
            h
            for h in (seed.get("forbidden_hints") or [])
            if any(_term_present(h, b) for b in blobs)
        ]
    elif intent == "trap":
        pair_id = seed.get("pair_with")
        pair_chunks = _flat_chunks_for_seed(recon, pair_id) if pair_id else []
        pair_blobs = [_searchable_blob(c) for c in pair_chunks]
        pair_seed = seeds_by_id.get(pair_id or "")
        pair_anchors = list((pair_seed or {}).get("anchor_terms") or [])
        expected_contains = collect_anchor_hits(
            pair_anchors, pair_blobs
        ) or collect_anchor_hits(pair_anchors, blobs)
        expected_sections = collect_section_markers(pair_chunks or chunks)
        expected_sources = collect_expected_sources(pair_chunks or chunks)
        forbidden = collect_forbidden_from_trap(
            chunks,
            pair_blobs,
            seed.get("forbidden_hints") or [],
        )
    else:
        return (
            {
                "id": f"live_{seed['id']}",
                "enabled": False,
                "intent": intent,
                "question": seed["question"],
                "expected_sections": [],
                "expected_chunk_contains": [],
                "forbidden_chunk_contains": [],
                "modes": modes,
                "top_k": default_top_k,
            },
            False,
            f"unsupported_intent:{intent}",
        )

    if intent == "crosswalk":
        sources_out = expected_sources
    else:
        sources_out = {k: v for k, v in expected_sources.items() if v}

    enabled = bool(expected_contains or expected_sections)
    skip_reason = "" if enabled else "no_expected_anchors_in_recon"

    case: dict[str, Any] = {
        "id": f"live_{seed['id']}",
        "enabled": enabled,
        "intent": intent,
        "question": seed["question"],
        "expected_sections": expected_sections,
        "expected_chunk_contains": expected_contains,
        "forbidden_chunk_contains": forbidden,
        "modes": modes,
        "mode": "mix",
        "top_k": default_top_k,
    }
    if sources_out:
        case["expected_sources"] = sources_out
    if seed.get("pair_with"):
        case["pair_with"] = seed["pair_with"]
    if skip_reason:
        case["draft_note"] = skip_reason
    return case, enabled, skip_reason


def draft_rules_case_from_recon(
    seed: dict[str, Any],
    recon: dict[str, Any],
    *,
    default_mode: str = "mix",
) -> tuple[dict[str, Any], bool, str]:
    """Draft one light rules case from recon for ``rules`` intents."""
    if seed["intent"] != "rules":
        return (
            {
                "id": f"live_rule_{seed['id']}",
                "enabled": False,
                "question": seed["question"],
                "mode": default_mode,
                "expected_answer_contains": [],
                "forbidden_answer_contains": [],
                "expected_citation_contains": [],
                "forbidden_citation_contains": [],
            },
            False,
            "not_rules_intent",
        )

    chunks = _flat_chunks_for_seed(recon, seed["id"])
    blobs = [_searchable_blob(c) for c in chunks]
    answer_contains = collect_anchor_hits(seed.get("anchor_terms") or [], blobs)
    citation_contains = collect_section_markers(chunks, max_markers=4)
    forbidden_answer = [
        h
        for h in (seed.get("forbidden_hints") or [])
        if any(_term_present(h, b) for b in blobs)
    ]
    enabled = bool(answer_contains or citation_contains)
    skip_reason = "" if enabled else "no_rules_anchors_in_recon"
    case = {
        "id": f"live_rule_{seed['id']}",
        "enabled": enabled,
        "question": seed["question"],
        "mode": default_mode,
        "expected_answer_contains": answer_contains,
        "forbidden_answer_contains": forbidden_answer,
        "expected_citation_contains": citation_contains,
        "forbidden_citation_contains": [],
    }
    if skip_reason:
        case["draft_note"] = skip_reason
    return case, enabled, skip_reason


def draft_live_oracle(
    seeds: Sequence[dict[str, Any]],
    recon: dict[str, Any],
    *,
    default_modes: Sequence[str] | None = None,
    default_top_k: int = 10,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build ``live_oracle.generated.json`` payload from recon.

    Returns (oracle_payload, counts) where counts has enabled/skipped.
    """
    seeds_by_id = {s["id"]: s for s in seeds}
    cases: list[dict[str, Any]] = []
    enabled_n = 0
    skipped_n = 0
    for seed in seeds:
        if seed["intent"] == "rules":
            skipped_n += 1
            continue
        case, enabled, _reason = draft_oracle_case_from_recon(
            seed,
            recon,
            seeds_by_id=seeds_by_id,
            default_modes=default_modes,
            default_top_k=default_top_k,
        )
        cases.append(case)
        if enabled:
            enabled_n += 1
        else:
            skipped_n += 1

    oracle = {
        "description": (
            "Auto-drafted live retrieval oracle from recon against a real index. "
            "Review expected_* strings before treating gates as authoritative."
        ),
        "defaults": {
            "top_k": default_top_k,
            "modes": list(default_modes or DEFAULT_MODES),
        },
        "cases": cases,
    }
    return oracle, {"enabled": enabled_n, "skipped": skipped_n}


def draft_live_rules(
    seeds: Sequence[dict[str, Any]],
    recon: dict[str, Any],
    *,
    default_mode: str = "mix",
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build ``live_rules.generated.json`` from rules-intent seeds + recon."""
    cases: list[dict[str, Any]] = []
    enabled_n = 0
    skipped_n = 0
    for seed in seeds:
        if seed["intent"] != "rules":
            continue
        case, enabled, _reason = draft_rules_case_from_recon(
            seed, recon, default_mode=default_mode
        )
        cases.append(case)
        if enabled:
            enabled_n += 1
        else:
            skipped_n += 1

    rules = {
        "description": (
            "Auto-drafted light rules/guidance checks from recon. "
            "Deterministic substring judges only."
        ),
        "defaults": {"mode": default_mode},
        "cases": cases,
    }
    return rules, {"enabled": enabled_n, "skipped": skipped_n}


# ---------------------------------------------------------------------------
# Phase C — Mode / top_k sweep
# ---------------------------------------------------------------------------


def build_mode_sets(modes: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Each single mode plus the full matrix."""
    unique = [m for m in modes if m]
    seen: set[str] = set()
    ordered: list[str] = []
    for mode in unique:
        if mode in seen:
            continue
        seen.add(mode)
        ordered.append(mode)
    sets: list[tuple[str, list[str]]] = [(m, [m]) for m in ordered]
    if len(ordered) > 1:
        sets.append(("matrix:" + ",".join(ordered), list(ordered)))
    return sets


def rank_key_from_summary(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    """Sort key: trap, MRR, provision Hit@K, section Hit@K (all descending)."""
    return (
        float(summary.get("trap_suppress_rate") or 0.0),
        float(summary.get("mean_reciprocal_rank") or 0.0),
        float(summary.get("average_provision_hit_at_k") or 0.0),
        float(summary.get("average_section_hit_at_k") or 0.0),
    )


def rank_configurations(
    configs: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return configs sorted best-first by the retrieval ranking tuple."""
    return sorted(
        configs,
        key=lambda c: rank_key_from_summary(c.get("retrieval_summary") or {}),
        reverse=True,
    )


def run_mode_topk_sweep(
    client: TaxFitnessClient,
    oracle: dict[str, Any],
    *,
    modes: Sequence[str],
    top_k_values: Sequence[int],
    min_section_hit: float,
    min_provision_hit: float,
    min_trap_suppress: float,
    min_mrr: float,
) -> list[dict[str, Any]]:
    """Score every (mode-set × top_k) using generated oracle cases."""
    results: list[dict[str, Any]] = []
    for label, mode_list in build_mode_sets(modes):
        for top_k in top_k_values:
            oracle_variant = json.loads(json.dumps(oracle))
            defaults = dict(oracle_variant.get("defaults") or {})
            defaults["top_k"] = int(top_k)
            defaults["modes"] = list(mode_list)
            oracle_variant["defaults"] = defaults
            for case in oracle_variant.get("cases") or []:
                if isinstance(case, dict):
                    case["top_k"] = int(top_k)
                    if "modes" in case:
                        case["modes"] = list(mode_list)

            scores = run_retrieval_fitness(client, oracle_variant, modes=mode_list)
            summary = summarize_retrieval(scores)
            passed, failures = gates_passed(
                summary,
                None,
                min_section_hit=min_section_hit,
                min_provision_hit=min_provision_hit,
                min_trap_suppress=min_trap_suppress,
                min_mrr=min_mrr,
                min_rules_pass_rate=1.0,
                check_rules=False,
            )
            results.append(
                {
                    "config_id": f"{label}|top_k={top_k}",
                    "mode_set_label": label,
                    "modes": list(mode_list),
                    "top_k": int(top_k),
                    "retrieval_summary": summary,
                    "gates_passed": passed,
                    "gate_failures": failures,
                    "cases": [asdict(s) for s in scores],
                }
            )
    return rank_configurations(results)


# ---------------------------------------------------------------------------
# Phase D — Diagnosis / recommendation
# ---------------------------------------------------------------------------


def diagnose_from_metrics(
    best: dict[str, Any] | None,
    recon: dict[str, Any],
    oracle_counts: dict[str, int],
) -> list[str]:
    """Human-readable diagnosis bullets from best config + recon shape."""
    bullets: list[str] = []
    enabled = int(oracle_counts.get("enabled") or 0)
    skipped = int(oracle_counts.get("skipped") or 0)
    bullets.append(
        f"Oracle draft: {enabled} enabled, {skipped} skipped "
        f"(skipped = no anchors in recon or rules-only seeds)."
    )

    total_chunks = 0
    headed = 0
    for query in recon.get("queries") or []:
        for mode_block in query.get("modes") or []:
            for chunk in mode_block.get("chunks") or []:
                total_chunks += 1
                if _headings_as_list(chunk.get("content_headings")):
                    headed += 1
    if total_chunks == 0:
        bullets.append(
            "Recon returned no chunks — index may be empty or API returned failures."
        )
    else:
        coverage = headed / total_chunks
        if coverage < 0.25:
            bullets.append(
                "content_headings sparse in recon → section Hit@K often stays low "
                "even when provision substring hits are high."
            )
        else:
            bullets.append(
                f"Heading coverage in recon ≈ {coverage:.0%} of chunks — "
                "section markers should be scorable when present in headings/body."
            )

    if not best:
        bullets.append("No sweep configs ranked (oracle may have zero enabled cases).")
        return bullets

    summary = best.get("retrieval_summary") or {}
    section = float(summary.get("average_section_hit_at_k") or 0.0)
    provision = float(summary.get("average_provision_hit_at_k") or 0.0)
    trap = float(summary.get("trap_suppress_rate") or 0.0)
    mrr = float(summary.get("mean_reciprocal_rank") or 0.0)

    if provision >= 0.7 and section < 0.4:
        bullets.append(
            "Provision Hit@K high but section Hit@K low → content OK, headings / "
            "section markers missing or drifted vs oracle."
        )
    elif provision < 0.4 and section < 0.4:
        bullets.append(
            "Both provision and section Hit@K low → retrieved chunks may not "
            "contain expected Act/TD language, or seed anchors need tighter curation."
        )
    elif provision >= 0.7 and section >= 0.5:
        bullets.append(
            "Provision and section Hits are healthy on the drafted oracle — "
            "prefer the recommended mode/top_k under current settings."
        )

    if trap >= 0.99:
        bullets.append("Trap suppress is strong on drafted forbidden strings.")
    elif trap < 0.5:
        bullets.append(
            "Trap suppress weak → keyword/penalty chunks may outrank provision "
            "anchors for bare 'deduction' queries."
        )

    if mrr >= 0.5:
        bullets.append(f"MRR {mrr:.3f} — expected anchors often rank near the top.")
    else:
        bullets.append(
            f"MRR {mrr:.3f} — expected anchors appear but rank poorly; try mix "
            "with reranker or inspect naive vs hybrid ordering."
        )

    if best.get("gates_passed"):
        bullets.append("Best config passes default informational gates.")
    else:
        failures = best.get("gate_failures") or []
        bullets.append(
            "Best config fails informational gates: "
            + ("; ".join(failures) if failures else "unknown")
        )
    return bullets


def build_recommendation(
    *,
    api_url: str,
    ranked: Sequence[dict[str, Any]],
    recon: dict[str, Any],
    oracle_counts: dict[str, int],
    rules_counts: dict[str, int],
    rules_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    best = ranked[0] if ranked else None
    metric_table = [
        {
            "config_id": c.get("config_id"),
            "modes": c.get("modes"),
            "top_k": c.get("top_k"),
            "trap_suppress_rate": (c.get("retrieval_summary") or {}).get(
                "trap_suppress_rate"
            ),
            "mean_reciprocal_rank": (c.get("retrieval_summary") or {}).get(
                "mean_reciprocal_rank"
            ),
            "average_provision_hit_at_k": (c.get("retrieval_summary") or {}).get(
                "average_provision_hit_at_k"
            ),
            "average_section_hit_at_k": (c.get("retrieval_summary") or {}).get(
                "average_section_hit_at_k"
            ),
            "gates_passed": c.get("gates_passed"),
        }
        for c in ranked
    ]
    diagnosis = diagnose_from_metrics(best, recon, oracle_counts)
    return {
        "api_url": api_url,
        "best_config_id": (best or {}).get("config_id"),
        "best_modes": (best or {}).get("modes"),
        "best_top_k": (best or {}).get("top_k"),
        "best_metrics": (best or {}).get("retrieval_summary"),
        "best_gates_passed": (best or {}).get("gates_passed"),
        "best_gate_failures": (best or {}).get("gate_failures") or [],
        "ranking_order": list(_RANK_METRICS),
        "metric_table": metric_table,
        "oracle_draft_counts": oracle_counts,
        "rules_draft_counts": rules_counts,
        "rules_summary": rules_summary,
        "diagnosis": diagnosis,
        "note": (
            "Gate failures are informational for this diagnostics loop. "
            "Cloud agents cannot validate a localhost API — run on a host that "
            "reaches the LightRAG service."
        ),
    }


def print_human_summary(recommendation: dict[str, Any]) -> None:
    print("LightRAG tax live evaluation loop")
    print(f"API: {recommendation.get('api_url')}")
    print(
        f"Best config: {recommendation.get('best_config_id')} "
        f"(modes={recommendation.get('best_modes')}, "
        f"top_k={recommendation.get('best_top_k')})"
    )
    metrics = recommendation.get("best_metrics") or {}
    if metrics:
        print(
            f"  trap={metrics.get('trap_suppress_rate', 0):.3f}  "
            f"MRR={metrics.get('mean_reciprocal_rank', 0):.3f}  "
            f"provision={metrics.get('average_provision_hit_at_k', 0):.3f}  "
            f"section={metrics.get('average_section_hit_at_k', 0):.3f}"
        )
    print("Metric table (ranked):")
    for row in recommendation.get("metric_table") or []:
        print(
            f"  {row.get('config_id')}: "
            f"trap={row.get('trap_suppress_rate', 0):.3f} "
            f"MRR={row.get('mean_reciprocal_rank', 0):.3f} "
            f"prov={row.get('average_provision_hit_at_k', 0):.3f} "
            f"sec={row.get('average_section_hit_at_k', 0):.3f} "
            f"gates={'PASS' if row.get('gates_passed') else 'FAIL'}"
        )
    print("Diagnosis:")
    for bullet in recommendation.get("diagnosis") or []:
        print(f"  - {bullet}")
    rules = recommendation.get("rules_summary")
    if rules and rules.get("queries", 0) > 0:
        print(
            f"Rules on best mode: pass_rate={rules.get('pass_rate', 0):.3f} "
            f"({rules.get('queries')} queries)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live tax recon → oracle draft → mode/top_k sweep against LightRAG."
        )
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LIGHTRAG_API_URL", DEFAULT_API_URL),
        help=f"LightRAG API base URL (default: {DEFAULT_API_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LIGHTRAG_API_KEY"),
        help="Optional API key (or set LIGHTRAG_API_KEY)",
    )
    parser.add_argument(
        "--seeds",
        default=str(DEFAULT_SEEDS),
        help="Path to seed questions JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: tax_fixtures/live_runs/<timestamp>)",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated retrieval modes (default: naive,local,hybrid,mix)",
    )
    parser.add_argument(
        "--top-k-values",
        default="10,20",
        help="Comma-separated top_k values for sweep (default: 10,20)",
    )
    parser.add_argument(
        "--recon-top-k",
        type=int,
        default=10,
        help="top_k used during recon queries (default: 10)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip rules phase after sweep (same as --skip-rules)",
    )
    parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Skip optional rules run on best retrieval mode",
    )
    parser.add_argument(
        "--allow-ungated",
        action="store_true",
        help="Run even when TAX_EVAL_LIVE / LIGHTRAG_RUN_INTEGRATION are unset",
    )
    parser.add_argument(
        "--min-section-hit",
        type=float,
        default=_env_float("TAX_EVAL_MIN_SECTION_HIT", 0.5),
    )
    parser.add_argument(
        "--min-provision-hit",
        type=float,
        default=_env_float("TAX_EVAL_MIN_PROVISION_HIT", 0.5),
    )
    parser.add_argument(
        "--min-trap-suppress",
        type=float,
        default=_env_float("TAX_EVAL_MIN_TRAP_SUPPRESS", 1.0),
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=_env_float("TAX_EVAL_MIN_MRR", 0.3),
    )
    parser.add_argument(
        "--min-rules-pass-rate",
        type=float,
        default=_env_float("TAX_EVAL_MIN_RULES_PASS_RATE", 1.0),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.allow_ungated and not live_eval_enabled():
        print(
            "Tax live loop is gated. Set TAX_EVAL_LIVE=true "
            "(or LIGHTRAG_RUN_INTEGRATION=true), or pass --allow-ungated.",
            file=sys.stderr,
        )
        return 1

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        print("At least one --modes value is required.", file=sys.stderr)
        return 1

    try:
        top_k_values = [
            int(x.strip())
            for x in str(args.top_k_values).split(",")
            if x.strip()
        ]
    except ValueError:
        print("--top-k-values must be comma-separated integers.", file=sys.stderr)
        return 1
    if not top_k_values:
        print("At least one --top-k-values entry is required.", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else default_output_dir()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds_path = Path(args.seeds).expanduser()
    try:
        seeds = load_seed_questions(seeds_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load seeds from {seeds_path}: {exc}", file=sys.stderr)
        return 1

    client = TaxFitnessClient(api_url=args.api_url, api_key=args.api_key)
    try:
        client.health_check()
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Start the API (e.g. lightrag-server on :9621) and ensure the tax "
            "corpus is ingested before re-running.",
            file=sys.stderr,
        )
        return 1

    # --- Phase A ---
    print(f"Phase A — Recon ({len(seeds)} seeds × {len(modes)} modes)…")
    try:
        recon = run_recon(
            client,
            seeds,
            modes=modes,
            top_k=int(args.recon_top_k),
        )
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"Recon failed: {exc}", file=sys.stderr)
        return 1
    _write_json(output_dir / "recon.json", recon)

    # --- Phase B ---
    print("Phase B — Auto-draft live oracle + rules…")
    oracle, oracle_counts = draft_live_oracle(
        seeds,
        recon,
        default_modes=modes,
        default_top_k=top_k_values[0],
    )
    rules, rules_counts = draft_live_rules(seeds, recon, default_mode="mix")
    _write_json(output_dir / "live_oracle.generated.json", oracle)
    _write_json(output_dir / "live_rules.generated.json", rules)
    print(
        f"  Oracle cases enabled={oracle_counts['enabled']} "
        f"skipped={oracle_counts['skipped']}"
    )
    print(
        f"  Rules cases enabled={rules_counts['enabled']} "
        f"skipped={rules_counts['skipped']}"
    )

    if oracle_counts["enabled"] == 0:
        print(
            "No enabled oracle cases after recon — sweep skipped. "
            "Check that the index returns chunks matching seed anchor_terms.",
            file=sys.stderr,
        )
        recommendation = build_recommendation(
            api_url=client.api_url,
            ranked=[],
            recon=recon,
            oracle_counts=oracle_counts,
            rules_counts=rules_counts,
            rules_summary=None,
        )
        sweep_payload = {"configs": [], "note": "no enabled oracle cases"}
        _write_json(output_dir / "sweep.json", sweep_payload)
        _write_json(output_dir / "recommendation.json", recommendation)
        print_human_summary(recommendation)
        print(f"Wrote outputs under {output_dir}")
        return 0

    # --- Phase C ---
    print(f"Phase C — Sweep mode sets × top_k={list(top_k_values)}…")
    try:
        ranked = run_mode_topk_sweep(
            client,
            oracle,
            modes=modes,
            top_k_values=top_k_values,
            min_section_hit=args.min_section_hit,
            min_provision_hit=args.min_provision_hit,
            min_trap_suppress=args.min_trap_suppress,
            min_mrr=args.min_mrr,
        )
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (ValueError, RuntimeError) as exc:
        print(f"Sweep failed: {exc}", file=sys.stderr)
        return 1

    sweep_payload = {
        "api_url": client.api_url,
        "modes": modes,
        "top_k_values": list(top_k_values),
        "ranking_order": list(_RANK_METRICS),
        "configs": ranked,
    }
    _write_json(output_dir / "sweep.json", sweep_payload)

    skip_rules = bool(args.skip_rules or args.retrieval_only)
    rules_summary: dict[str, Any] | None = None
    if not skip_rules and rules_counts.get("enabled", 0) > 0 and ranked:
        best_modes = ranked[0].get("modes") or ["mix"]
        rules_mode = str(best_modes[0])
        rules_variant = json.loads(json.dumps(rules))
        rules_variant["defaults"] = {"mode": rules_mode}
        for case in rules_variant.get("cases") or []:
            if isinstance(case, dict):
                case["mode"] = rules_mode
        print(f"Phase C+ — Rules on best mode={rules_mode}…")
        try:
            rules_scores = run_rules_fitness(client, rules_variant)
            rules_summary = summarize_rules(rules_scores)
            _, rule_failures = gates_passed(
                ranked[0].get("retrieval_summary") or {},
                rules_summary,
                min_section_hit=args.min_section_hit,
                min_provision_hit=args.min_provision_hit,
                min_trap_suppress=args.min_trap_suppress,
                min_mrr=args.min_mrr,
                min_rules_pass_rate=args.min_rules_pass_rate,
                check_rules=True,
            )
            if rule_failures:
                print(
                    "  Rules informational gate notes: "
                    + "; ".join(rule_failures)
                )
        except ServerUnavailableError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except (ValueError, RuntimeError) as exc:
            print(f"Rules run failed (continuing): {exc}", file=sys.stderr)
            rules_summary = {"queries": 0, "pass_rate": 0.0, "error": str(exc)}

    # --- Phase D ---
    recommendation = build_recommendation(
        api_url=client.api_url,
        ranked=ranked,
        recon=recon,
        oracle_counts=oracle_counts,
        rules_counts=rules_counts,
        rules_summary=rules_summary,
    )
    _write_json(output_dir / "recommendation.json", recommendation)
    print_human_summary(recommendation)
    print(f"Wrote outputs under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
