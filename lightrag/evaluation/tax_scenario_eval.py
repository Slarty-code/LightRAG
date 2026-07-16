#!/usr/bin/env python3
"""Live client-scenario + WHAT-IF evaluation harness (Part 2 first slice).

Phases:
  A) Recon seeds via ``POST /query/data``
  B) Draft ``scenario_oracle.generated.json`` from hints that appear in recon
  C) Score packs: retrieval metrics + ``POST /query`` answer path
  D) WHAT-IF citation / issue-hit set deltas vs baseline
  E) Write recommendation / scenario report (emphasize answer metrics)

Usage:
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_scenario_eval
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_scenario_eval \\
        --api-url http://localhost:9621 --label sonnet --modes hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lightrag.evaluation.tax_model_ab import _slug, extract_model_tags
from lightrag.evaluation.tax_retrieval_fitness import (
    DEFAULT_API_URL,
    FIXTURES_DIR,
    ServerUnavailableError,
    TaxFitnessClient,
    case_enabled,
    citation_set,
    extract_answer_and_citation_blob,
    extract_graph_context_present,
    extract_ranked_chunks,
    hit_fraction_at_k,
    live_eval_enabled,
    set_delta,
    source_type_recall,
    substring_present,
    trap_suppressed,
    _env_float,
)

DEFAULT_SEEDS = FIXTURES_DIR / "tax_scenario_seeds.json"
DEFAULT_LIVE_RUNS_DIR = FIXTURES_DIR / "live_runs"
DEFAULT_MODES = ("hybrid",)
DEFAULT_TOP_K = 50
DEFAULT_CHUNK_TOP_K = 50
_SNIPPET_CHARS = 400
_MIN_ANCHOR_HITS_TO_ENABLE = 1

_SECTION_MARKER_RE = re.compile(
    r"(?:"
    r"s\s*8-1"
    r"|section\s*8-1"
    r"|Division\s*8"
    r"|Part\s*[24]"
    r"|8-1"
    r"|TD\s*\d{4}/\d+"
    r"|PG\s*\d{4}/\d+"
    r"|PR\s*\d{4}/\d+"
    r"|Taxation\s+Determination"
    r"|Practice\s+Guide"
    r"|Product\s+Ruling"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Seeds / I/O
# ---------------------------------------------------------------------------


def default_output_dir(
    label: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Return ``tax_fixtures/live_runs/scenario_<label_>timestamp``."""
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    label_slug = _slug(label) if label else None
    name = f"scenario_{label_slug}_{stamp}" if label_slug else f"scenario_{stamp}"
    return DEFAULT_LIVE_RUNS_DIR / name


def load_scenario_seeds(path: Path) -> list[dict[str, Any]]:
    """Load scenario seed objects from JSON (``seeds`` list or bare list)."""
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
        seed_id = str(entry.get("id") or "").strip()
        if not seed_id:
            raise ValueError("Each seed needs an id")
        question = str(entry.get("question") or "").strip()
        if not question:
            raise ValueError(f"Seed {seed_id!r} needs a question")
        intent = str(entry.get("intent") or "scenario").strip().casefold()
        if intent not in {"scenario", "what_if"}:
            raise ValueError(
                f"Seed {seed_id!r} intent must be scenario|what_if (got {intent!r})"
            )
        sources_hints = entry.get("expected_sources_hints") or {}
        if not isinstance(sources_hints, dict):
            sources_hints = {}
        out.append(
            {
                "id": seed_id,
                "scenario_id": (
                    str(entry["scenario_id"]).strip()
                    if entry.get("scenario_id")
                    else None
                ),
                "client_facts": (
                    str(entry["client_facts"]).strip()
                    if entry.get("client_facts")
                    else None
                ),
                "question": question,
                "what_if": (
                    str(entry["what_if"]).strip() if entry.get("what_if") else None
                ),
                "baseline_case_id": (
                    str(entry["baseline_case_id"]).strip()
                    if entry.get("baseline_case_id")
                    else None
                ),
                "intent": intent,
                "anchor_terms": [
                    str(t) for t in (entry.get("anchor_terms") or []) if str(t).strip()
                ],
                "expected_sources_hints": {
                    "acts": [str(x) for x in (sources_hints.get("acts") or []) if str(x).strip()],
                    "tds": [str(x) for x in (sources_hints.get("tds") or []) if str(x).strip()],
                    "pgs": [str(x) for x in (sources_hints.get("pgs") or []) if str(x).strip()],
                    "prs": [str(x) for x in (sources_hints.get("prs") or []) if str(x).strip()],
                },
                "expected_issues_hints": [
                    str(t)
                    for t in (entry.get("expected_issues_hints") or [])
                    if str(t).strip()
                ],
                "expected_benefits_hints": [
                    str(t)
                    for t in (entry.get("expected_benefits_hints") or [])
                    if str(t).strip()
                ],
                "forbidden_hints": [
                    str(t)
                    for t in (entry.get("forbidden_hints") or [])
                    if str(t).strip()
                ],
                "variant_must_change_hints": [
                    str(t)
                    for t in (entry.get("variant_must_change_hints") or [])
                    if str(t).strip()
                ],
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
    snippet = chunk.get("content_snippet")
    if isinstance(snippet, str):
        parts.append(snippet)
    return "\n".join(parts)


def _term_present(term: str, blob: str) -> bool:
    if not term or not blob:
        return False
    return term.casefold() in blob.casefold()


def _hints_present(hints: Sequence[str], blobs: Sequence[str]) -> list[str]:
    """Return hints that appear in at least one blob (case-insensitive)."""
    hits: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        key = hint.casefold()
        if key in seen:
            continue
        if any(_term_present(hint, blob) for blob in blobs):
            hits.append(hint)
            seen.add(key)
    return hits


def collect_section_markers(
    chunks: Sequence[dict[str, Any]],
    *,
    max_markers: int = 8,
) -> list[str]:
    """Derive section-style markers from headings / content regex hits."""
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
    top_k: int = DEFAULT_TOP_K,
    chunk_top_k: int = DEFAULT_CHUNK_TOP_K,
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
                chunk_top_k=chunk_top_k,
            )
            per_mode.append(
                summarize_mode_response(payload, mode=str(mode), top_k=top_k)
            )
        queries.append(
            {
                "id": seed["id"],
                "scenario_id": seed.get("scenario_id"),
                "question": seed["question"],
                "intent": seed["intent"],
                "what_if": seed.get("what_if"),
                "baseline_case_id": seed.get("baseline_case_id"),
                "anchor_terms": list(seed.get("anchor_terms") or []),
                "forbidden_hints": list(seed.get("forbidden_hints") or []),
                "modes": per_mode,
            }
        )
    return {
        "api_url": client.api_url,
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "modes": list(modes),
        "queries": queries,
    }


def _flat_chunks_for_seed(
    recon: dict[str, Any],
    seed_id: str,
) -> list[dict[str, Any]]:
    """All recon chunks across modes for one seed."""
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


def _graph_present_for_seed(recon: dict[str, Any], seed_id: str) -> bool:
    for query in recon.get("queries") or []:
        if query.get("id") != seed_id:
            continue
        for mode_block in query.get("modes") or []:
            if mode_block.get("graph_context_present"):
                return True
    return False


# ---------------------------------------------------------------------------
# Phase B — Draft oracle
# ---------------------------------------------------------------------------


def draft_scenario_case_from_recon(
    seed: dict[str, Any],
    recon: dict[str, Any],
    *,
    default_modes: Sequence[str] | None = None,
    default_top_k: int = DEFAULT_TOP_K,
    min_anchor_hits: int = _MIN_ANCHOR_HITS_TO_ENABLE,
) -> tuple[dict[str, Any], bool, str]:
    """Draft one scenario oracle case. Returns (case, enabled, skip_reason)."""
    modes = list(default_modes or DEFAULT_MODES)
    chunks = _flat_chunks_for_seed(recon, seed["id"])
    blobs = [_searchable_blob(c) for c in chunks]
    combined = "\n".join(blobs)

    anchor_hits = _hints_present(seed.get("anchor_terms") or [], blobs)
    expected_chunk_contains = list(anchor_hits)
    expected_sections = collect_section_markers(chunks)

    sources_hints = seed.get("expected_sources_hints") or {}
    expected_sources: dict[str, list[str]] = {}
    for key in ("acts", "tds", "pgs", "prs"):
        hits = _hints_present(sources_hints.get(key) or [], blobs)
        if hits:
            expected_sources[key] = hits

    expected_issues = _hints_present(seed.get("expected_issues_hints") or [], blobs)
    expected_benefits = _hints_present(
        seed.get("expected_benefits_hints") or [], blobs
    )
    forbidden = _hints_present(seed.get("forbidden_hints") or [], blobs)

    # Trap / negative seeds (no anchors): enable when forbidden hints appear so
    # trap_suppress scoring can run; they stay answer-light.
    is_trap_style = not (seed.get("anchor_terms") or []) and bool(
        seed.get("forbidden_hints")
    )
    if is_trap_style:
        enabled = bool(forbidden) or bool(chunks)
        skip_reason = "" if enabled else "trap_seed_no_recon_chunks"
    else:
        hit_count = len(anchor_hits) + len(expected_sections)
        enabled = hit_count >= min_anchor_hits
        skip_reason = "" if enabled else "insufficient_anchors_in_recon"

    case: dict[str, Any] = {
        "id": seed["id"],
        "enabled": enabled,
        "scenario_id": seed.get("scenario_id"),
        "client_facts": seed.get("client_facts"),
        "question": seed["question"],
        "what_if": seed.get("what_if"),
        "baseline_case_id": seed.get("baseline_case_id"),
        "intent": seed["intent"],
        "expected_sections": expected_sections,
        "expected_chunk_contains": expected_chunk_contains,
        "forbidden_chunk_contains": forbidden,
        "expected_sources": expected_sources,
        "expected_issues": expected_issues,
        "expected_benefits": expected_benefits,
        "forbidden_answer_contains": list(forbidden) if not is_trap_style else [],
        "variant_must_change_hints": list(
            seed.get("variant_must_change_hints") or []
        ),
        "modes": modes,
        "mode": modes[0] if modes else "hybrid",
        "top_k": default_top_k,
        "graph_context_present_in_recon": _graph_present_for_seed(recon, seed["id"]),
        "anchor_hits_in_recon": anchor_hits,
        "recon_blob_chars": len(combined),
    }
    if skip_reason:
        case["draft_note"] = skip_reason
    return case, enabled, skip_reason


def draft_scenario_oracle(
    seeds: Sequence[dict[str, Any]],
    recon: dict[str, Any],
    *,
    default_modes: Sequence[str] | None = None,
    default_top_k: int = DEFAULT_TOP_K,
    min_anchor_hits: int = _MIN_ANCHOR_HITS_TO_ENABLE,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build ``scenario_oracle.generated.json`` from recon + seed hints."""
    cases: list[dict[str, Any]] = []
    enabled_n = 0
    skipped_n = 0
    for seed in seeds:
        case, enabled, _reason = draft_scenario_case_from_recon(
            seed,
            recon,
            default_modes=default_modes,
            default_top_k=default_top_k,
            min_anchor_hits=min_anchor_hits,
        )
        cases.append(case)
        if enabled:
            enabled_n += 1
        else:
            skipped_n += 1

    oracle = {
        "description": (
            "Auto-drafted scenario oracle from live recon. "
            "expected_* filled only from hints that appeared in retrieved chunks. "
            "Review before treating gates as authoritative."
        ),
        "defaults": {
            "top_k": default_top_k,
            "modes": list(default_modes or DEFAULT_MODES),
            "mode": (list(default_modes or DEFAULT_MODES) or ["hybrid"])[0],
        },
        "cases": cases,
    }
    return oracle, {"enabled": enabled_n, "skipped": skipped_n}


# ---------------------------------------------------------------------------
# Phase C — Score packs
# ---------------------------------------------------------------------------


def _ranked_texts_from_chunks(chunks: Sequence[dict[str, Any]]) -> list[str]:
    return [_searchable_blob(c) for c in chunks if _searchable_blob(c).strip()]


def score_scenario_retrieval(
    case: dict[str, Any],
    ranked_chunks: Sequence[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
    graph_context_present: bool = False,
) -> dict[str, Any]:
    """Retrieval metrics for one scenario case."""
    ranked_texts = _ranked_texts_from_chunks(ranked_chunks)
    expected_sections = [str(x) for x in case.get("expected_sections") or []]
    expected_contains = [str(x) for x in case.get("expected_chunk_contains") or []]
    forbidden = [str(x) for x in case.get("forbidden_chunk_contains") or []]
    expected_sources = case.get("expected_sources")
    recall = source_type_recall(
        expected_sources if isinstance(expected_sources, dict) else None,
        ranked_chunks,
        top_k,
    )
    by_type = {k: v for k, v in recall.items() if k != "overall"}
    trap_ok = trap_suppressed(
        expected_sections + expected_contains,
        forbidden,
        ranked_texts,
        top_k,
    )
    return {
        "case_id": str(case.get("id") or "unknown"),
        "mode": mode,
        "top_k": top_k,
        "section_hit_at_k": hit_fraction_at_k(expected_sections, ranked_texts, top_k),
        "provision_hit_at_k": hit_fraction_at_k(
            expected_contains, ranked_texts, top_k
        ),
        "trap_suppress": trap_ok,
        "source_type_recall": float(recall.get("overall", 1.0)),
        "source_type_recall_by_type": by_type,
        "graph_context_present": graph_context_present,
        "chunk_count": len(ranked_texts),
        "citation_ids": sorted(citation_set(ranked_chunks, top_k)),
    }


def _hit_rate(needles: Sequence[str], haystack: str) -> float:
    if not needles:
        return 1.0
    hits = sum(1 for n in needles if substring_present(n, haystack))
    return hits / len(needles)


def _hit_set(needles: Sequence[str], haystack: str) -> set[str]:
    return {n for n in needles if substring_present(n, haystack)}


def score_scenario_answer(
    case: dict[str, Any],
    answer: str,
    citation_blob: str,
    *,
    mode: str,
) -> dict[str, Any]:
    """Answer-path metrics: issues/benefits, citation markers, forbidden absent."""
    expected_issues = [str(x) for x in case.get("expected_issues") or []]
    expected_benefits = [str(x) for x in case.get("expected_benefits") or []]
    forbidden_answer = [str(x) for x in case.get("forbidden_answer_contains") or []]
    # Also treat forbidden_chunk_contains as soft answer forbids for non-trap packs
    # when forbidden_answer_contains is empty and case has expected issues/benefits.
    if not forbidden_answer and (expected_issues or expected_benefits):
        forbidden_answer = [str(x) for x in case.get("forbidden_chunk_contains") or []]

    answer_and_cites = f"{answer}\n{citation_blob}"
    issues_hit = _hit_set(expected_issues, answer_and_cites)
    benefits_hit = _hit_set(expected_benefits, answer_and_cites)
    issues_hit_rate = _hit_rate(expected_issues, answer_and_cites)
    benefits_hit_rate = _hit_rate(expected_benefits, answer_and_cites)

    # Citation should contain expected source markers when present.
    source_needles: list[str] = []
    expected_sources = case.get("expected_sources") or {}
    if isinstance(expected_sources, dict):
        for key in ("acts", "tds", "pgs", "prs"):
            source_needles.extend(str(x) for x in (expected_sources.get(key) or []))
    citation_source_hit_rate = _hit_rate(source_needles, citation_blob)

    forbidden_absent = not any(
        substring_present(n, answer_and_cites) for n in forbidden_answer
    )
    # Trap-style cases (forbidden present in chunk oracle, no issues/benefits):
    # "pass" means we do not require answer quality — only record presence.
    is_trap_style = not expected_issues and not expected_benefits and bool(
        case.get("forbidden_chunk_contains")
    )
    if is_trap_style:
        rules_like_pass = True
    else:
        rules_like_pass = (
            (issues_hit_rate >= 0.5 if expected_issues else True)
            and (benefits_hit_rate >= 0.5 if expected_benefits else True)
            and forbidden_absent
            and (citation_source_hit_rate >= 0.5 if source_needles else True)
        )

    return {
        "case_id": str(case.get("id") or "unknown"),
        "mode": mode,
        "issues_hit_rate": issues_hit_rate,
        "benefits_hit_rate": benefits_hit_rate,
        "issues_hit": sorted(issues_hit),
        "benefits_hit": sorted(benefits_hit),
        "citation_source_hit_rate": citation_source_hit_rate,
        "forbidden_absent": forbidden_absent,
        "rules_like_pass": rules_like_pass,
        "answer_chars": len(answer or ""),
        "is_trap_style": is_trap_style,
    }


def _references_summary(payload: dict[str, Any]) -> list[dict[str, Any]]:
    refs = payload.get("references") or []
    if not isinstance(refs, list):
        return []
    out: list[dict[str, Any]] = []
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        content = ref.get("content")
        if isinstance(content, list):
            snippet = " ".join(str(c) for c in content[:2] if c)[:200]
        elif isinstance(content, str):
            snippet = content[:200]
        else:
            snippet = ""
        out.append(
            {
                "file_path": ref.get("file_path"),
                "reference_id": ref.get("reference_id"),
                "content_snippet": snippet,
            }
        )
    return out


def score_scenario_packs(
    client: TaxFitnessClient,
    oracle: dict[str, Any],
    recon: dict[str, Any] | None,
    *,
    modes: Sequence[str],
    top_k: int,
    chunk_top_k: int,
    retrieval_only: bool,
    answers_only: bool,
    model_tags: dict[str, Any] | None,
    answers_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score enabled cases; optionally write per-case answer artifacts.

    Returns (pack_scores, answer_artifacts_meta).
    """
    cases = [c for c in (oracle.get("cases") or []) if isinstance(c, dict)]
    enabled = [c for c in cases if case_enabled(c)]
    pack_scores: list[dict[str, Any]] = []
    answer_metas: list[dict[str, Any]] = []

    # Index recon chunks by (case_id, mode) when available to avoid re-query.
    recon_index: dict[tuple[str, str], dict[str, Any]] = {}
    if recon:
        for query in recon.get("queries") or []:
            qid = str(query.get("id") or "")
            for mode_block in query.get("modes") or []:
                recon_index[(qid, str(mode_block.get("mode")))] = mode_block

    for case in enabled:
        case_id = str(case["id"])
        case_modes = [str(m) for m in (case.get("modes") or modes)]
        case_top_k = int(case.get("top_k") or top_k)
        for mode in case_modes:
            retrieval_score: dict[str, Any] | None = None
            if not answers_only:
                mode_block = recon_index.get((case_id, mode))
                if mode_block is not None:
                    # Rebuild chunk dicts compatible with source_type_recall.
                    ranked_chunks = []
                    for chunk in mode_block.get("chunks") or []:
                        ranked_chunks.append(
                            {
                                "content": chunk.get("content_snippet") or "",
                                "content_headings": chunk.get("content_headings"),
                                "file_path": chunk.get("file_path"),
                                "instrument": chunk.get("instrument"),
                            }
                        )
                    graph_present = bool(mode_block.get("graph_context_present"))
                else:
                    payload = client.query_data(
                        str(case["question"]),
                        mode=mode,
                        top_k=case_top_k,
                        chunk_top_k=chunk_top_k,
                    )
                    if payload.get("status") == "failure":
                        ranked_chunks = []
                        graph_present = False
                    else:
                        ranked_chunks = extract_ranked_chunks(payload)
                        graph_present = extract_graph_context_present(payload)
                retrieval_score = score_scenario_retrieval(
                    case,
                    ranked_chunks,
                    mode=mode,
                    top_k=case_top_k,
                    graph_context_present=graph_present,
                )

            answer_score: dict[str, Any] | None = None
            answer_text = ""
            citation_blob = ""
            refs_summary: list[dict[str, Any]] = []
            if not retrieval_only:
                ans_payload = client.query_answer(
                    str(case["question"]),
                    mode=mode,
                    top_k=case_top_k,
                )
                answer_text, citation_blob = extract_answer_and_citation_blob(
                    ans_payload
                )
                refs_summary = _references_summary(ans_payload)
                answer_score = score_scenario_answer(
                    case, answer_text, citation_blob, mode=mode
                )
                if answers_dir is not None:
                    artifact = {
                        "case_id": case_id,
                        "scenario_id": case.get("scenario_id"),
                        "question": case.get("question"),
                        "mode": mode,
                        "what_if": case.get("what_if"),
                        "baseline_case_id": case.get("baseline_case_id"),
                        "model_ab": model_tags or {},
                        "answer": answer_text,
                        "references": refs_summary,
                        "scores": answer_score,
                        "retrieval": retrieval_score,
                    }
                    out_path = answers_dir / f"{case_id}.json"
                    _write_json(out_path, artifact)
                    answer_metas.append(
                        {"case_id": case_id, "path": str(out_path), "mode": mode}
                    )

            pack_scores.append(
                {
                    "case_id": case_id,
                    "scenario_id": case.get("scenario_id"),
                    "intent": case.get("intent"),
                    "what_if": case.get("what_if"),
                    "baseline_case_id": case.get("baseline_case_id"),
                    "mode": mode,
                    "retrieval": retrieval_score,
                    "answer": answer_score,
                    "issues_hit": (
                        list((answer_score or {}).get("issues_hit") or [])
                    ),
                    "benefits_hit": (
                        list((answer_score or {}).get("benefits_hit") or [])
                    ),
                    "citation_ids": list(
                        (retrieval_score or {}).get("citation_ids") or []
                    ),
                }
            )
    return pack_scores, answer_metas


# ---------------------------------------------------------------------------
# Phase D — WHAT-IF diffs
# ---------------------------------------------------------------------------


def score_what_if_diffs(
    pack_scores: Sequence[dict[str, Any]],
    oracle: dict[str, Any],
) -> list[dict[str, Any]]:
    """For cases with baseline_case_id, compute citation/issue set deltas."""
    by_id: dict[str, dict[str, Any]] = {}
    for pack in pack_scores:
        # Prefer first mode entry per case_id for pairing.
        cid = str(pack.get("case_id") or "")
        if cid and cid not in by_id:
            by_id[cid] = pack

    cases_by_id = {
        str(c["id"]): c
        for c in (oracle.get("cases") or [])
        if isinstance(c, dict) and c.get("id")
    }

    diffs: list[dict[str, Any]] = []
    for pack in pack_scores:
        baseline_id = pack.get("baseline_case_id")
        if not baseline_id:
            continue
        baseline = by_id.get(str(baseline_id))
        if baseline is None:
            diffs.append(
                {
                    "case_id": pack.get("case_id"),
                    "baseline_case_id": baseline_id,
                    "status": "baseline_missing",
                    "changed": False,
                }
            )
            continue

        cite_a = set(baseline.get("citation_ids") or [])
        cite_b = set(pack.get("citation_ids") or [])
        issue_a = set(baseline.get("issues_hit") or [])
        issue_b = set(pack.get("issues_hit") or [])
        cite_delta = set_delta(cite_a, cite_b)
        issue_delta = set_delta(issue_a, issue_b)

        case = cases_by_id.get(str(pack.get("case_id")) or "", {})
        must_change = [
            str(x) for x in (case.get("variant_must_change_hints") or []) if str(x)
        ]
        # Soft: hints that appear in added/removed citation or issue strings.
        delta_blob = "\n".join(
            cite_delta["added"]
            + cite_delta["removed"]
            + issue_delta["added"]
            + issue_delta["removed"]
        )
        # Also check answer-side hit strings joined for soft matching.
        must_change_hits = [
            h for h in must_change if substring_present(h, delta_blob)
        ]
        # Broader soft check: variant hint appears in variant's issues/benefits
        # but not baseline's (or vice versa) — use pack answer scores if present.
        variant_answer = pack.get("answer") or {}
        baseline_answer = baseline.get("answer") or {}
        variant_side = "\n".join(
            list(variant_answer.get("issues_hit") or [])
            + list(variant_answer.get("benefits_hit") or [])
            + list(pack.get("citation_ids") or [])
        )
        baseline_side = "\n".join(
            list(baseline_answer.get("issues_hit") or [])
            + list(baseline_answer.get("benefits_hit") or [])
            + list(baseline.get("citation_ids") or [])
        )
        soft_hits: list[str] = []
        for hint in must_change:
            in_v = substring_present(hint, variant_side)
            in_b = substring_present(hint, baseline_side)
            if in_v != in_b or substring_present(hint, delta_blob):
                soft_hits.append(hint)
        must_change_hits = sorted(set(must_change_hits) | set(soft_hits))

        changed = bool(
            cite_delta["added"]
            or cite_delta["removed"]
            or issue_delta["added"]
            or issue_delta["removed"]
        )
        diffs.append(
            {
                "case_id": pack.get("case_id"),
                "baseline_case_id": baseline_id,
                "mode": pack.get("mode"),
                "status": "ok",
                "changed": changed,
                "citation_delta": cite_delta,
                "issue_delta": issue_delta,
                "variant_must_change_hints": must_change,
                "variant_must_change_hits": must_change_hits,
                "variant_must_change_hit_rate": (
                    (len(must_change_hits) / len(must_change)) if must_change else 1.0
                ),
            }
        )
    return diffs


# ---------------------------------------------------------------------------
# Phase E — Report
# ---------------------------------------------------------------------------


def aggregate_scenario_metrics(
    pack_scores: Sequence[dict[str, Any]],
    what_if_diffs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate retrieval + answer + WHAT-IF rates."""
    retrieval_rows = [
        p["retrieval"] for p in pack_scores if isinstance(p.get("retrieval"), dict)
    ]
    answer_rows = [
        p["answer"] for p in pack_scores if isinstance(p.get("answer"), dict)
    ]
    # Exclude trap-style from answer pass / issues / benefits means when flagged.
    answer_scored = [a for a in answer_rows if not a.get("is_trap_style")]

    def _mean(vals: Sequence[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    what_if_ok = [d for d in what_if_diffs if d.get("status") == "ok"]
    return {
        "packs": len(pack_scores),
        "mean_source_type_recall": _mean(
            [float(r.get("source_type_recall") or 0.0) for r in retrieval_rows]
        ),
        "mean_section_hit_at_k": _mean(
            [float(r.get("section_hit_at_k") or 0.0) for r in retrieval_rows]
        ),
        "mean_provision_hit_at_k": _mean(
            [float(r.get("provision_hit_at_k") or 0.0) for r in retrieval_rows]
        ),
        "trap_suppress_rate": _mean(
            [1.0 if r.get("trap_suppress") else 0.0 for r in retrieval_rows]
        ),
        "graph_context_rate": _mean(
            [1.0 if r.get("graph_context_present") else 0.0 for r in retrieval_rows]
        ),
        "issues_hit_rate": _mean(
            [float(a.get("issues_hit_rate") or 0.0) for a in answer_scored]
        ),
        "benefits_hit_rate": _mean(
            [float(a.get("benefits_hit_rate") or 0.0) for a in answer_scored]
        ),
        "citation_source_hit_rate": _mean(
            [float(a.get("citation_source_hit_rate") or 0.0) for a in answer_scored]
        ),
        "answer_pass_rate": _mean(
            [1.0 if a.get("rules_like_pass") else 0.0 for a in answer_scored]
        ),
        "answer_cases_scored": len(answer_scored),
        "what_if_pairs": len(what_if_ok),
        "what_if_changed_rate": _mean(
            [1.0 if d.get("changed") else 0.0 for d in what_if_ok]
        ),
        "what_if_must_change_hit_rate": _mean(
            [
                float(d.get("variant_must_change_hit_rate") or 0.0)
                for d in what_if_ok
            ]
        ),
    }


def build_scenario_report(
    *,
    api_url: str,
    label: str | None,
    modes: Sequence[str],
    top_k: int,
    chunk_top_k: int,
    oracle_counts: dict[str, int],
    aggregates: dict[str, Any],
    pack_scores: Sequence[dict[str, Any]],
    what_if_diffs: Sequence[dict[str, Any]],
    model_tags: dict[str, Any],
    answer_metas: Sequence[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "label": label,
        "api_url": api_url,
        "modes": list(modes),
        "top_k": top_k,
        "chunk_top_k": chunk_top_k,
        "oracle_counts": oracle_counts,
        "aggregates": aggregates,
        "model_ab": {
            "run_id": run_id,
            "label": label,
            "queried_at": datetime.now(timezone.utc).isoformat(),
            **model_tags,
        },
        "packs": list(pack_scores),
        "what_if_diffs": list(what_if_diffs),
        "answer_artifacts": list(answer_metas),
    }


def build_recommendation(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Compact recommendation emphasizing answer metrics for model A/B."""
    agg = report.get("aggregates") or {}
    model_ab = report.get("model_ab") or {}
    bullets: list[str] = []
    answer_pass = float(agg.get("answer_pass_rate") or 0.0)
    issues = float(agg.get("issues_hit_rate") or 0.0)
    benefits = float(agg.get("benefits_hit_rate") or 0.0)
    src_recall = float(agg.get("mean_source_type_recall") or 0.0)
    what_if = float(agg.get("what_if_changed_rate") or 0.0)

    bullets.append(
        f"Answer pass rate={answer_pass:.2f} "
        f"(issues_hit={issues:.2f}, benefits_hit={benefits:.2f}) — "
        "primary signal for QUERY model A/B."
    )
    bullets.append(
        f"Retrieval source_type_recall={src_recall:.2f}; "
        f"graph_context_rate={float(agg.get('graph_context_rate') or 0.0):.2f}."
    )
    if int(agg.get("what_if_pairs") or 0) > 0:
        bullets.append(
            f"WHAT-IF changed_rate={what_if:.2f} "
            f"(must_change_hit="
            f"{float(agg.get('what_if_must_change_hit_rate') or 0.0):.2f})."
        )
    else:
        bullets.append("No WHAT-IF pairs scored in this run.")

    return {
        "label": report.get("label"),
        "query_llm_model": model_ab.get("query_llm_model"),
        "modes": report.get("modes"),
        "top_k": report.get("top_k"),
        "headline_answer_metrics": {
            "answer_pass_rate": answer_pass,
            "issues_hit_rate": issues,
            "benefits_hit_rate": benefits,
            "citation_source_hit_rate": float(
                agg.get("citation_source_hit_rate") or 0.0
            ),
        },
        "headline_retrieval_metrics": {
            "mean_source_type_recall": src_recall,
            "mean_section_hit_at_k": float(agg.get("mean_section_hit_at_k") or 0.0),
            "mean_provision_hit_at_k": float(
                agg.get("mean_provision_hit_at_k") or 0.0
            ),
            "trap_suppress_rate": float(agg.get("trap_suppress_rate") or 0.0),
            "graph_context_rate": float(agg.get("graph_context_rate") or 0.0),
        },
        "what_if": {
            "pairs": int(agg.get("what_if_pairs") or 0),
            "changed_rate": what_if,
            "must_change_hit_rate": float(
                agg.get("what_if_must_change_hit_rate") or 0.0
            ),
        },
        "diagnosis": bullets,
        "compare_hint": (
            "Compare answers/ folders and recommendation.json across --label "
            "runs (e.g. sonnet vs oss). Prefer answer_pass_rate / issues_hit_rate "
            "/ benefits_hit_rate over retrieval-only deltas when judging QUERY models."
        ),
    }


def print_human_summary(recommendation: dict[str, Any], report: dict[str, Any]) -> None:
    """Stdout summary emphasizing answer metrics."""
    print("=== Scenario eval summary (answer-first) ===")
    model = recommendation.get("query_llm_model")
    label = recommendation.get("label")
    print(f"label={label!r}  query_llm_model={model}")
    ans = recommendation.get("headline_answer_metrics") or {}
    print(
        f"ANSWER  pass_rate={ans.get('answer_pass_rate', 0):.3f}  "
        f"issues_hit={ans.get('issues_hit_rate', 0):.3f}  "
        f"benefits_hit={ans.get('benefits_hit_rate', 0):.3f}  "
        f"cite_src={ans.get('citation_source_hit_rate', 0):.3f}"
    )
    ret = recommendation.get("headline_retrieval_metrics") or {}
    print(
        f"RETRIEVE source_type_recall={ret.get('mean_source_type_recall', 0):.3f}  "
        f"section={ret.get('mean_section_hit_at_k', 0):.3f}  "
        f"provision={ret.get('mean_provision_hit_at_k', 0):.3f}  "
        f"trap={ret.get('trap_suppress_rate', 0):.3f}  "
        f"graph={ret.get('graph_context_rate', 0):.3f}"
    )
    wi = recommendation.get("what_if") or {}
    print(
        f"WHAT-IF pairs={wi.get('pairs', 0)}  "
        f"changed_rate={wi.get('changed_rate', 0):.3f}  "
        f"must_change_hit={wi.get('must_change_hit_rate', 0):.3f}"
    )
    for bullet in recommendation.get("diagnosis") or []:
        print(f"  - {bullet}")
    print(recommendation.get("compare_hint") or "")
    packs = report.get("oracle_counts") or {}
    print(
        f"oracle enabled={packs.get('enabled', 0)} "
        f"skipped={packs.get('skipped', 0)}  "
        f"answer artifacts={len(report.get('answer_artifacts') or [])}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Part 2 live scenario + WHAT-IF eval: recon → draft oracle → "
            "score packs (retrieval + answers) → WHAT-IF diffs → report."
        )
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LIGHTRAG_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument("--api-key", default=os.getenv("LIGHTRAG_API_KEY"))
    parser.add_argument(
        "--seeds",
        default=str(DEFAULT_SEEDS),
        help="Scenario seeds JSON (default: tax_scenario_seeds.json)",
    )
    parser.add_argument(
        "--modes",
        default="hybrid",
        help="Comma-separated modes (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--chunk-top-k", type=int, default=DEFAULT_CHUNK_TOP_K)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: live_runs/scenario_<label_>timestamp)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Run label for model A/B (e.g. sonnet, oss)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip /query answer path",
    )
    parser.add_argument(
        "--answers-only",
        action="store_true",
        help="Skip retrieval scoring (still drafts oracle from recon unless "
        "recon is skipped — recon always runs)",
    )
    parser.add_argument("--allow-ungated", action="store_true")
    parser.add_argument(
        "--skip-what-if",
        action="store_true",
        help="Skip WHAT-IF diff phase",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when answer_pass_rate is below --min-answer-pass-rate",
    )
    parser.add_argument(
        "--min-answer-pass-rate",
        type=float,
        default=_env_float("TAX_EVAL_MIN_ANSWER_PASS_RATE", 0.5),
    )
    parser.add_argument(
        "--min-anchor-hits",
        type=int,
        default=_MIN_ANCHOR_HITS_TO_ENABLE,
        help="Min anchor/section hits in recon to enable a drafted case",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.allow_ungated and not live_eval_enabled():
        print(
            "Tax scenario eval is gated. Set TAX_EVAL_LIVE=true "
            "(or LIGHTRAG_RUN_INTEGRATION=true), or pass --allow-ungated.",
            file=sys.stderr,
        )
        return 2

    if args.retrieval_only and args.answers_only:
        print(
            "Choose at most one of --retrieval-only / --answers-only.",
            file=sys.stderr,
        )
        return 2

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        print("At least one --modes value is required.", file=sys.stderr)
        return 2

    seeds_path = Path(args.seeds).expanduser()
    try:
        seeds = load_scenario_seeds(seeds_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Failed to load seeds: {exc}", file=sys.stderr)
        return 1

    out_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else default_output_dir(args.label)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    answers_dir = out_dir / "answers"
    if not args.retrieval_only:
        answers_dir.mkdir(parents=True, exist_ok=True)

    client = TaxFitnessClient(api_url=args.api_url, api_key=args.api_key)
    try:
        health = client.health_check()
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    model_tags = extract_model_tags(health if isinstance(health, dict) else {})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print("Server LLM tags (from /health):")
    print(f"  query_llm_model:   {model_tags.get('query_llm_model')}")
    print(f"  keyword_llm_model: {model_tags.get('keyword_llm_model')}")
    print(f"  llm_model (base):  {model_tags.get('llm_model')}")
    if args.label:
        print(f"  label:             {args.label}")

    try:
        # Phase A
        print(f"Phase A — Recon ({len(seeds)} seeds × {modes})…")
        recon = run_recon(
            client,
            seeds,
            modes=modes,
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
        )
        _write_json(out_dir / "recon.json", recon)

        # Phase B
        print("Phase B — Draft scenario oracle…")
        oracle, counts = draft_scenario_oracle(
            seeds,
            recon,
            default_modes=modes,
            default_top_k=args.top_k,
            min_anchor_hits=args.min_anchor_hits,
        )
        _write_json(out_dir / "scenario_oracle.generated.json", oracle)
        print(f"  enabled={counts['enabled']} skipped={counts['skipped']}")

        # Phase C
        print("Phase C — Score packs…")
        pack_scores, answer_metas = score_scenario_packs(
            client,
            oracle,
            recon,
            modes=modes,
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
            retrieval_only=bool(args.retrieval_only),
            answers_only=bool(args.answers_only),
            model_tags=model_tags,
            answers_dir=None if args.retrieval_only else answers_dir,
        )

        # Phase D
        if args.skip_what_if:
            what_if_diffs: list[dict[str, Any]] = []
            print("Phase D — WHAT-IF diffs skipped.")
        else:
            print("Phase D — WHAT-IF diffs…")
            what_if_diffs = score_what_if_diffs(pack_scores, oracle)

        # Phase E
        print("Phase E — Report…")
        aggregates = aggregate_scenario_metrics(pack_scores, what_if_diffs)
        report = build_scenario_report(
            api_url=client.api_url,
            label=args.label,
            modes=modes,
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
            oracle_counts=counts,
            aggregates=aggregates,
            pack_scores=pack_scores,
            what_if_diffs=what_if_diffs,
            model_tags=model_tags,
            answer_metas=answer_metas,
            run_id=run_id,
        )
        recommendation = build_recommendation(report)
        _write_json(out_dir / "scenario_report.json", report)
        _write_json(out_dir / "recommendation.json", recommendation)
        print_human_summary(recommendation, report)
        print(f"Wrote outputs under: {out_dir}")

    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Tax scenario eval failed: {exc}", file=sys.stderr)
        return 1

    if args.strict and not args.retrieval_only:
        pass_rate = float(
            (recommendation.get("headline_answer_metrics") or {}).get(
                "answer_pass_rate"
            )
            or 0.0
        )
        if pass_rate < args.min_answer_pass_rate:
            print(
                f"Strict gate: answer_pass_rate {pass_rate:.3f} "
                f"< {args.min_answer_pass_rate}",
                file=sys.stderr,
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
