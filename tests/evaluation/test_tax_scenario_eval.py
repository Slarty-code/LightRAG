"""Unit tests for tax_scenario_eval helpers (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

from lightrag.evaluation.tax_fixtures import (
    TAX_SCENARIO_SEEDS_JSON,
    fixture_path,
)
from lightrag.evaluation.tax_model_ab import extract_model_tags
from lightrag.evaluation.tax_scenario_eval import (
    aggregate_scenario_metrics,
    build_recommendation,
    build_scenario_report,
    draft_scenario_case_from_recon,
    draft_scenario_oracle,
    load_scenario_seeds,
    score_scenario_answer,
    score_scenario_retrieval,
    score_what_if_diffs,
    summarize_mode_response,
)


def _tiny_recon() -> dict:
    """Minimal recon: home-office baseline + sole-trader WHAT-IF + trap."""
    return {
        "api_url": "http://localhost:9621",
        "top_k": 50,
        "chunk_top_k": 50,
        "modes": ["hybrid"],
        "queries": [
            {
                "id": "scen_home_office_employee",
                "scenario_id": "home_office_v1",
                "question": "Employee home office pack?",
                "intent": "scenario",
                "what_if": None,
                "baseline_case_id": None,
                "anchor_terms": ["s 8-1", "home office", "to the extent"],
                "forbidden_hints": ["penalty for false"],
                "modes": [
                    {
                        "mode": "hybrid",
                        "top_k": 50,
                        "status": "success",
                        "graph_context_present": True,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "itaa1997_deductions.md",
                                "content_headings": "s 8-1 General deductions",
                                "instrument": "act",
                                "content_snippet": (
                                    "s 8-1 You can deduct a loss to the extent "
                                    "it is incurred in gaining assessable income. "
                                    "Home office running expenses may be claimed "
                                    "subject to private or domestic apportionment."
                                ),
                            },
                            {
                                "rank": 2,
                                "file_path": "td_general_deductions.md",
                                "content_headings": "TD 2024/1",
                                "instrument": "td",
                                "content_snippet": (
                                    "TD confirms home office apportionment under "
                                    "to the extent."
                                ),
                            },
                            {
                                "rank": 3,
                                "file_path": "pg_general_deductions.md",
                                "content_headings": "PG home office",
                                "instrument": "pg",
                                "content_snippet": (
                                    "Practice Guide on home office deduction "
                                    "running expense benefits."
                                ),
                            },
                        ],
                    }
                ],
            },
            {
                "id": "scen_home_office_sole_trader",
                "scenario_id": "home_office_v1",
                "question": "WHAT-IF sole trader home office?",
                "intent": "what_if",
                "what_if": "sole trader",
                "baseline_case_id": "scen_home_office_employee",
                "anchor_terms": ["s 8-1", "sole trader", "business"],
                "forbidden_hints": ["penalty for false"],
                "modes": [
                    {
                        "mode": "hybrid",
                        "top_k": 50,
                        "status": "success",
                        "graph_context_present": True,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "itaa1997_business.md",
                                "content_headings": "s 8-1 business deductions",
                                "instrument": "act",
                                "content_snippet": (
                                    "s 8-1 for a sole trader carrying on a business: "
                                    "deduct business expense to the extent incurred "
                                    "in gaining assessable income."
                                ),
                            },
                            {
                                "rank": 2,
                                "file_path": "td_business.md",
                                "content_headings": "TD business premises",
                                "instrument": "td",
                                "content_snippet": (
                                    "TD on sole trader home office used as a "
                                    "business place of business."
                                ),
                            },
                        ],
                    }
                ],
            },
            {
                "id": "scen_keyword_trap",
                "scenario_id": "keyword_trap_v1",
                "question": "deduction",
                "intent": "scenario",
                "what_if": None,
                "baseline_case_id": None,
                "anchor_terms": [],
                "forbidden_hints": ["penalty", "false or misleading"],
                "modes": [
                    {
                        "mode": "hybrid",
                        "top_k": 50,
                        "status": "success",
                        "graph_context_present": False,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "penalties_part4.md",
                                "content_headings": "Part 4 — Penalties",
                                "instrument": "unknown",
                                "content_snippet": (
                                    "A penalty applies for a false or misleading "
                                    "statement about a deduction claim."
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "id": "scen_empty",
                "scenario_id": "empty_v1",
                "question": "obscure unused xyzzy scenario",
                "intent": "scenario",
                "what_if": None,
                "baseline_case_id": None,
                "anchor_terms": ["xyzzy-never-indexed"],
                "forbidden_hints": [],
                "modes": [
                    {
                        "mode": "hybrid",
                        "top_k": 50,
                        "status": "success",
                        "graph_context_present": False,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "unrelated.md",
                                "content_headings": None,
                                "instrument": "unknown",
                                "content_snippet": "Nothing tax related here.",
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _tiny_seeds() -> list[dict]:
    return [
        {
            "id": "scen_home_office_employee",
            "scenario_id": "home_office_v1",
            "client_facts": "Employee WFH 3 days/week.",
            "question": "Employee home office pack?",
            "what_if": None,
            "baseline_case_id": None,
            "intent": "scenario",
            "anchor_terms": ["s 8-1", "home office", "to the extent", "missing"],
            "expected_sources_hints": {
                "acts": ["8-1", "ITAA"],
                "tds": ["TD"],
                "pgs": ["PG"],
                "prs": [],
            },
            "expected_issues_hints": [
                "apportion",
                "private or domestic",
                "home office",
            ],
            "expected_benefits_hints": ["deduction", "running expense"],
            "forbidden_hints": ["penalty for false"],
            "variant_must_change_hints": [],
        },
        {
            "id": "scen_home_office_sole_trader",
            "scenario_id": "home_office_v1",
            "client_facts": "Sole trader variant.",
            "question": "WHAT-IF sole trader home office?",
            "what_if": "sole trader",
            "baseline_case_id": "scen_home_office_employee",
            "intent": "what_if",
            "anchor_terms": ["s 8-1", "sole trader", "business"],
            "expected_sources_hints": {
                "acts": ["8-1"],
                "tds": ["TD"],
                "pgs": [],
                "prs": [],
            },
            "expected_issues_hints": ["business", "sole trader"],
            "expected_benefits_hints": ["deduction", "business expense"],
            "forbidden_hints": ["penalty for false"],
            "variant_must_change_hints": ["sole trader", "business", "employee"],
        },
        {
            "id": "scen_keyword_trap",
            "scenario_id": "keyword_trap_v1",
            "client_facts": None,
            "question": "deduction",
            "what_if": None,
            "baseline_case_id": None,
            "intent": "scenario",
            "anchor_terms": [],
            "expected_sources_hints": {"acts": [], "tds": [], "pgs": [], "prs": []},
            "expected_issues_hints": [],
            "expected_benefits_hints": [],
            "forbidden_hints": ["penalty", "false or misleading"],
            "variant_must_change_hints": [],
        },
        {
            "id": "scen_empty",
            "scenario_id": "empty_v1",
            "client_facts": None,
            "question": "obscure unused xyzzy scenario",
            "what_if": None,
            "baseline_case_id": None,
            "intent": "scenario",
            "anchor_terms": ["xyzzy-never-indexed"],
            "expected_sources_hints": {"acts": [], "tds": [], "pgs": [], "prs": []},
            "expected_issues_hints": [],
            "expected_benefits_hints": [],
            "forbidden_hints": [],
            "variant_must_change_hints": [],
        },
    ]


def test_fixture_seeds_load_and_validate():
    path = fixture_path(TAX_SCENARIO_SEEDS_JSON)
    seeds = load_scenario_seeds(path)
    assert len(seeds) >= 8
    ids = {s["id"] for s in seeds}
    assert "scen_home_office_employee" in ids
    assert "scen_home_office_sole_trader" in ids
    sole = next(s for s in seeds if s["id"] == "scen_home_office_sole_trader")
    assert sole["intent"] == "what_if"
    assert sole["baseline_case_id"] == "scen_home_office_employee"
    assert sole["what_if"]


def test_summarize_mode_response_extracts_chunks():
    payload = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "s 8-1 to the extent",
                    "content_headings": "Division 8",
                    "file_path": "itaa.md",
                }
            ],
            "entities": [{"name": "s 8-1"}],
            "relationships": [],
        },
    }
    summary = summarize_mode_response(payload, mode="hybrid", top_k=50)
    assert summary["status"] == "success"
    assert summary["graph_context_present"] is True
    assert len(summary["chunks"]) == 1
    assert summary["chunks"][0]["instrument"] == "act"


def test_draft_oracle_enables_on_anchor_hits():
    recon = _tiny_recon()
    seeds = _tiny_seeds()
    case, enabled, reason = draft_scenario_case_from_recon(
        seeds[0], recon, default_modes=["hybrid"], default_top_k=50
    )
    assert enabled is True
    assert reason == ""
    assert case["scenario_id"] == "home_office_v1"
    assert case["what_if"] is None
    assert case["baseline_case_id"] is None
    assert "s 8-1" in case["expected_chunk_contains"]
    assert "home office" in case["expected_issues"] or "private or domestic" in case[
        "expected_issues"
    ]
    assert case["expected_sources"].get("acts")
    assert case["expected_sources"].get("tds")


def test_draft_oracle_skips_when_no_anchors():
    recon = _tiny_recon()
    seeds = _tiny_seeds()
    empty = next(s for s in seeds if s["id"] == "scen_empty")
    case, enabled, reason = draft_scenario_case_from_recon(empty, recon)
    assert enabled is False
    assert reason == "insufficient_anchors_in_recon"
    assert case["enabled"] is False


def test_draft_trap_seed_enabled_when_forbidden_in_recon():
    recon = _tiny_recon()
    seeds = _tiny_seeds()
    trap = next(s for s in seeds if s["id"] == "scen_keyword_trap")
    case, enabled, reason = draft_scenario_case_from_recon(trap, recon)
    assert enabled is True
    assert "penalty" in case["forbidden_chunk_contains"]
    assert reason == ""


def test_draft_scenario_oracle_preserves_what_if_fields():
    recon = _tiny_recon()
    seeds = _tiny_seeds()
    oracle, counts = draft_scenario_oracle(
        seeds, recon, default_modes=["hybrid"], default_top_k=50
    )
    assert counts["enabled"] >= 2
    by_id = {c["id"]: c for c in oracle["cases"]}
    variant = by_id["scen_home_office_sole_trader"]
    assert variant["intent"] == "what_if"
    assert variant["baseline_case_id"] == "scen_home_office_employee"
    assert variant["what_if"] == "sole trader"
    assert variant["scenario_id"] == "home_office_v1"


def test_score_scenario_retrieval_source_type_and_trap():
    case = {
        "id": "scen_home_office_employee",
        "expected_sections": ["s 8-1"],
        "expected_chunk_contains": ["home office", "to the extent"],
        "forbidden_chunk_contains": ["penalty for false"],
        "expected_sources": {"acts": ["8-1"], "tds": ["TD"], "pgs": ["PG"]},
    }
    chunks = [
        {
            "content": "s 8-1 home office to the extent",
            "content_headings": "s 8-1",
            "file_path": "itaa.md",
            "instrument": "act",
        },
        {
            "content": "TD on home office",
            "content_headings": "TD 2024/1",
            "file_path": "td.md",
            "instrument": "td",
        },
        {
            "content": "PG running expense",
            "content_headings": "PG",
            "file_path": "pg.md",
            "instrument": "pg",
        },
    ]
    score = score_scenario_retrieval(
        case, chunks, mode="hybrid", top_k=50, graph_context_present=True
    )
    assert score["source_type_recall"] > 0.5
    assert score["trap_suppress"] is True
    assert score["graph_context_present"] is True
    assert score["citation_ids"]


def test_score_scenario_answer_issues_benefits_and_forbidden():
    case = {
        "id": "scen_home_office_employee",
        "expected_issues": ["apportion", "private or domestic"],
        "expected_benefits": ["deduction", "to the extent"],
        "expected_sources": {"acts": ["8-1"], "tds": ["TD"]},
        "forbidden_answer_contains": ["penalty for false"],
        "forbidden_chunk_contains": [],
    }
    answer = (
        "Key issues include apportionment for private or domestic use. "
        "Benefit: a deduction to the extent the expense relates to assessable income."
    )
    cites = "file: itaa.md containing s 8-1\nfile: td.md TD 2024/1"
    score = score_scenario_answer(case, answer, cites, mode="hybrid")
    assert score["issues_hit_rate"] == 1.0
    assert score["benefits_hit_rate"] == 1.0
    assert score["forbidden_absent"] is True
    assert score["rules_like_pass"] is True
    assert score["citation_source_hit_rate"] == 1.0


def test_score_scenario_answer_fails_when_forbidden_present():
    case = {
        "id": "x",
        "expected_issues": ["apportion"],
        "expected_benefits": ["deduction"],
        "expected_sources": {},
        "forbidden_answer_contains": ["penalty for false"],
    }
    score = score_scenario_answer(
        case,
        "You get a deduction but penalty for false statements apply.",
        "cite",
        mode="hybrid",
    )
    assert score["forbidden_absent"] is False
    assert score["rules_like_pass"] is False


def test_what_if_set_delta_scoring():
    oracle = {
        "cases": [
            {
                "id": "scen_home_office_employee",
                "baseline_case_id": None,
                "variant_must_change_hints": [],
            },
            {
                "id": "scen_home_office_sole_trader",
                "baseline_case_id": "scen_home_office_employee",
                "variant_must_change_hints": ["sole trader", "business", "employee"],
            },
        ]
    }
    packs = [
        {
            "case_id": "scen_home_office_employee",
            "baseline_case_id": None,
            "mode": "hybrid",
            "citation_ids": ["itaa1997_deductions.md", "td_general_deductions.md"],
            "issues_hit": ["home office", "private or domestic"],
            "benefits_hit": ["deduction"],
            "answer": {
                "issues_hit": ["home office", "private or domestic"],
                "benefits_hit": ["deduction"],
            },
            "retrieval": {},
        },
        {
            "case_id": "scen_home_office_sole_trader",
            "baseline_case_id": "scen_home_office_employee",
            "mode": "hybrid",
            "citation_ids": ["itaa1997_business.md", "td_business.md"],
            "issues_hit": ["business", "sole trader"],
            "benefits_hit": ["deduction", "business expense"],
            "answer": {
                "issues_hit": ["business", "sole trader"],
                "benefits_hit": ["deduction", "business expense"],
            },
            "retrieval": {},
        },
    ]
    diffs = score_what_if_diffs(packs, oracle)
    assert len(diffs) == 1
    diff = diffs[0]
    assert diff["status"] == "ok"
    assert diff["changed"] is True
    assert "itaa1997_business.md" in diff["citation_delta"]["added"]
    assert "itaa1997_deductions.md" in diff["citation_delta"]["removed"]
    assert "sole trader" in diff["variant_must_change_hits"]
    assert diff["variant_must_change_hit_rate"] > 0.0


def test_model_tag_merge_into_report():
    health = {
        "configuration": {
            "llm_model": "vendor/oss-model",
            "role_llm_config": {
                "QUERY": {"model": "vendor/sonnet-model", "binding": "openai"},
            },
        }
    }
    tags = extract_model_tags(health)
    assert tags["query_llm_model"] == "vendor/sonnet-model"
    aggregates = {
        "packs": 2,
        "mean_source_type_recall": 0.8,
        "mean_section_hit_at_k": 0.7,
        "mean_provision_hit_at_k": 0.6,
        "trap_suppress_rate": 1.0,
        "graph_context_rate": 1.0,
        "issues_hit_rate": 0.9,
        "benefits_hit_rate": 0.85,
        "citation_source_hit_rate": 0.75,
        "answer_pass_rate": 1.0,
        "answer_cases_scored": 2,
        "what_if_pairs": 1,
        "what_if_changed_rate": 1.0,
        "what_if_must_change_hit_rate": 0.5,
    }
    report = build_scenario_report(
        api_url="http://localhost:9621",
        label="sonnet",
        modes=["hybrid"],
        top_k=50,
        chunk_top_k=50,
        oracle_counts={"enabled": 2, "skipped": 0},
        aggregates=aggregates,
        pack_scores=[],
        what_if_diffs=[],
        model_tags=tags,
        answer_metas=[],
        run_id="20260101T000000Z",
    )
    assert report["model_ab"]["label"] == "sonnet"
    assert report["model_ab"]["query_llm_model"] == "vendor/sonnet-model"
    assert report["model_ab"]["run_id"] == "20260101T000000Z"
    rec = build_recommendation(report)
    assert rec["query_llm_model"] == "vendor/sonnet-model"
    assert rec["headline_answer_metrics"]["answer_pass_rate"] == 1.0
    assert "answer" in (rec.get("compare_hint") or "").lower()


def test_aggregate_includes_what_if_changed_rate():
    packs = [
        {
            "retrieval": {
                "source_type_recall": 1.0,
                "section_hit_at_k": 1.0,
                "provision_hit_at_k": 1.0,
                "trap_suppress": True,
                "graph_context_present": True,
            },
            "answer": {
                "issues_hit_rate": 1.0,
                "benefits_hit_rate": 0.5,
                "citation_source_hit_rate": 1.0,
                "rules_like_pass": True,
                "is_trap_style": False,
            },
        }
    ]
    diffs = [
        {
            "status": "ok",
            "changed": True,
            "variant_must_change_hit_rate": 0.5,
        }
    ]
    agg = aggregate_scenario_metrics(packs, diffs)
    assert agg["what_if_changed_rate"] == 1.0
    assert agg["answer_pass_rate"] == 1.0
    assert agg["issues_hit_rate"] == 1.0


def test_seed_file_is_valid_json():
    raw = Path(fixture_path(TAX_SCENARIO_SEEDS_JSON)).read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert "seeds" in payload
    assert isinstance(payload["seeds"], list)
