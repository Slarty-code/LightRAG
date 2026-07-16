"""Unit tests for tax live loop helpers (no live API)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.evaluation.tax_fixtures import (
    TAX_LIVE_SEED_QUESTIONS_JSON,
    fixture_path,
)
from lightrag.evaluation.tax_live_loop import (
    build_mode_sets,
    build_recommendation,
    collect_anchor_hits,
    collect_section_markers,
    diagnose_from_metrics,
    draft_live_oracle,
    draft_live_rules,
    draft_oracle_case_from_recon,
    load_seed_questions,
    rank_configurations,
    rank_key_from_summary,
    summarize_mode_response,
)


def _tiny_recon() -> dict:
    """Minimal recon: provision hits s 8-1; trap surfaces penalty-only text."""
    return {
        "api_url": "http://localhost:9621",
        "top_k": 5,
        "modes": ["mix", "naive"],
        "queries": [
            {
                "id": "seed_s8_1",
                "question": "When is a general deduction allowed under s 8-1?",
                "intent": "provision",
                "anchor_terms": ["s 8-1", "to the extent", "missing-anchor"],
                "forbidden_hints": ["penalty"],
                "pair_with": None,
                "modes": [
                    {
                        "mode": "mix",
                        "top_k": 5,
                        "status": "success",
                        "graph_context_present": True,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "itaa1997_deductions.md",
                                "content_headings": "Division 8 — General deductions",
                                "instrument": "act",
                                "content_snippet": (
                                    "s 8-1 You can deduct a loss to the extent "
                                    "it is incurred in gaining assessable income."
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "id": "seed_trap_deduction",
                "question": "deduction",
                "intent": "trap",
                "anchor_terms": [],
                "forbidden_hints": ["penalty", "false or misleading"],
                "pair_with": "seed_s8_1",
                "modes": [
                    {
                        "mode": "naive",
                        "top_k": 5,
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
                "id": "seed_crosswalk",
                "question": "Does a TD cover apportionment under s 8-1?",
                "intent": "crosswalk",
                "anchor_terms": ["apportionment", "TD 2024/1", "s 8-1"],
                "forbidden_hints": [],
                "pair_with": None,
                "modes": [
                    {
                        "mode": "mix",
                        "top_k": 5,
                        "status": "success",
                        "graph_context_present": True,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "td_general_deductions.md",
                                "content_headings": "TD 2024/1 (Synthetic)",
                                "instrument": "td",
                                "content_snippet": (
                                    "TD 2024/1 confirms s 8-1 apportionment via "
                                    "to the extent."
                                ),
                            },
                            {
                                "rank": 2,
                                "file_path": "itaa1997_deductions.md",
                                "content_headings": "s 8-1",
                                "instrument": "act",
                                "content_snippet": "s 8-1 general deduction limbs.",
                            },
                        ],
                    }
                ],
            },
            {
                "id": "seed_rules",
                "question": "Cite the limbs of s 8-1.",
                "intent": "rules",
                "anchor_terms": ["s 8-1", "assessable income"],
                "forbidden_hints": ["stamp duty"],
                "pair_with": None,
                "modes": [
                    {
                        "mode": "mix",
                        "top_k": 5,
                        "status": "success",
                        "graph_context_present": False,
                        "chunks": [
                            {
                                "rank": 1,
                                "file_path": "itaa1997_deductions.md",
                                "content_headings": "s 8-1",
                                "instrument": "act",
                                "content_snippet": (
                                    "s 8-1 positive limb: incurred in gaining "
                                    "assessable income."
                                ),
                            }
                        ],
                    }
                ],
            },
            {
                "id": "seed_empty",
                "question": "obscure unused provision xyzzy",
                "intent": "provision",
                "anchor_terms": ["xyzzy-never-indexed"],
                "forbidden_hints": [],
                "pair_with": None,
                "modes": [
                    {
                        "mode": "mix",
                        "top_k": 5,
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
            "id": "seed_s8_1",
            "question": "When is a general deduction allowed under s 8-1?",
            "intent": "provision",
            "anchor_terms": ["s 8-1", "to the extent", "missing-anchor"],
            "forbidden_hints": ["penalty"],
            "pair_with": None,
        },
        {
            "id": "seed_trap_deduction",
            "question": "deduction",
            "intent": "trap",
            "anchor_terms": [],
            "forbidden_hints": ["penalty", "false or misleading"],
            "pair_with": "seed_s8_1",
        },
        {
            "id": "seed_crosswalk",
            "question": "Does a TD cover apportionment under s 8-1?",
            "intent": "crosswalk",
            "anchor_terms": ["apportionment", "TD 2024/1", "s 8-1"],
            "forbidden_hints": [],
            "pair_with": None,
        },
        {
            "id": "seed_rules",
            "question": "Cite the limbs of s 8-1.",
            "intent": "rules",
            "anchor_terms": ["s 8-1", "assessable income"],
            "forbidden_hints": ["stamp duty"],
            "pair_with": None,
        },
        {
            "id": "seed_empty",
            "question": "obscure unused provision xyzzy",
            "intent": "provision",
            "anchor_terms": ["xyzzy-never-indexed"],
            "forbidden_hints": [],
            "pair_with": None,
        },
    ]


def test_seed_fixture_loads():
    path = fixture_path(TAX_LIVE_SEED_QUESTIONS_JSON)
    assert path.is_file()
    seeds = load_seed_questions(path)
    assert len(seeds) >= 5
    intents = {s["intent"] for s in seeds}
    assert "provision" in intents
    assert "trap" in intents
    assert "crosswalk" in intents
    assert "rules" in intents
    trap = next(s for s in seeds if s["intent"] == "trap" and s.get("pair_with"))
    assert trap["pair_with"]


def test_collect_anchor_hits_and_section_markers():
    blobs = [
        "Division 8 — General deductions\ns 8-1 to the extent incurred",
        "unrelated fluff",
    ]
    hits = collect_anchor_hits(
        ["s 8-1", "to the extent", "never-seen"],
        blobs,
    )
    assert hits == ["s 8-1", "to the extent"]

    markers = collect_section_markers(
        [
            {
                "content_headings": "Division 8 — General deductions",
                "content_snippet": "see s 8-1 for the positive limbs",
                "file_path": "act.md",
            }
        ]
    )
    joined = " ".join(markers).casefold()
    assert "division 8" in joined or "s 8-1" in joined or "8-1" in joined


def test_draft_oracle_from_tiny_recon_enables_and_skips():
    seeds = _tiny_seeds()
    recon = _tiny_recon()
    oracle, counts = draft_live_oracle(seeds, recon, default_modes=["mix", "naive"])

    assert counts["enabled"] >= 3
    assert counts["skipped"] >= 1  # rules + empty

    by_id = {c["id"]: c for c in oracle["cases"]}
    provision = by_id["live_seed_s8_1"]
    assert provision["enabled"] is True
    assert "s 8-1" in provision["expected_chunk_contains"]
    assert "to the extent" in provision["expected_chunk_contains"]
    assert "missing-anchor" not in provision["expected_chunk_contains"]
    assert provision["expected_sections"]
    assert provision.get("expected_sources", {}).get("acts")

    trap = by_id["live_seed_trap_deduction"]
    assert trap["enabled"] is True
    assert trap["pair_with"] == "seed_s8_1"
    assert "s 8-1" in trap["expected_chunk_contains"] or trap["expected_sections"]
    assert trap["forbidden_chunk_contains"]
    assert any(
        "penalty" in f.casefold() or "false" in f.casefold()
        for f in trap["forbidden_chunk_contains"]
    )

    crosswalk = by_id["live_seed_crosswalk"]
    assert crosswalk["enabled"] is True
    sources = crosswalk["expected_sources"]
    assert sources["tds"]
    assert sources["acts"]

    empty = by_id["live_seed_empty"]
    assert empty["enabled"] is False

    # Rules seeds are not oracle cases.
    assert "live_seed_rules" not in by_id


def test_draft_rules_from_tiny_recon():
    seeds = _tiny_seeds()
    recon = _tiny_recon()
    rules, counts = draft_live_rules(seeds, recon)
    assert counts["enabled"] == 1
    assert counts["skipped"] == 0
    case = rules["cases"][0]
    assert case["enabled"] is True
    assert "s 8-1" in case["expected_answer_contains"]
    assert case["expected_citation_contains"]


def test_trap_pairs_with_provision_case():
    seeds = {s["id"]: s for s in _tiny_seeds()}
    trap = seeds["seed_trap_deduction"]
    case, enabled, reason = draft_oracle_case_from_recon(
        trap,
        _tiny_recon(),
        seeds_by_id=seeds,
        default_modes=["mix"],
    )
    assert enabled is True
    assert reason == ""
    assert "s 8-1" in " ".join(case["expected_chunk_contains"]).casefold() or case[
        "expected_sections"
    ]


def test_rank_configurations_prefers_trap_then_mrr():
    configs = [
        {
            "config_id": "a",
            "retrieval_summary": {
                "trap_suppress_rate": 0.5,
                "mean_reciprocal_rank": 0.9,
                "average_provision_hit_at_k": 0.9,
                "average_section_hit_at_k": 0.9,
            },
        },
        {
            "config_id": "b",
            "retrieval_summary": {
                "trap_suppress_rate": 1.0,
                "mean_reciprocal_rank": 0.2,
                "average_provision_hit_at_k": 0.2,
                "average_section_hit_at_k": 0.1,
            },
        },
        {
            "config_id": "c",
            "retrieval_summary": {
                "trap_suppress_rate": 1.0,
                "mean_reciprocal_rank": 0.8,
                "average_provision_hit_at_k": 0.5,
                "average_section_hit_at_k": 0.2,
            },
        },
    ]
    ranked = rank_configurations(configs)
    assert [c["config_id"] for c in ranked] == ["c", "b", "a"]
    assert rank_key_from_summary(ranked[0]["retrieval_summary"])[0] == 1.0


def test_build_mode_sets_includes_matrix():
    sets = build_mode_sets(["mix", "hybrid", "mix"])
    labels = [label for label, _ in sets]
    assert labels == ["mix", "hybrid", "matrix:mix,hybrid"]


def test_summarize_mode_response_snippet():
    payload = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "A" * 500,
                    "content_headings": "s 8-1",
                    "file_path": "itaa1997_deductions.md",
                }
            ],
            "entities": [{"name": "s 8-1"}],
            "relationships": [],
        },
    }
    out = summarize_mode_response(payload, mode="mix", top_k=10)
    assert out["graph_context_present"] is True
    assert out["chunks"][0]["instrument"] == "act"
    assert len(out["chunks"][0]["content_snippet"]) <= 400


def test_diagnose_high_provision_low_section():
    best = {
        "gates_passed": False,
        "gate_failures": ["section_hit_at_k 0.200 < 0.5"],
        "retrieval_summary": {
            "average_section_hit_at_k": 0.2,
            "average_provision_hit_at_k": 0.85,
            "trap_suppress_rate": 1.0,
            "mean_reciprocal_rank": 0.8,
        },
    }
    recon = {
        "queries": [
            {
                "modes": [
                    {
                        "chunks": [
                            {"content_headings": None},
                            {"content_headings": None},
                            {"content_headings": None},
                            {"content_headings": None},
                            {"content_headings": "s 8-1"},
                        ]
                    }
                ]
            }
        ]
    }
    bullets = diagnose_from_metrics(best, recon, {"enabled": 3, "skipped": 1})
    assert any("section Hit@K low" in b for b in bullets)
    assert any("headings sparse" in b for b in bullets)


def test_build_recommendation_shape():
    ranked = rank_configurations(
        [
            {
                "config_id": "mix|top_k=10",
                "modes": ["mix"],
                "top_k": 10,
                "gates_passed": True,
                "gate_failures": [],
                "retrieval_summary": {
                    "trap_suppress_rate": 1.0,
                    "mean_reciprocal_rank": 0.7,
                    "average_provision_hit_at_k": 0.8,
                    "average_section_hit_at_k": 0.3,
                    "queries": 4,
                },
            }
        ]
    )
    rec = build_recommendation(
        api_url="http://localhost:9621",
        ranked=ranked,
        recon=_tiny_recon(),
        oracle_counts={"enabled": 3, "skipped": 2},
        rules_counts={"enabled": 1, "skipped": 0},
        rules_summary={"queries": 1, "pass_rate": 0.0},
    )
    assert rec["best_modes"] == ["mix"]
    assert rec["best_top_k"] == 10
    assert rec["metric_table"]
    assert rec["diagnosis"]
    assert "informational" in rec["note"].casefold()


def test_load_seed_questions_rejects_bad_intent(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "seeds": [
                    {
                        "id": "x",
                        "question": "q",
                        "intent": "scenario",
                        "anchor_terms": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intent"):
        load_seed_questions(bad)
