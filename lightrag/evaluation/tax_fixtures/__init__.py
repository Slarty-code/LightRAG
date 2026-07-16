"""Synthetic tax fixtures for retrieval-fitness and chunk-integrity evaluation.

Fixtures are loadable via :data:`FIXTURES_DIR` / :func:`fixture_path` without
importing document text into Python packages.
"""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent

ACT_MARKDOWN = "mini_ita_deductions.md"
ACT_BLOCKS_JSONL = "mini_ita_deductions.blocks.jsonl"
TD_MARKDOWN = "td_general_deductions.md"
PG_MARKDOWN = "pg_general_deductions.md"
PR_MARKDOWN = "pr_general_deductions.md"
# Chunk-integrity / offline schema from the harness brief (``oracle`` key).
ORACLE_JSON = "oracle.json"
RULES_DATASET_JSON = "rules_dataset.json"
# HTTP fitness runner defaults (``cases`` key) — same Act-grounded content.
TAX_RETRIEVAL_ORACLE_JSON = "tax_retrieval_oracle.json"
TAX_RULES_DATASET_JSON = "tax_rules_dataset.json"
# Reserved for Part 2 scenario / WHAT-IF packs (not evaluated in v1).
TAX_SCENARIO_ORACLE_JSON = "tax_scenario_oracle.json"
# Copy-paste template for curating live ITAA/ATO oracles (all cases enabled=false).
LIVE_ORACLE_TEMPLATE_JSON = "live_oracle_template.json"
# Seed questions for tax_live_loop recon → auto-draft → sweep.
TAX_LIVE_SEED_QUESTIONS_JSON = "tax_live_seed_questions.json"


def fixture_path(name: str) -> Path:
    """Return an absolute path under the fixtures directory."""
    return FIXTURES_DIR / name


__all__ = [
    "ACT_BLOCKS_JSONL",
    "ACT_MARKDOWN",
    "FIXTURES_DIR",
    "LIVE_ORACLE_TEMPLATE_JSON",
    "ORACLE_JSON",
    "PG_MARKDOWN",
    "PR_MARKDOWN",
    "RULES_DATASET_JSON",
    "TAX_LIVE_SEED_QUESTIONS_JSON",
    "TAX_RETRIEVAL_ORACLE_JSON",
    "TAX_RULES_DATASET_JSON",
    "TAX_SCENARIO_ORACLE_JSON",
    "TD_MARKDOWN",
    "fixture_path",
]
