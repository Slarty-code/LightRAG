"""Offline tests for tax fixtures and the chunk integrity auditor."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pytest

from lightrag.evaluation.chunk_integrity_audit import (
    DEFAULT_CHUNK_TOKEN_SIZE,
    TRAP_NEEDLE,
    assess_trap_isolation,
    audit_chunk_integrity,
    count_boundary_breaks,
    load_fixture_act,
    make_audit_tokenizer,
)
from lightrag.evaluation.tax_fixtures import (
    ACT_BLOCKS_JSONL,
    ACT_MARKDOWN,
    ORACLE_JSON,
    RULES_DATASET_JSON,
    TD_MARKDOWN,
    fixture_path,
)


@pytest.mark.offline
class TaxFixturesTests(unittest.TestCase):
    def test_fixture_paths_exist(self):
        for name in (
            ACT_MARKDOWN,
            ACT_BLOCKS_JSONL,
            TD_MARKDOWN,
            ORACLE_JSON,
            RULES_DATASET_JSON,
        ):
            path = fixture_path(name)
            self.assertTrue(path.is_file(), msg=f"missing fixture: {path}")

    def test_act_has_hierarchy_and_keyword_trap(self):
        text = fixture_path(ACT_MARKDOWN).read_text(encoding="utf-8")
        for needle in (
            "Part 2 — Deductions",
            "Division 8 — General deductions",
            "s 8-1",
            "s 8-5",
            "to the extent",
            "Part 4 — Penalties",
            TRAP_NEEDLE,
        ):
            self.assertIn(needle, text)

        # Trap lives in Part 4, not in the Deduction Division body as an operative rule.
        part4 = text.split("## Part 4 — Penalties", 1)[1]
        self.assertIn(TRAP_NEEDLE, part4)
        division = text.split("### Division 8 — General deductions", 1)[1].split(
            "## Part 3", 1
        )[0]
        self.assertNotIn(TRAP_NEEDLE, division)

    def test_blocks_sidecar_matches_merged_hierarchy(self):
        content, blocks_path, headings = load_fixture_act()
        self.assertIsNotNone(blocks_path)
        self.assertTrue(content)
        for heading in (
            "Part 2 — Deductions",
            "Division 8 — General deductions",
            "s 8-1",
            "s 8-5",
            "Part 4 — Penalties and offences",
        ):
            self.assertIn(heading, headings)
        self.assertIn(TRAP_NEEDLE, content)
        self.assertIn("s 8-1", content)

    def test_oracle_schema(self):
        payload = json.loads(fixture_path(ORACLE_JSON).read_text(encoding="utf-8"))
        oracle = payload["oracle"]
        self.assertGreaterEqual(len(oracle), 1)
        for entry in oracle:
            self.assertTrue(entry["question"])
            self.assertIsInstance(entry["expected_sections"], list)
            self.assertTrue(entry["expected_sections"])
            self.assertIsInstance(entry["expected_chunk_contains"], list)
            self.assertIsInstance(entry["forbidden_chunk_contains"], list)
            forbidden = " ".join(entry["forbidden_chunk_contains"]).lower()
            self.assertIn("penalty for false", forbidden)
            self.assertIn(entry["mode"], {"mix", "hybrid", "local", "global", "naive"})

    def test_rules_dataset_minimal(self):
        payload = json.loads(
            fixture_path(RULES_DATASET_JSON).read_text(encoding="utf-8")
        )
        rules = payload["rules"]
        self.assertGreaterEqual(len(rules), 1)
        self.assertIn("expected_cite", rules[0])
        self.assertTrue(rules[0]["rule_skeleton"])


@pytest.mark.offline
class ChunkIntegrityAuditTests(unittest.TestCase):
    def test_f_weaker_than_p_on_boundary_breaks(self):
        report = audit_chunk_integrity(
            chunk_token_size=DEFAULT_CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=0,
            tokenizer=make_audit_tokenizer("char"),
            include_r=False,
        )
        comparison = report["comparison"]
        self.assertGreater(
            comparison["f_boundary_breaks"],
            comparison["p_boundary_breaks"],
            msg=report["strategies"],
        )
        self.assertTrue(comparison["f_weaker_on_boundary"])

    def test_p_meets_heading_and_trap_thresholds(self):
        report = audit_chunk_integrity(
            chunk_token_size=DEFAULT_CHUNK_TOKEN_SIZE,
            tokenizer=make_audit_tokenizer("char"),
            include_r=False,
        )
        p = report["strategies"]["P"]
        self.assertEqual(p["heading_coverage"], 1.0)
        self.assertTrue(p["trap_isolated"])
        self.assertFalse(p["trap_glued_into_deduction"])
        self.assertEqual(p["boundary_breaks"], 0)

    def test_td_cross_references_s_8_1(self):
        td = fixture_path(TD_MARKDOWN).read_text(encoding="utf-8")
        self.assertIn("s 8-1", td)
        self.assertIn("to the extent", td)

    def test_trap_helper_detects_glued_chunk(self):
        isolated, glued = assess_trap_isolation(
            [
                {
                    "content": (
                        "#### s 8-1\nGeneral deductions.\n"
                        f"Also {TRAP_NEEDLE} or misleading statements."
                    )
                }
            ]
        )
        self.assertFalse(isolated)
        self.assertTrue(glued)

    def test_boundary_helper_counts_edge_split(self):
        source = "Keep the marker ``s 8-5`` intact for auditors."
        chunks = [
            {"content": "Keep the marker ``s 8"},
            {"content": "-5`` intact for auditors."},
        ]
        breaks, total = count_boundary_breaks(chunks, source)
        self.assertEqual(total, 1)
        self.assertEqual(breaks, 1)

    def test_missing_sidecar_still_returns_report(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            act = root / "act.md"
            act.write_text(
                fixture_path(ACT_MARKDOWN).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            missing_blocks = root / "missing.blocks.jsonl"
            report = audit_chunk_integrity(
                act_path=act,
                blocks_path=missing_blocks,
                chunk_token_size=DEFAULT_CHUNK_TOKEN_SIZE,
                tokenizer=make_audit_tokenizer("char"),
                include_r=False,
            )
            self.assertIsNone(report["comparison"]["blocks_sidecar"])
            # Without sidecar, content falls back to markdown; P notes or
            # falls back internally to R. Report must still be structured.
            self.assertIn("P", report["strategies"])
            self.assertIn("F", report["strategies"])


if __name__ == "__main__":
    unittest.main()
