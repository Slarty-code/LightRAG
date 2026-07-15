#!/usr/bin/env python3
"""Chunk integrity auditor for synthetic tax Act fixtures.

Runs LightRAG file-chunkers (F / P / optional R) on the mini-Act text and
reports structural metrics that should favour heading-aware chunking (P):

* **Boundary break rate** — ``s N-N`` anchors split across adjacent chunks
* **Heading coverage** — fixture headings retained under P
* **Trap isolation** — penalty/trap sentence not glued into Deduction Division
  chunks under P

P requires the ``.blocks.jsonl`` sidecar shipped with the fixture. When the
sidecar is missing or unreadable, P falls back to R inside
:func:`lightrag.chunker.chunking_by_paragraph_semantic` (documented there);
this auditor still compares F against that fallback path and labels the
report accordingly.

The check is offline: no LightRAG instance, embeddings, or LLM calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lightrag.chunker import (
    chunking_by_fixed_token,
    chunking_by_paragraph_semantic,
    chunking_by_recursive_character,
)
from lightrag.evaluation.tax_fixtures import (
    ACT_BLOCKS_JSONL,
    ACT_MARKDOWN,
    FIXTURES_DIR,
    fixture_path,
)
from lightrag.utils import Tokenizer, TokenizerInterface

# Default audit window (tokens). With the shipped fixture + char tokenizer
# used in unit tests, F tends to straddle an in-body ``s 8-5`` marker while
# P keeps anchors intact and isolates the Part 4 trap sentence.
DEFAULT_CHUNK_TOKEN_SIZE = 250
DEFAULT_OVERLAP = 0

SECTION_ANCHOR_RE = re.compile(r"\bs \d+-\d+\b")
TRAP_NEEDLE = "penalty for false"
DEDUCTION_MARKERS = (
    "Division 8 — General deductions",
    "s 8-1",
    "s 8-5",
    "Part 2 — Deductions",
)


@dataclass
class StrategyResult:
    strategy: str
    chunks: list[dict[str, Any]]
    boundary_breaks: int
    anchors_total: int
    boundary_break_rate: float
    heading_coverage: float | None
    headings_total: int
    headings_covered: int
    trap_isolated: bool
    trap_glued_into_deduction: bool
    notes: list[str] = field(default_factory=list)


class _CharTokenizer(TokenizerInterface):
    """1:1 character tokenizer for deterministic offline audits/tests."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def make_audit_tokenizer(kind: str = "char") -> Tokenizer:
    """Build a tokenizer for the auditor.

    ``char`` (default) keeps fixture-relative token windows stable in unit
    tests. ``tiktoken`` mirrors production LightRAG defaults when available.
    """
    if kind == "char":
        return Tokenizer(model_name="char-audit", tokenizer=_CharTokenizer())
    if kind == "tiktoken":
        from lightrag.utils import TiktokenTokenizer

        return TiktokenTokenizer()
    raise ValueError(f"Unsupported tokenizer kind: {kind!r}")


def load_blocks_rows(blocks_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in blocks_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("type") == "content":
            rows.append(obj)
    return rows


def merged_content_from_blocks(rows: list[dict[str, Any]]) -> str:
    """Rebuild the merged document text the F/R/V path would receive."""
    parts = [
        str(row.get("content") or "")
        for row in rows
        if str(row.get("content") or "").strip()
    ]
    return "\n\n".join(parts)


def load_fixture_act(
    act_path: Path | None = None,
    blocks_path: Path | None = None,
) -> tuple[str, Path | None, list[str]]:
    """Load Act markdown / blocks sidecar and return ``(content, blocks, headings)``.

    Prefers the sidecar merge (canonical for P). Falls back to raw markdown
    when the sidecar is absent.
    """
    act = (act_path or fixture_path(ACT_MARKDOWN)).expanduser().resolve()
    blocks = (blocks_path or fixture_path(ACT_BLOCKS_JSONL)).expanduser().resolve()
    headings: list[str] = []

    if blocks.is_file():
        rows = load_blocks_rows(blocks)
        headings = [
            str(row.get("heading") or "").strip()
            for row in rows
            if str(row.get("heading") or "").strip()
        ]
        return merged_content_from_blocks(rows), blocks, headings

    if not act.is_file():
        raise FileNotFoundError(
            f"Neither blocks sidecar {blocks} nor Act markdown {act} found"
        )
    return act.read_text(encoding="utf-8"), None, headings


def _chunk_texts(chunks: list[dict[str, Any]]) -> list[str]:
    return [str(chunk.get("content") or "") for chunk in chunks]


def count_boundary_breaks(chunks: list[dict[str, Any]], source: str) -> tuple[int, int]:
    """Count ``s N-N`` anchors split across adjacent chunk edges.

    Returns ``(break_events, distinct_anchors_in_source)``.
    """
    anchors = sorted(set(SECTION_ANCHOR_RE.findall(source)))
    texts = _chunk_texts(chunks)
    if len(texts) < 2:
        return 0, len(anchors)

    breaks = 0
    for left, right in zip(texts, texts[1:]):
        if not left or not right:
            continue
        # Wide enough to hold an ``s N-N`` token plus markdown backticks.
        left_tail = left[-24:]
        right_head = right[:24]
        bridge = left_tail + right_head
        join_at = len(left_tail)
        for match in SECTION_ANCHOR_RE.finditer(bridge):
            if match.start() < join_at <= match.end():
                breaks += 1
    return breaks, len(anchors)


def heading_coverage(
    chunks: list[dict[str, Any]],
    headings: list[str],
) -> tuple[float | None, int, int]:
    """Fraction of fixture headings present in chunk content or P heading meta."""
    if not headings:
        return None, 0, 0

    covered = 0
    for heading in headings:
        found = False
        for chunk in chunks:
            content = str(chunk.get("content") or "")
            if heading and heading in content:
                found = True
                break
            meta = chunk.get("heading")
            if isinstance(meta, dict) and meta.get("heading") == heading:
                found = True
                break
        if found:
            covered += 1
    return covered / len(headings), covered, len(headings)


def assess_trap_isolation(chunks: list[dict[str, Any]]) -> tuple[bool, bool]:
    """Return ``(trap_isolated, trap_glued_into_deduction)``.

    A trap is glued when any chunk contains both the Part 4 penalty needle and
    a Deduction Division / section marker from Part 2.
    """
    glued = False
    saw_trap = False
    for chunk in chunks:
        text = str(chunk.get("content") or "")
        if TRAP_NEEDLE in text:
            saw_trap = True
            if any(marker in text for marker in DEDUCTION_MARKERS):
                glued = True
                break
    if not saw_trap:
        # Trap missing from all chunks is treated as non-isolated (content loss).
        return False, False
    return (not glued), glued


def run_strategy(
    strategy: str,
    tokenizer: Tokenizer,
    content: str,
    *,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
    blocks_path: Path | None,
    headings: list[str],
) -> StrategyResult:
    notes: list[str] = []
    if strategy == "F":
        chunks = chunking_by_fixed_token(
            tokenizer,
            content,
            chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )
    elif strategy == "R":
        chunks = chunking_by_recursive_character(
            tokenizer,
            content,
            chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )
    elif strategy == "P":
        path_arg = str(blocks_path) if blocks_path is not None else None
        if path_arg is None:
            notes.append(
                "blocks_path missing; P will fall back to recursive-character (R)"
            )
        chunks = chunking_by_paragraph_semantic(
            tokenizer,
            content,
            chunk_token_size,
            blocks_path=path_arg,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )
        if path_arg is not None and chunks and not any(
            isinstance(chunk.get("heading"), dict) for chunk in chunks
        ):
            notes.append(
                "P chunks lack heading metadata (likely R fallback from unreadable sidecar)"
            )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    breaks, anchors_total = count_boundary_breaks(chunks, content)
    rate = (breaks / anchors_total) if anchors_total else 0.0
    coverage, covered, total = heading_coverage(chunks, headings)
    isolated, glued = assess_trap_isolation(chunks)

    # Heading coverage is most meaningful for P; still computed for F/R as a
    # secondary signal (content may retain heading lines by chance).
    return StrategyResult(
        strategy=strategy,
        chunks=chunks,
        boundary_breaks=breaks,
        anchors_total=anchors_total,
        boundary_break_rate=rate,
        heading_coverage=coverage,
        headings_total=total,
        headings_covered=covered,
        trap_isolated=isolated,
        trap_glued_into_deduction=glued,
        notes=notes,
    )


def audit_chunk_integrity(
    *,
    act_path: Path | None = None,
    blocks_path: Path | None = None,
    chunk_token_size: int = DEFAULT_CHUNK_TOKEN_SIZE,
    chunk_overlap_token_size: int = DEFAULT_OVERLAP,
    tokenizer: Tokenizer | None = None,
    include_r: bool = True,
) -> dict[str, Any]:
    """Run F / P (/ R) on the tax Act fixture and return a printable report dict."""
    if chunk_token_size <= 0:
        raise ValueError("chunk_token_size must be positive")

    content, resolved_blocks, headings = load_fixture_act(act_path, blocks_path)
    if not content.strip():
        raise ValueError("Fixture Act content is empty")

    tok = tokenizer or make_audit_tokenizer("char")
    strategies = ["F", "P"]
    if include_r:
        strategies.append("R")

    results: dict[str, StrategyResult] = {}
    for name in strategies:
        try:
            results[name] = run_strategy(
                name,
                tok,
                content,
                chunk_token_size=chunk_token_size,
                chunk_overlap_token_size=chunk_overlap_token_size,
                blocks_path=resolved_blocks,
                headings=headings,
            )
        except ImportError as exc:
            # R (and P's missing-sidecar fallback) need langchain-text-splitters.
            results[name] = StrategyResult(
                strategy=name,
                chunks=[],
                boundary_breaks=0,
                anchors_total=len(set(SECTION_ANCHOR_RE.findall(content))),
                boundary_break_rate=0.0,
                heading_coverage=None,
                headings_total=len(headings),
                headings_covered=0,
                trap_isolated=False,
                trap_glued_into_deduction=False,
                notes=[f"skipped: {exc}"],
            )

    f_res = results["F"]
    p_res = results["P"]
    comparison = {
        "f_boundary_breaks": f_res.boundary_breaks,
        "p_boundary_breaks": p_res.boundary_breaks,
        "f_weaker_on_boundary": f_res.boundary_breaks > p_res.boundary_breaks,
        "p_trap_isolated": p_res.trap_isolated,
        "p_heading_coverage": p_res.heading_coverage,
        "blocks_sidecar": str(resolved_blocks) if resolved_blocks else None,
        "fixtures_dir": str(FIXTURES_DIR),
    }

    def _strategy_dict(res: StrategyResult) -> dict[str, Any]:
        return {
            "strategy": res.strategy,
            "chunk_count": len(res.chunks),
            "boundary_breaks": res.boundary_breaks,
            "anchors_total": res.anchors_total,
            "boundary_break_rate": res.boundary_break_rate,
            "heading_coverage": res.heading_coverage,
            "headings_covered": res.headings_covered,
            "headings_total": res.headings_total,
            "trap_isolated": res.trap_isolated,
            "trap_glued_into_deduction": res.trap_glued_into_deduction,
            "notes": list(res.notes),
        }

    return {
        "chunk_token_size": chunk_token_size,
        "chunk_overlap_token_size": chunk_overlap_token_size,
        "content_chars": len(content),
        "strategies": {name: _strategy_dict(res) for name, res in results.items()},
        "comparison": comparison,
        "_results": results,  # in-process use (tests); stripped from CLI JSON
    }


def print_report(report: dict[str, Any]) -> None:
    print("LightRAG tax chunk integrity audit")
    print(f"Chunk token size: {report['chunk_token_size']}")
    print(f"Overlap: {report['chunk_overlap_token_size']}")
    print(f"Content chars: {report['content_chars']}")
    print(f"Blocks sidecar: {report['comparison']['blocks_sidecar']}")
    print()
    for name, payload in report["strategies"].items():
        print(f"[{name}] chunks={payload['chunk_count']}")
        print(
            f"  boundary_breaks={payload['boundary_breaks']}/"
            f"{payload['anchors_total']} "
            f"(rate={payload['boundary_break_rate']:.3f})"
        )
        cov = payload["heading_coverage"]
        cov_s = "n/a" if cov is None else f"{cov:.3f}"
        print(
            f"  heading_coverage={cov_s} "
            f"({payload['headings_covered']}/{payload['headings_total']})"
        )
        print(
            f"  trap_isolated={payload['trap_isolated']} "
            f"glued={payload['trap_glued_into_deduction']}"
        )
        for note in payload["notes"]:
            print(f"  note: {note}")
    print()
    comparison = report["comparison"]
    print(
        "F weaker on boundary than P: "
        f"{comparison['f_weaker_on_boundary']} "
        f"(F={comparison['f_boundary_breaks']}, P={comparison['p_boundary_breaks']})"
    )
    print(f"P trap isolated: {comparison['p_trap_isolated']}")
    print(f"P heading coverage: {comparison['p_heading_coverage']}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit chunk integrity on LightRAG tax evaluation fixtures."
    )
    parser.add_argument(
        "--act",
        default=str(fixture_path(ACT_MARKDOWN)),
        help="Path to the synthetic Act markdown (fallback when sidecar missing).",
    )
    parser.add_argument(
        "--blocks",
        default=str(fixture_path(ACT_BLOCKS_JSONL)),
        help="Path to the Act .blocks.jsonl sidecar for the P strategy.",
    )
    parser.add_argument(
        "--chunk-token-size",
        type=int,
        default=DEFAULT_CHUNK_TOKEN_SIZE,
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="chunk_overlap_token_size passed to each chunker",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("char", "tiktoken"),
        default="char",
        help="char keeps fixture windows deterministic; tiktoken mirrors production",
    )
    parser.add_argument(
        "--skip-r",
        action="store_true",
        help="Do not run the optional R strategy comparison.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report dict as JSON instead of the text summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 unless F is weaker on boundary breaks and P isolates the trap "
        "with full heading coverage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        report = audit_chunk_integrity(
            act_path=Path(args.act),
            blocks_path=Path(args.blocks),
            chunk_token_size=args.chunk_token_size,
            chunk_overlap_token_size=args.overlap,
            tokenizer=make_audit_tokenizer(args.tokenizer),
            include_r=not args.skip_r,
        )
    except (OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        print(f"Chunk integrity audit failed: {exc}", file=sys.stderr)
        return 2

    public = {key: value for key, value in report.items() if not key.startswith("_")}
    if args.json:
        print(json.dumps(public, indent=2, ensure_ascii=False))
    else:
        print_report(public)

    if args.strict:
        comparison = report["comparison"]
        p = report["strategies"]["P"]
        ok = (
            comparison["f_weaker_on_boundary"]
            and comparison["p_trap_isolated"]
            and (p["heading_coverage"] or 0.0) >= 1.0
        )
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
