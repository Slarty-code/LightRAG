#!/usr/bin/env python3
"""Tag tax fitness runs with the server QUERY (and related) LLM model.

Reads ``GET /health`` → ``configuration.role_llm_config.QUERY.model`` (fallback
``configuration.llm_model``), runs the existing tax fitness harness, and writes
a report whose filename and payload include the model fingerprint so cheap vs
ZDR A/B runs do not get mixed up.

Usage:
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_model_ab \\
      --api-url http://localhost:9621 \\
      --label cheap \\
      --output-dir ./live_runs/model_ab

    # After switching QUERY_LLM_MODEL and restarting the server:
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_model_ab \\
      --label zdr --output-dir ./live_runs/model_ab
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

from lightrag.evaluation.tax_retrieval_fitness import (
    DEFAULT_API_URL,
    DEFAULT_ORACLE,
    DEFAULT_RULES,
    ServerUnavailableError,
    TaxFitnessClient,
    build_report,
    gates_passed,
    live_eval_enabled,
    load_oracle,
    load_rules_dataset,
    print_stdout_report,
    run_retrieval_fitness,
    run_rules_fitness,
    summarize_retrieval,
    summarize_rules,
    _env_float,
)

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "tax_fixtures"


def _slug(value: str, *, max_len: int = 80) -> str:
    """Filesystem-safe slug for model ids like ``vendor/model-name:free``."""
    text = (value or "unknown").strip().lower()
    text = text.replace("/", "__").replace(":", "_")
    text = re.sub(r"[^a-z0-9._+-]+", "-", text)
    text = text.strip("-._") or "unknown"
    return text[:max_len]


def extract_model_tags(health: dict[str, Any]) -> dict[str, Any]:
    """Pull LLM identity fields from a ``/health`` JSON body.

    Prefer role-specific QUERY model; fall back to base ``llm_model``.
    Also records KEYWORD / EXTRACT / VLM when ``role_llm_config`` is present.
    """
    if not isinstance(health, dict):
        return {
            "query_llm_model": None,
            "llm_model": None,
            "keyword_llm_model": None,
            "llm_binding": None,
            "llm_binding_host": None,
            "role_models": {},
            "source": "invalid_health",
        }

    config = health.get("configuration")
    if not isinstance(config, dict):
        config = {}

    base_model = config.get("llm_model")
    binding = config.get("llm_binding")
    host = config.get("llm_binding_host")

    role_cfg = config.get("role_llm_config")
    role_models: dict[str, Any] = {}
    query_model = None
    keyword_model = None
    if isinstance(role_cfg, dict):
        for role_name, entry in role_cfg.items():
            if not isinstance(entry, dict):
                continue
            model = entry.get("model")
            role_models[str(role_name)] = {
                "model": model,
                "binding": entry.get("binding"),
                "host": entry.get("host"),
            }
            key = str(role_name).upper()
            if key == "QUERY" and model:
                query_model = model
            if key == "KEYWORD" and model:
                keyword_model = model

    effective_query = query_model or base_model
    source = "role_llm_config.QUERY" if query_model else "configuration.llm_model"

    return {
        "query_llm_model": effective_query,
        "llm_model": base_model,
        "keyword_llm_model": keyword_model or base_model,
        "llm_binding": binding,
        "llm_binding_host": host,
        "role_models": role_models,
        "source": source,
        "query_model_explicit": bool(query_model),
    }


def build_tagged_report(
    fitness_report: dict[str, Any],
    *,
    model_tags: dict[str, Any],
    label: str | None,
    run_id: str,
) -> dict[str, Any]:
    """Merge fitness report with model fingerprint metadata."""
    tagged = dict(fitness_report)
    tagged["model_ab"] = {
        "run_id": run_id,
        "label": label,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        **model_tags,
    }
    return tagged


def default_report_filename(model_tags: dict[str, Any], label: str | None) -> str:
    model_slug = _slug(str(model_tags.get("query_llm_model") or "unknown"))
    label_slug = _slug(label) if label else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parts = ["fitness"]
    if label_slug:
        parts.append(label_slug)
    parts.append(model_slug)
    parts.append(stamp)
    return "_".join(parts) + ".json"


def run_ab(
    client: TaxFitnessClient,
    *,
    oracle_path: Path,
    rules_path: Path,
    modes: Sequence[str],
    retrieval_only: bool,
    rules_only: bool,
    label: str | None,
    min_section_hit: float,
    min_provision_hit: float,
    min_trap_suppress: float,
    min_mrr: float,
    min_rules_pass_rate: float,
) -> dict[str, Any]:
    health = client.health_check()
    model_tags = extract_model_tags(health if isinstance(health, dict) else {})
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    retrieval_scores = []
    rules_scores = []
    if not rules_only:
        oracle = load_oracle(oracle_path)
        # Pin modes on defaults for this A/B run.
        defaults = dict(oracle.get("defaults") or {})
        defaults["modes"] = list(modes)
        oracle["defaults"] = defaults
        for case in oracle.get("cases") or []:
            if isinstance(case, dict):
                case["modes"] = list(modes)
        retrieval_scores = run_retrieval_fitness(client, oracle, modes=modes)

    if not retrieval_only:
        rules_dataset = load_rules_dataset(rules_path)
        # Align /query mode with the A/B retrieval mode (default first entry).
        rules_defaults = dict(rules_dataset.get("defaults") or {})
        rules_defaults["mode"] = str(modes[0] if modes else "hybrid")
        rules_dataset["defaults"] = rules_defaults
        for case in rules_dataset.get("cases") or []:
            if isinstance(case, dict):
                case["mode"] = rules_defaults["mode"]
        rules_scores = run_rules_fitness(client, rules_dataset)

    retrieval_summary = summarize_retrieval(retrieval_scores)
    rules_summary = summarize_rules(rules_scores)
    passed, failures = gates_passed(
        retrieval_summary,
        rules_summary,
        min_section_hit=min_section_hit,
        min_provision_hit=min_provision_hit,
        min_trap_suppress=min_trap_suppress,
        min_mrr=min_mrr,
        min_rules_pass_rate=min_rules_pass_rate,
        check_rules=not retrieval_only,
    )
    fitness_report = build_report(
        retrieval_scores,
        rules_scores if not retrieval_only else [],
        api_url=client.api_url,
        gate_failures=failures if not passed else [],
    )
    return build_tagged_report(
        fitness_report,
        model_tags=model_tags,
        label=label,
        run_id=run_id,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tax fitness and tag the JSON report with QUERY_LLM_MODEL "
            "from GET /health (role_llm_config)."
        )
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LIGHTRAG_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument("--api-key", default=os.getenv("LIGHTRAG_API_KEY"))
    parser.add_argument(
        "--oracle",
        default=str(DEFAULT_ORACLE),
        help="Oracle JSON (prefer a live_oracle.generated.json from tax_live_loop)",
    )
    parser.add_argument(
        "--rules",
        default=str(DEFAULT_RULES),
        help="Rules JSON (prefer live_rules.generated.json from tax_live_loop)",
    )
    parser.add_argument(
        "--modes",
        default="hybrid",
        help="Comma-separated modes (default: hybrid)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human run label stored in report + filename (e.g. cheap, zdr)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FIXTURES_DIR / "live_runs" / "model_ab"),
        help="Directory for tagged fitness JSON reports",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit report path (overrides --output-dir auto name)",
    )
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--allow-ungated", action="store_true")
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
    parser.add_argument(
        "--print-model-only",
        action="store_true",
        help="Only fetch /health and print model tags (no fitness run)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.allow_ungated and not live_eval_enabled():
        print(
            "Tax model A/B is gated. Set TAX_EVAL_LIVE=true "
            "(or LIGHTRAG_RUN_INTEGRATION=true), or pass --allow-ungated.",
            file=sys.stderr,
        )
        return 2

    if args.retrieval_only and args.rules_only:
        print("Choose at most one of --retrieval-only / --rules-only.", file=sys.stderr)
        return 2

    client = TaxFitnessClient(api_url=args.api_url, api_key=args.api_key)
    try:
        health = client.health_check()
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    model_tags = extract_model_tags(health if isinstance(health, dict) else {})
    print("Server LLM tags (from /health):")
    print(f"  query_llm_model:   {model_tags.get('query_llm_model')}")
    print(f"  keyword_llm_model: {model_tags.get('keyword_llm_model')}")
    print(f"  llm_model (base):  {model_tags.get('llm_model')}")
    print(f"  source:            {model_tags.get('source')}")
    if args.label:
        print(f"  label:             {args.label}")

    if args.print_model_only:
        print("--- JSON ---")
        print(json.dumps(model_tags, indent=2, ensure_ascii=False))
        return 0

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    try:
        report = run_ab(
            client,
            oracle_path=Path(args.oracle).expanduser(),
            rules_path=Path(args.rules).expanduser(),
            modes=modes,
            retrieval_only=bool(args.retrieval_only),
            rules_only=bool(args.rules_only),
            label=args.label,
            min_section_hit=args.min_section_hit,
            min_provision_hit=args.min_provision_hit,
            min_trap_suppress=args.min_trap_suppress,
            min_mrr=args.min_mrr,
            min_rules_pass_rate=args.min_rules_pass_rate,
        )
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Tax model A/B run failed: {exc}", file=sys.stderr)
        return 2

    print_stdout_report(report)
    model_ab = report.get("model_ab") or {}
    print(
        f"Tagged query model: {model_ab.get('query_llm_model')} "
        f"(label={model_ab.get('label')!r})"
    )

    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        out_dir = Path(args.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / default_report_filename(model_tags, args.label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote tagged report: {output_path}")
    return 0 if report.get("gates_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
