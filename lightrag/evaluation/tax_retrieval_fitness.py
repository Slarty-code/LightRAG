#!/usr/bin/env python3
"""HTTP tax retrieval fitness runner for a local LightRAG API.

Scores retrieval via ``POST /query/data`` (contexts/chunks/content_headings)
and light guidance/rules answers via ``POST /query``.

Live runs are gated by ``TAX_EVAL_LIVE=true`` or ``LIGHTRAG_RUN_INTEGRATION=true``.

Usage:
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness
    TAX_EVAL_LIVE=true python -m lightrag.evaluation.tax_retrieval_fitness \\
        --api-url http://localhost:9621 --output /tmp/tax_fitness.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import httpx

EVAL_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = EVAL_DIR / "tax_fixtures"
DEFAULT_ORACLE = FIXTURES_DIR / "tax_retrieval_oracle.json"
DEFAULT_RULES = FIXTURES_DIR / "tax_rules_dataset.json"
DEFAULT_API_URL = "http://localhost:9621"
DEFAULT_MODES = ("naive", "local", "hybrid", "mix")

CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 180.0


class ServerUnavailableError(RuntimeError):
    """Raised when the LightRAG API health check fails."""


def live_eval_enabled() -> bool:
    """Return True when live HTTP tax eval is explicitly enabled."""
    for name in ("TAX_EVAL_LIVE", "LIGHTRAG_RUN_INTEGRATION"):
        if os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return float(raw)


def _normalize_api_url(url: str) -> str:
    return url.rstrip("/")


def _auth_headers(api_key: str | None = None) -> dict[str, str]:
    key = api_key if api_key is not None else os.getenv("LIGHTRAG_API_KEY")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-API-Key"] = key
    return headers


def substring_present(needle: str, haystack: str, *, case_sensitive: bool = False) -> bool:
    if not needle:
        return False
    if case_sensitive:
        return needle in haystack
    return needle.casefold() in haystack.casefold()


def first_match_rank(
    needles: Sequence[str],
    ranked_texts: Sequence[str],
    *,
    case_sensitive: bool = False,
) -> int | None:
    """1-based rank of the first chunk matching any needle, or None."""
    for rank, text in enumerate(ranked_texts, start=1):
        for needle in needles:
            if substring_present(needle, text, case_sensitive=case_sensitive):
                return rank
    return None


def hit_fraction_at_k(
    needles: Sequence[str],
    ranked_texts: Sequence[str],
    top_k: int,
    *,
    case_sensitive: bool = False,
) -> float:
    """Fraction of needles that appear in at least one of the top-K texts."""
    if not needles:
        return 1.0
    window = ranked_texts[:top_k]
    hits = 0
    for needle in needles:
        if any(
            substring_present(needle, text, case_sensitive=case_sensitive)
            for text in window
        ):
            hits += 1
    return hits / len(needles)


def trap_suppressed(
    expected: Sequence[str],
    forbidden: Sequence[str],
    ranked_texts: Sequence[str],
    top_k: int,
    *,
    case_sensitive: bool = False,
) -> bool:
    """True when no forbidden string outranks the best expected hit in top-K.

    If there are no forbidden strings, returns True. If expected is absent
    from top-K but a forbidden string is present, returns False.
    """
    if not forbidden:
        return True
    window = ranked_texts[:top_k]
    expected_rank = first_match_rank(expected, window, case_sensitive=case_sensitive)
    forbidden_rank = first_match_rank(forbidden, window, case_sensitive=case_sensitive)
    if forbidden_rank is None:
        return True
    if expected_rank is None:
        return False
    return forbidden_rank > expected_rank


def reciprocal_rank(
    needles: Sequence[str],
    ranked_texts: Sequence[str],
    top_k: int,
    *,
    case_sensitive: bool = False,
) -> float:
    rank = first_match_rank(
        needles, ranked_texts[:top_k], case_sensitive=case_sensitive
    )
    if rank is None:
        return 0.0
    return 1.0 / rank


def chunk_search_text(chunk: dict[str, Any]) -> str:
    """Build the searchable text for one /query/data chunk.

    Includes ``content``, optional ``content_headings`` (str or list), and
    ``file_path`` so section labels in headings can score without full Act text.
    """
    parts: list[str] = []
    content = chunk.get("content")
    if isinstance(content, str) and content.strip():
        parts.append(content)
    headings = chunk.get("content_headings")
    if isinstance(headings, str) and headings.strip():
        parts.append(headings)
    elif isinstance(headings, list):
        parts.extend(str(item) for item in headings if item)
    file_path = chunk.get("file_path")
    if isinstance(file_path, str) and file_path.strip():
        parts.append(file_path)
    return "\n".join(parts)


def extract_ranked_chunk_texts(query_data_payload: dict[str, Any]) -> list[str]:
    """Extract ranked chunk texts from a /query/data JSON body.

    Expected shape (current API)::

        {
          "status": "success",
          "message": "...",
          "data": {
            "chunks": [{"content": "...", "content_headings": "...", ...}, ...],
            "entities": [...],
            "relationships": [...],
            "references": [...]
          },
          "metadata": {...}
        }

    Also tolerates a legacy top-level ``chunks`` list for mocked fixtures.
    """
    if not isinstance(query_data_payload, dict):
        return []

    data = query_data_payload.get("data")
    chunks: Any = None
    if isinstance(data, dict):
        chunks = data.get("chunks")
    if chunks is None:
        chunks = query_data_payload.get("chunks")
    if not isinstance(chunks, list):
        return []

    ranked: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            if chunk.strip():
                ranked.append(chunk)
            continue
        if isinstance(chunk, dict):
            text = chunk_search_text(chunk)
            if text.strip():
                ranked.append(text)
    return ranked


def extract_answer_and_citation_blob(query_payload: dict[str, Any]) -> tuple[str, str]:
    """Return (answer_text, citation_blob) from a /query JSON body."""
    answer = str(query_payload.get("response") or "")
    references = query_payload.get("references") or []
    citation_parts: list[str] = [answer]
    if isinstance(references, list):
        for ref in references:
            if not isinstance(ref, dict):
                continue
            for key in ("file_path", "reference_id"):
                value = ref.get(key)
                if isinstance(value, str) and value.strip():
                    citation_parts.append(value)
            content = ref.get("content")
            if isinstance(content, list):
                citation_parts.extend(str(item) for item in content if item)
            elif isinstance(content, str) and content.strip():
                citation_parts.append(content)
    return answer, "\n".join(citation_parts)


@dataclass
class RetrievalCaseScore:
    case_id: str
    question: str
    mode: str
    top_k: int
    section_hit_at_k: float
    provision_hit_at_k: float
    trap_suppress: bool
    mrr: float
    chunk_count: int
    expected_sections: list[str] = field(default_factory=list)
    expected_chunk_contains: list[str] = field(default_factory=list)
    forbidden_chunk_contains: list[str] = field(default_factory=list)

    def passes(
        self,
        *,
        min_section_hit: float,
        min_provision_hit: float,
        require_trap_suppress: bool,
        min_mrr: float,
    ) -> bool:
        if self.section_hit_at_k < min_section_hit:
            return False
        if self.provision_hit_at_k < min_provision_hit:
            return False
        if require_trap_suppress and not self.trap_suppress:
            return False
        if self.mrr < min_mrr:
            return False
        return True


@dataclass
class RulesCaseScore:
    case_id: str
    question: str
    mode: str
    expected_answer_ok: bool
    forbidden_answer_ok: bool
    expected_citation_ok: bool
    forbidden_citation_ok: bool

    @property
    def passed(self) -> bool:
        return (
            self.expected_answer_ok
            and self.forbidden_answer_ok
            and self.expected_citation_ok
            and self.forbidden_citation_ok
        )


def score_retrieval_case(
    case: dict[str, Any],
    ranked_texts: Sequence[str],
    *,
    mode: str,
    top_k: int,
) -> RetrievalCaseScore:
    expected_sections = [str(x) for x in case.get("expected_sections") or []]
    expected_contains = [str(x) for x in case.get("expected_chunk_contains") or []]
    forbidden = [str(x) for x in case.get("forbidden_chunk_contains") or []]
    mrr_needles = expected_contains or expected_sections

    return RetrievalCaseScore(
        case_id=str(case.get("id") or case.get("question") or "unknown"),
        question=str(case.get("question") or ""),
        mode=mode,
        top_k=top_k,
        section_hit_at_k=hit_fraction_at_k(expected_sections, ranked_texts, top_k),
        provision_hit_at_k=hit_fraction_at_k(expected_contains, ranked_texts, top_k),
        trap_suppress=trap_suppressed(
            expected_sections + expected_contains,
            forbidden,
            ranked_texts,
            top_k,
        ),
        mrr=reciprocal_rank(mrr_needles, ranked_texts, top_k),
        chunk_count=len(ranked_texts),
        expected_sections=expected_sections,
        expected_chunk_contains=expected_contains,
        forbidden_chunk_contains=forbidden,
    )


def score_rules_case(
    case: dict[str, Any],
    answer: str,
    citation_blob: str,
) -> RulesCaseScore:
    expected_answer = [str(x) for x in case.get("expected_answer_contains") or []]
    forbidden_answer = [str(x) for x in case.get("forbidden_answer_contains") or []]
    expected_cite = [str(x) for x in case.get("expected_citation_contains") or []]
    forbidden_cite = [str(x) for x in case.get("forbidden_citation_contains") or []]

    def _all_present(needles: Sequence[str], text: str) -> bool:
        if not needles:
            return True
        return all(substring_present(n, text) for n in needles)

    def _none_present(needles: Sequence[str], text: str) -> bool:
        return not any(substring_present(n, text) for n in needles)

    return RulesCaseScore(
        case_id=str(case.get("id") or case.get("question") or "unknown"),
        question=str(case.get("question") or ""),
        mode=str(case.get("mode") or "mix"),
        expected_answer_ok=_all_present(expected_answer, answer),
        forbidden_answer_ok=_none_present(forbidden_answer, answer),
        expected_citation_ok=_all_present(expected_cite, citation_blob),
        forbidden_citation_ok=_none_present(forbidden_cite, citation_blob),
    )


def summarize_retrieval(scores: Sequence[RetrievalCaseScore]) -> dict[str, Any]:
    if not scores:
        return {
            "queries": 0,
            "average_section_hit_at_k": 0.0,
            "average_provision_hit_at_k": 0.0,
            "trap_suppress_rate": 0.0,
            "mean_reciprocal_rank": 0.0,
        }
    n = len(scores)
    return {
        "queries": n,
        "average_section_hit_at_k": sum(s.section_hit_at_k for s in scores) / n,
        "average_provision_hit_at_k": sum(s.provision_hit_at_k for s in scores) / n,
        "trap_suppress_rate": sum(1 for s in scores if s.trap_suppress) / n,
        "mean_reciprocal_rank": sum(s.mrr for s in scores) / n,
        "by_mode": _by_mode_summary(scores),
    }


def _by_mode_summary(scores: Sequence[RetrievalCaseScore]) -> dict[str, Any]:
    modes: dict[str, list[RetrievalCaseScore]] = {}
    for score in scores:
        modes.setdefault(score.mode, []).append(score)
    out: dict[str, Any] = {}
    for mode, group in sorted(modes.items()):
        n = len(group)
        out[mode] = {
            "queries": n,
            "average_section_hit_at_k": sum(s.section_hit_at_k for s in group) / n,
            "average_provision_hit_at_k": sum(s.provision_hit_at_k for s in group) / n,
            "trap_suppress_rate": sum(1 for s in group if s.trap_suppress) / n,
            "mean_reciprocal_rank": sum(s.mrr for s in group) / n,
        }
    return out


def summarize_rules(scores: Sequence[RulesCaseScore]) -> dict[str, Any]:
    if not scores:
        return {"queries": 0, "pass_rate": 0.0}
    n = len(scores)
    return {
        "queries": n,
        "pass_rate": sum(1 for s in scores if s.passed) / n,
        "failed_case_ids": [s.case_id for s in scores if not s.passed],
    }


def load_oracle(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty cases list")
    for entry in cases:
        if not str(entry.get("question", "")).strip():
            raise ValueError("Each oracle case needs a question")
    return payload


def load_rules_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path} must contain a non-empty cases list")
    for entry in cases:
        if not str(entry.get("question", "")).strip():
            raise ValueError("Each rules case needs a question")
    return payload


class TaxFitnessClient:
    """Thin HTTP client for LightRAG tax fitness checks."""

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self.api_url = _normalize_api_url(api_url)
        self.api_key = api_key
        self.timeout = timeout or httpx.Timeout(
            CONNECT_TIMEOUT_SECONDS,
            read=READ_TIMEOUT_SECONDS,
        )

    def health_check(self) -> dict[str, Any]:
        headers = _auth_headers(self.api_key)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(f"{self.api_url}/health", headers=headers)
                response.raise_for_status()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    return response.json()
                return {"status": "ok", "raw": response.text[:200]}
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPError) as exc:
            raise ServerUnavailableError(
                f"LightRAG API unavailable at {self.api_url}/health ({exc})"
            ) from exc

    def query_data(
        self,
        question: str,
        *,
        mode: str,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": question,
            "mode": mode,
            # /query/data always returns structured retrieval; references always included.
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if chunk_top_k is not None:
            payload["chunk_top_k"] = chunk_top_k
        return self._post_json("/query/data", payload)

    def query_answer(
        self,
        question: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": question,
            "mode": mode,
            "include_references": True,
            "include_chunk_content": True,
            "stream": False,
        }
        if top_k is not None:
            payload["top_k"] = top_k
        return self._post_json("/query", payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = _auth_headers(self.api_key)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_url}{path}",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
        except httpx.ConnectError as exc:
            raise ServerUnavailableError(
                f"Cannot connect to LightRAG API at {self.api_url}{path}: {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"LightRAG API error {exc.response.status_code} on {path}: "
                f"{exc.response.text[:500]}"
            ) from exc


def run_retrieval_fitness(
    client: TaxFitnessClient,
    oracle: dict[str, Any],
    *,
    modes: Sequence[str] | None = None,
) -> list[RetrievalCaseScore]:
    defaults = oracle.get("defaults") or {}
    default_modes = modes or defaults.get("modes") or list(DEFAULT_MODES)
    default_top_k = int(defaults.get("top_k") or 10)
    scores: list[RetrievalCaseScore] = []

    for case in oracle["cases"]:
        case_modes = case.get("modes") or default_modes
        top_k = int(case.get("top_k") or default_top_k)
        for mode in case_modes:
            payload = client.query_data(
                str(case["question"]),
                mode=str(mode),
                top_k=top_k,
                chunk_top_k=top_k,
            )
            if payload.get("status") == "failure":
                ranked: list[str] = []
            else:
                ranked = extract_ranked_chunk_texts(payload)
            scores.append(
                score_retrieval_case(case, ranked, mode=str(mode), top_k=top_k)
            )
    return scores


def run_rules_fitness(
    client: TaxFitnessClient,
    rules_dataset: dict[str, Any],
) -> list[RulesCaseScore]:
    """Deterministic citation/trap checks on /query answers.

    Optional future hook: RAGAS (Faithfulness / AnswerRelevancy) can wrap the
    same answer+contexts after deterministic judges pass — not required for v1.
    """
    defaults = rules_dataset.get("defaults") or {}
    default_mode = str(defaults.get("mode") or "mix")
    scores: list[RulesCaseScore] = []
    for case in rules_dataset["cases"]:
        mode = str(case.get("mode") or default_mode)
        payload = client.query_answer(str(case["question"]), mode=mode)
        answer, citation_blob = extract_answer_and_citation_blob(payload)
        scores.append(score_rules_case(case, answer, citation_blob))
    return scores


def gates_passed(
    retrieval_summary: dict[str, Any],
    rules_summary: dict[str, Any] | None,
    *,
    min_section_hit: float,
    min_provision_hit: float,
    min_trap_suppress: float,
    min_mrr: float,
    min_rules_pass_rate: float,
    check_rules: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if retrieval_summary.get("queries", 0) > 0:
        if retrieval_summary["average_section_hit_at_k"] < min_section_hit:
            failures.append(
                f"section_hit_at_k "
                f"{retrieval_summary['average_section_hit_at_k']:.3f} "
                f"< {min_section_hit}"
            )
        if retrieval_summary["average_provision_hit_at_k"] < min_provision_hit:
            failures.append(
                f"provision_hit_at_k "
                f"{retrieval_summary['average_provision_hit_at_k']:.3f} "
                f"< {min_provision_hit}"
            )
        if retrieval_summary["trap_suppress_rate"] < min_trap_suppress:
            failures.append(
                f"trap_suppress_rate "
                f"{retrieval_summary['trap_suppress_rate']:.3f} "
                f"< {min_trap_suppress}"
            )
        if retrieval_summary["mean_reciprocal_rank"] < min_mrr:
            failures.append(
                f"mrr {retrieval_summary['mean_reciprocal_rank']:.3f} < {min_mrr}"
            )
    if check_rules and rules_summary is not None and rules_summary.get("queries", 0) > 0:
        if rules_summary["pass_rate"] < min_rules_pass_rate:
            failures.append(
                f"rules_pass_rate {rules_summary['pass_rate']:.3f} "
                f"< {min_rules_pass_rate}"
            )
    return (not failures), failures


def build_report(
    retrieval_scores: Sequence[RetrievalCaseScore],
    rules_scores: Sequence[RulesCaseScore] | None,
    *,
    api_url: str,
    gate_failures: Sequence[str],
) -> dict[str, Any]:
    retrieval_summary = summarize_retrieval(retrieval_scores)
    rules_summary = summarize_rules(rules_scores or [])
    return {
        "api_url": api_url,
        "gates_passed": len(gate_failures) == 0,
        "gate_failures": list(gate_failures),
        "retrieval": {
            "summary": retrieval_summary,
            "cases": [asdict(score) for score in retrieval_scores],
        },
        "rules": {
            "summary": rules_summary,
            "cases": [asdict(score) for score in (rules_scores or [])],
            # RAGAS optional later — deterministic judges only in v1.
            "ragas": None,
        },
    }


def print_stdout_report(report: dict[str, Any]) -> None:
    retrieval = report["retrieval"]["summary"]
    rules = report["rules"]["summary"]
    print("LightRAG tax retrieval fitness")
    print(f"API: {report['api_url']}")
    print(f"Retrieval queries: {retrieval.get('queries', 0)}")
    print(f"  section Hit@K:   {retrieval.get('average_section_hit_at_k', 0.0):.3f}")
    print(f"  provision Hit@K: {retrieval.get('average_provision_hit_at_k', 0.0):.3f}")
    print(f"  trap suppress:   {retrieval.get('trap_suppress_rate', 0.0):.3f}")
    print(f"  MRR:             {retrieval.get('mean_reciprocal_rank', 0.0):.3f}")
    print(f"Rules queries: {rules.get('queries', 0)}")
    print(f"  pass rate:       {rules.get('pass_rate', 0.0):.3f}")
    if report["gates_passed"]:
        print("Gates: PASSED")
    else:
        print("Gates: FAILED")
        for failure in report["gate_failures"]:
            print(f"  - {failure}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tax retrieval fitness against a local LightRAG API."
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
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument("--rules", default=str(DEFAULT_RULES))
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated retrieval modes (default: naive,local,hybrid,mix)",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip /query rules/guidance checks",
    )
    parser.add_argument(
        "--rules-only",
        action="store_true",
        help="Skip /query/data retrieval scoring",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the full JSON report",
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
            "Tax fitness live eval is gated. Set TAX_EVAL_LIVE=true "
            "(or LIGHTRAG_RUN_INTEGRATION=true), or pass --allow-ungated.",
            file=sys.stderr,
        )
        return 2

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    client = TaxFitnessClient(api_url=args.api_url, api_key=args.api_key)

    try:
        client.health_check()
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Start the API (e.g. lightrag-server / WebUI backend on :9621) "
            "and ensure tax fixtures are ingested before re-running.",
            file=sys.stderr,
        )
        return 2

    retrieval_scores: list[RetrievalCaseScore] = []
    rules_scores: list[RulesCaseScore] = []

    try:
        if not args.rules_only:
            oracle = load_oracle(Path(args.oracle).expanduser())
            retrieval_scores = run_retrieval_fitness(client, oracle, modes=modes)
        if not args.retrieval_only:
            rules_dataset = load_rules_dataset(Path(args.rules).expanduser())
            rules_scores = run_rules_fitness(client, rules_dataset)
    except ServerUnavailableError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Tax fitness run failed: {exc}", file=sys.stderr)
        return 2

    retrieval_summary = summarize_retrieval(retrieval_scores)
    rules_summary = summarize_rules(rules_scores)
    passed, failures = gates_passed(
        retrieval_summary,
        rules_summary,
        min_section_hit=args.min_section_hit,
        min_provision_hit=args.min_provision_hit,
        min_trap_suppress=args.min_trap_suppress,
        min_mrr=args.min_mrr,
        min_rules_pass_rate=args.min_rules_pass_rate,
        check_rules=not args.retrieval_only,
    )
    report = build_report(
        retrieval_scores,
        rules_scores if not args.retrieval_only else [],
        api_url=client.api_url,
        gate_failures=failures if not passed else [],
    )
    print_stdout_report(report)

    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote JSON report: {output_path}")
    else:
        # Always emit machine-readable report on stdout after the summary.
        print("--- JSON ---")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
