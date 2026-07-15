"""Unit and gated integration tests for tax retrieval fitness scoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lightrag.evaluation.tax_fixtures import (
    LIVE_ORACLE_TEMPLATE_JSON,
    PG_MARKDOWN,
    PR_MARKDOWN,
    TAX_SCENARIO_ORACLE_JSON,
    fixture_path,
)
from lightrag.evaluation.tax_retrieval_fitness import (
    DEFAULT_ORACLE,
    DEFAULT_RULES,
    ServerUnavailableError,
    TaxFitnessClient,
    case_enabled,
    citation_set,
    classify_instrument,
    enabled_cases,
    extract_graph_context_present,
    extract_ranked_chunk_texts,
    extract_ranked_chunks,
    hit_fraction_at_k,
    jaccard,
    live_eval_enabled,
    load_oracle,
    load_rules_dataset,
    reciprocal_rank,
    run_retrieval_fitness,
    score_retrieval_case,
    score_rules_case,
    set_delta,
    source_type_recall,
    summarize_retrieval,
    trap_suppressed,
)


FIXTURES = Path("lightrag/evaluation/tax_fixtures")


def test_fixture_stubs_exist_and_load():
    assert DEFAULT_ORACLE.is_file()
    assert DEFAULT_RULES.is_file()
    assert fixture_path(PG_MARKDOWN).is_file()
    assert fixture_path(PR_MARKDOWN).is_file()
    scenario = json.loads(fixture_path(TAX_SCENARIO_ORACLE_JSON).read_text())
    assert scenario["cases"] == []
    assert "NOT YET" in scenario["description"]

    oracle = load_oracle(DEFAULT_ORACLE)
    rules = load_rules_dataset(DEFAULT_RULES)
    assert oracle["cases"]
    assert rules["cases"]
    case = oracle["cases"][0]
    assert "expected_sections" in case
    assert "expected_chunk_contains" in case
    assert "forbidden_chunk_contains" in case

    td_case = next(c for c in oracle["cases"] if c["id"] == "td_apportionment_s8_1")
    assert "expected_sources" in td_case
    assert td_case["expected_sources"]["tds"]
    assert td_case["expected_issues"] == []
    assert td_case["expected_benefits"] == []
    assert "forbidden_sources" in td_case
    assert "optional v2" in oracle["description"].casefold()


def test_live_oracle_template_loads_all_disabled():
    template_path = fixture_path(LIVE_ORACLE_TEMPLATE_JSON)
    assert template_path.is_file()
    payload = load_oracle(template_path)
    assert "TEMPLATE" in payload["description"]
    assert payload["cases"]
    assert all(case.get("enabled") is False for case in payload["cases"])
    assert enabled_cases(payload) == []
    ids = {c["id"] for c in payload["cases"]}
    assert "live_keyword_trap_deduction" in ids
    assert "live_act_td_crosswalk" in ids
    assert "live_multi_source_pack" in ids
    multi = next(c for c in payload["cases"] if c["id"] == "live_multi_source_pack")
    assert multi["expected_sources"]["pgs"] == []
    assert multi["expected_sources"]["prs"] == []

    client = TaxFitnessClient(api_url="http://localhost:9621")
    with pytest.raises(ValueError, match="no enabled cases"):
        run_retrieval_fitness(client, payload)


def test_case_enabled_false_skipped_by_runner():
    assert case_enabled({"question": "q"}) is True
    assert case_enabled({"enabled": True, "question": "q"}) is True
    assert case_enabled({"enabled": False, "question": "q"}) is False
    assert case_enabled({"enabled": "false", "question": "q"}) is False

    oracle = {
        "defaults": {"modes": ["mix"], "top_k": 2},
        "cases": [
            {
                "id": "disabled",
                "enabled": False,
                "question": "should skip",
                "expected_sections": ["Section 1"],
                "expected_chunk_contains": ["basic rate"],
                "forbidden_chunk_contains": [],
                "modes": ["mix"],
            },
            {
                "id": "active",
                "question": "What is the basic rate?",
                "expected_sections": ["Section 1"],
                "expected_chunk_contains": ["basic rate"],
                "forbidden_chunk_contains": [],
                "modes": ["mix"],
            },
        ],
    }
    assert [c["id"] for c in enabled_cases(oracle)] == ["active"]

    response_json = {
        "status": "success",
        "data": {
            "chunks": [
                {
                    "content": "Section 1 sets the basic rate of income tax.",
                    "content_headings": "Section 1",
                    "file_path": "tax.pdf",
                }
            ],
            "entities": [],
            "relationships": [],
        },
    }
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_json

    client = TaxFitnessClient(api_url="http://localhost:9621")
    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        client_cls.return_value = mock_client
        scores = run_retrieval_fitness(client, oracle)

    assert len(scores) == 1
    assert scores[0].case_id == "active"
    assert mock_client.post.call_count == 1


def test_extract_ranked_chunk_texts_prefers_data_chunks():
    payload = {
        "status": "success",
        "message": "ok",
        "data": {
            "chunks": [
                {
                    "content": "Income tax basic rate is set here.",
                    "content_headings": "Part 1 → Section 1",
                    "file_path": "ita.pdf",
                    "chunk_id": "c1",
                    "reference_id": "1",
                },
                {
                    "content": "Unrelated stamp duty material.",
                    "file_path": "other.pdf",
                },
            ]
        },
        "metadata": {"query_mode": "mix"},
    }
    ranked = extract_ranked_chunk_texts(payload)
    assert len(ranked) == 2
    assert "Section 1" in ranked[0]
    assert "basic rate" in ranked[0]


def test_extract_ranked_chunk_texts_legacy_top_level_chunks():
    payload = {"chunks": [{"content": "legacy chunk"}, "plain string"]}
    assert extract_ranked_chunk_texts(payload) == ["legacy chunk", "plain string"]


def test_classify_instrument_filename_and_content():
    assert classify_instrument("td_general_deductions.md") == "td"
    assert classify_instrument("pg_general_deductions.md") == "pg"
    assert classify_instrument("pr_general_deductions.md") == "pr"
    assert classify_instrument("mini_ita_deductions.md") == "act"
    assert classify_instrument("TD 2024/1 (Synthetic) apportionment") == "td"
    assert classify_instrument("Practice Guide PG 2024/1 summary") == "pg"
    assert classify_instrument("Product Ruling PR 2024/1 arrangement") == "pr"
    assert classify_instrument("Income Tax Assessment Act 1997 s 8-1") == "act"
    assert classify_instrument("random memo without markers") == "unknown"


def test_extract_ranked_chunks_infers_instrument():
    payload = {
        "data": {
            "chunks": [
                {
                    "content": "Apportionment under to the extent.",
                    "content_headings": "TD 2024/1 (Synthetic)",
                    "file_path": "td_general_deductions.md",
                },
                {
                    "content": "General deduction positive limbs.",
                    "file_path": "mini_ita_deductions.md",
                },
                "plain practice guide pg 1 filler",
            ],
            "entities": [{"name": "s 8-1"}],
            "relationships": [],
        }
    }
    chunks = extract_ranked_chunks(payload)
    assert len(chunks) == 3
    assert chunks[0]["instrument"] == "td"
    assert chunks[0]["file_path"] == "td_general_deductions.md"
    assert chunks[1]["instrument"] == "act"
    assert chunks[2]["instrument"] == "pg"
    assert extract_ranked_chunk_texts(payload)[0].startswith("Apportionment")
    assert extract_graph_context_present(payload) is True
    assert extract_graph_context_present({"data": {"entities": [], "relationships": []}}) is False


def test_source_type_recall_and_citation_helpers():
    chunks = [
        {
            "content": "s 8-1 general deduction",
            "content_headings": None,
            "file_path": "mini_ita_deductions.md",
            "instrument": "act",
        },
        {
            "content": "TD 2024/1 apportionment",
            "content_headings": None,
            "file_path": "td_general_deductions.md",
            "instrument": "td",
        },
        {
            "content": "unrelated",
            "content_headings": None,
            "file_path": "other.md",
            "instrument": "unknown",
        },
    ]
    recall = source_type_recall(
        {
            "acts": ["s 8-1", "mini_ita_deductions"],
            "tds": ["TD 2024/1", "missing_td_marker"],
            "pgs": [],
            "prs": [],
        },
        chunks,
        top_k=3,
    )
    assert recall["act"] == 1.0
    assert recall["td"] == 0.5
    assert "pg" not in recall
    assert recall["overall"] == pytest.approx(0.75)

    empty = source_type_recall(None, chunks, top_k=3)
    assert empty == {"overall": 1.0}

    cites = citation_set(chunks, top_k=2)
    assert cites == {"mini_ita_deductions.md", "td_general_deductions.md"}
    assert jaccard(cites, cites) == 1.0
    assert jaccard(set(), set()) == 1.0
    delta = set_delta({"a", "b"}, {"b", "c"})
    assert delta["added"] == ["c"]
    assert delta["removed"] == ["a"]


def test_score_retrieval_soft_source_type_recall():
    case = {
        "id": "multi",
        "question": "crosswalk?",
        "expected_sections": ["s 8-1"],
        "expected_chunk_contains": ["apportionment"],
        "forbidden_chunk_contains": [],
        "expected_sources": {
            "acts": ["s 8-1"],
            "tds": ["TD 2024/1"],
            "pgs": ["PG 2024/1"],
            "prs": [],
        },
    }
    ranked_chunks = [
        {
            "content": "s 8-1 and apportionment guidance",
            "content_headings": "TD 2024/1 (Synthetic)",
            "file_path": "td_general_deductions.md",
            "instrument": "td",
        },
        {
            "content": "s 8-1 positive limbs",
            "content_headings": None,
            "file_path": "mini_ita_deductions.md",
            "instrument": "act",
        },
    ]
    ranked = extract_ranked_chunk_texts({"chunks": ranked_chunks})
    score = score_retrieval_case(
        case,
        ranked,
        mode="mix",
        top_k=2,
        ranked_chunks=ranked_chunks,
        graph_context_present=True,
    )
    assert score.source_type_recall_by_type["act"] == 1.0
    assert score.source_type_recall_by_type["td"] == 1.0
    assert score.source_type_recall_by_type["pg"] == 0.0
    assert score.source_type_recall == pytest.approx(2.0 / 3.0)
    assert score.graph_context_present is True
    # v1 passes() ignores source_type_recall unless min is provided
    assert score.passes(
        min_section_hit=0.5,
        min_provision_hit=0.5,
        require_trap_suppress=True,
        min_mrr=0.3,
    )
    assert not score.passes(
        min_section_hit=0.5,
        min_provision_hit=0.5,
        require_trap_suppress=True,
        min_mrr=0.3,
        min_source_type_recall=0.9,
    )


def test_score_rules_soft_issues_and_benefits():
    base = {
        "id": "r_soft",
        "question": "issues?",
        "expected_answer_contains": ["deduction"],
        "forbidden_answer_contains": [],
        "expected_citation_contains": ["s 8-1"],
        "forbidden_citation_contains": [],
    }
    # Empty / absent lists: soft fields default True and do not fail.
    empty_lists = {
        **base,
        "expected_issues": [],
        "expected_benefits": [],
    }
    ok = score_rules_case(
        empty_lists,
        "A deduction is available.",
        "s 8-1",
    )
    assert ok.issues_ok is True
    assert ok.benefits_ok is True
    assert ok.passed is True

    with_lists = {
        **base,
        "expected_issues": ["apportionment issue"],
        "expected_benefits": ["income benefit"],
    }
    miss = score_rules_case(
        with_lists,
        "A deduction is available.",
        "s 8-1",
    )
    assert miss.issues_ok is False
    assert miss.benefits_ok is False
    assert miss.passed is False

    hit = score_rules_case(
        with_lists,
        "A deduction is available; apportionment issue noted.",
        "s 8-1\nincome benefit applies",
    )
    assert hit.issues_ok is True
    assert hit.benefits_ok is True
    assert hit.passed is True


def test_hit_fraction_and_mrr_helpers():
    ranked = [
        "Stamp duty overview",
        "Part 1 → Section 1\nbasic rate of income tax",
        "capital gains rates",
    ]
    assert hit_fraction_at_k(["Section 1", "s. 1"], ranked, top_k=2) == 0.5
    assert hit_fraction_at_k(["basic rate"], ranked, top_k=2) == 1.0
    assert reciprocal_rank(["basic rate"], ranked, top_k=3) == pytest.approx(0.5)
    assert reciprocal_rank(["missing"], ranked, top_k=3) == 0.0


def test_trap_must_not_outrank_expected():
    ranked = [
        "capital gains trap phrase",
        "Section 1 basic rate",
    ]
    assert trap_suppressed(
        ["Section 1", "basic rate"],
        ["capital gains"],
        ranked,
        top_k=2,
    ) is False
    assert trap_suppressed(
        ["Section 1", "basic rate"],
        ["capital gains"],
        list(reversed(ranked)),
        top_k=2,
    ) is True
    assert trap_suppressed(
        ["Section 1"],
        ["capital gains"],
        ["unrelated only"],
        top_k=2,
    ) is True


def test_score_retrieval_case_metrics():
    case = {
        "id": "t1",
        "question": "basic rate?",
        "expected_sections": ["Section 1"],
        "expected_chunk_contains": ["basic rate"],
        "forbidden_chunk_contains": ["stamp duty"],
    }
    ranked = [
        "Section 1 — the basic rate of income tax",
        "stamp duty land tax",
    ]
    score = score_retrieval_case(case, ranked, mode="mix", top_k=2)
    assert score.section_hit_at_k == 1.0
    assert score.provision_hit_at_k == 1.0
    assert score.trap_suppress is True
    assert score.mrr == 1.0


def test_score_rules_case_citations():
    case = {
        "id": "r1",
        "question": "cite?",
        "expected_answer_contains": ["basic rate"],
        "forbidden_answer_contains": ["stamp duty"],
        "expected_citation_contains": ["Section 1"],
        "forbidden_citation_contains": ["capital gains"],
    }
    answer = "The basic rate is defined in the Act."
    citation = "Section 1\nThe basic rate applies to income tax."
    score = score_rules_case(case, answer, citation)
    assert score.passed is True

    bad = score_rules_case(
        case,
        "Mentions stamp duty incorrectly.",
        "capital gains schedule",
    )
    assert bad.passed is False


def test_summarize_retrieval_aggregates():
    case = {
        "id": "a",
        "question": "q",
        "expected_sections": ["Section 1"],
        "expected_chunk_contains": ["basic rate"],
        "forbidden_chunk_contains": [],
    }
    scores = [
        score_retrieval_case(
            case, ["Section 1 basic rate"], mode="naive", top_k=1
        ),
        score_retrieval_case(case, ["nope"], mode="local", top_k=1),
    ]
    summary = summarize_retrieval(scores)
    assert summary["queries"] == 2
    assert summary["average_section_hit_at_k"] == pytest.approx(0.5)
    assert "naive" in summary["by_mode"]


def test_health_check_raises_when_down():
    client = TaxFitnessClient(api_url="http://localhost:9621")

    def _raise(*_args, **_kwargs):
        raise httpx.ConnectError("refused")

    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.get.side_effect = _raise
        client_cls.return_value = mock_client
        with pytest.raises(ServerUnavailableError):
            client.health_check()


def test_query_data_mocked_roundtrip_scoring():
    client = TaxFitnessClient(api_url="http://localhost:9621")
    response_json = {
        "status": "success",
        "message": "Query executed successfully",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [
                {
                    "content": "Section 1 sets the basic rate of income tax.",
                    "content_headings": "Section 1",
                    "file_path": "tax.pdf",
                    "chunk_id": "c1",
                    "reference_id": "1",
                }
            ],
            "references": [{"reference_id": "1", "file_path": "tax.pdf"}],
        },
        "metadata": {"query_mode": "hybrid"},
    }

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = response_json

    with patch("httpx.Client") as client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        client_cls.return_value = mock_client
        payload = client.query_data("What is the basic rate?", mode="hybrid", top_k=5)

    ranked = extract_ranked_chunk_texts(payload)
    case = json.loads(FIXTURES.joinpath("tax_retrieval_oracle.json").read_text())[
        "cases"
    ][0]
    score = score_retrieval_case(case, ranked, mode="hybrid", top_k=5)
    assert score.section_hit_at_k > 0
    assert score.provision_hit_at_k > 0


@pytest.mark.integration
def test_live_tax_fitness_preflight_or_skip():
    """Hit localhost:9621 only when live flag is set; skip if server is down."""
    if not live_eval_enabled():
        pytest.skip("Set TAX_EVAL_LIVE=true or LIGHTRAG_RUN_INTEGRATION=true")

    client = TaxFitnessClient()
    try:
        client.health_check()
    except ServerUnavailableError as exc:
        pytest.skip(str(exc))

    oracle = load_oracle(DEFAULT_ORACLE)
    cases = enabled_cases(oracle)
    assert cases, "default oracle must have at least one enabled case"
    case = cases[0]
    mode = (case.get("modes") or ["mix"])[0]
    payload = client.query_data(str(case["question"]), mode=str(mode), top_k=5)
    assert isinstance(payload, dict)
    assert "status" in payload or "data" in payload or "chunks" in payload
