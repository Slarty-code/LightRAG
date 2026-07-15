"""Unit and gated integration tests for tax retrieval fitness scoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lightrag.evaluation.tax_retrieval_fitness import (
    DEFAULT_ORACLE,
    DEFAULT_RULES,
    ServerUnavailableError,
    TaxFitnessClient,
    extract_ranked_chunk_texts,
    hit_fraction_at_k,
    live_eval_enabled,
    load_oracle,
    load_rules_dataset,
    reciprocal_rank,
    score_retrieval_case,
    score_rules_case,
    summarize_retrieval,
    trap_suppressed,
)


FIXTURES = Path("lightrag/evaluation/tax_fixtures")


def test_fixture_stubs_exist_and_load():
    assert DEFAULT_ORACLE.is_file()
    assert DEFAULT_RULES.is_file()
    oracle = load_oracle(DEFAULT_ORACLE)
    rules = load_rules_dataset(DEFAULT_RULES)
    assert oracle["cases"]
    assert rules["cases"]
    case = oracle["cases"][0]
    assert "expected_sections" in case
    assert "expected_chunk_contains" in case
    assert "forbidden_chunk_contains" in case


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
    case = oracle["cases"][0]
    mode = (case.get("modes") or ["mix"])[0]
    payload = client.query_data(str(case["question"]), mode=str(mode), top_k=5)
    assert isinstance(payload, dict)
    assert "status" in payload or "data" in payload or "chunks" in payload
