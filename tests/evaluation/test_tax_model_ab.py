"""Unit tests for tax_model_ab tagging helpers (no live API)."""

from __future__ import annotations

from lightrag.evaluation.tax_model_ab import (
    _slug,
    build_tagged_report,
    default_report_filename,
    extract_model_tags,
)


def test_extract_prefers_query_role_model():
    health = {
        "status": "healthy",
        "configuration": {
            "llm_binding": "openai",
            "llm_binding_host": "https://openrouter.ai/api/v1",
            "llm_model": "vendor/cheap-model",
            "role_llm_config": {
                "QUERY": {
                    "model": "vendor/zdr-strong-model",
                    "binding": "openai",
                    "host": "https://openrouter.ai/api/v1",
                },
                "KEYWORD": {
                    "model": "vendor/cheap-model",
                    "binding": "openai",
                    "host": "https://openrouter.ai/api/v1",
                },
            },
        },
    }
    tags = extract_model_tags(health)
    assert tags["query_llm_model"] == "vendor/zdr-strong-model"
    assert tags["keyword_llm_model"] == "vendor/cheap-model"
    assert tags["llm_model"] == "vendor/cheap-model"
    assert tags["source"] == "role_llm_config.QUERY"
    assert tags["query_model_explicit"] is True


def test_extract_falls_back_to_base_llm_model():
    health = {
        "configuration": {
            "llm_model": "gpt-4o-mini",
            "llm_binding": "openai",
            "role_llm_config": {},
        }
    }
    tags = extract_model_tags(health)
    assert tags["query_llm_model"] == "gpt-4o-mini"
    assert tags["source"] == "configuration.llm_model"
    assert tags["query_model_explicit"] is False


def test_slug_and_filename_include_label_and_model():
    assert _slug("OpenAI/gpt-4o:mini") == "openai__gpt-4o_mini"
    tags = {"query_llm_model": "vendor/zdr-model"}
    name = default_report_filename(tags, "zdr")
    assert name.startswith("fitness_zdr_vendor__zdr-model_")
    assert name.endswith(".json")


def test_build_tagged_report_nests_model_ab():
    fitness = {"gates_passed": True, "retrieval": {"summary": {}}}
    tags = extract_model_tags(
        {"configuration": {"llm_model": "base-model", "role_llm_config": {}}}
    )
    tagged = build_tagged_report(
        fitness, model_tags=tags, label="cheap", run_id="20260101T000000Z"
    )
    assert tagged["gates_passed"] is True
    assert tagged["model_ab"]["label"] == "cheap"
    assert tagged["model_ab"]["query_llm_model"] == "base-model"
    assert tagged["model_ab"]["run_id"] == "20260101T000000Z"
