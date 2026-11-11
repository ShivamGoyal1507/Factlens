import json
import os

import pytest

import src.query_generation.generator as generator
from src.query_generation import slm_client
from src.query_generation.subquery_generator import (
    filter_queries,
    parse_entities,
    parse_queries,
)

CLAIM = "India launched its first AI-powered agricultural satellite in 2025."


def _fake_model(queries=None, raw=None, calls=None):
    """Replace the SLM with a canned response (no model download)."""
    def fake_generate_chat(messages, max_new_tokens=192):
        if calls is not None:
            calls.append(messages)
        text = raw if raw is not None else json.dumps({"queries": queries})
        return {"text": text}
    return fake_generate_chat


@pytest.fixture
def good_model(monkeypatch):
    calls = []
    monkeypatch.setattr(slm_client, "generate_chat", _fake_model([
        "India first AI-powered agricultural satellite 2025",
        "ISRO AI agricultural satellite launch",
        "first AI-powered satellite India",
    ], calls=calls))
    return calls


def test_generate_queries_uses_slm_output(good_model):
    queries = generator.generate_queries(CLAIM)

    assert len(good_model) == 1
    assert queries[0] == "India first AI-powered agricultural satellite 2025"
    assert 1 <= len(queries) <= 5


def test_prompt_contains_claim_and_limit(good_model):
    generator.generate_queries(CLAIM)
    last_user = good_model[0][-1]["content"]

    assert CLAIM in last_user
    assert "Maximum queries: 3" in last_user


def test_feature_descriptions_are_rejected():
    invalid = [
        "Key events/actions: launched",
        "Important relationships: first occurrence or first launch; AI or artificial intelligence involved",
        "Named entities: India",
        "Search for subqueries related to India",
    ]

    assert generator._filter_and_prune_queries(invalid) == []


def test_invented_years_are_filtered(monkeypatch):
    monkeypatch.setattr(slm_client, "generate_chat", _fake_model([
        "India satellite launch 2026",
        "India AI agricultural satellite 2025",
    ]))

    queries = generator.generate_queries(CLAIM)

    assert all("2026" not in q for q in queries)
    assert "India AI agricultural satellite 2025" in queries


def test_unrelated_queries_are_filtered():
    kept = filter_queries(
        ["Brazil football world cup", "India agricultural satellite 2025"],
        CLAIM,
    )
    assert kept == ["India agricultural satellite 2025"]


def test_near_duplicate_queries_are_removed():
    kept = filter_queries(
        [
            "India AI agricultural satellite 2025",
            "india ai agricultural satellite 2025",
            "India agricultural satellite AI 2025",
            "ISRO satellite launch",
        ],
        CLAIM + " ISRO",
    )
    assert kept == ["India AI agricultural satellite 2025", "ISRO satellite launch"]


def test_top_up_with_fallback_when_model_returns_too_few(monkeypatch):
    monkeypatch.setattr(slm_client, "generate_chat", _fake_model([
        "India AI agricultural satellite 2025",
    ]))

    queries = generator.generate_queries(CLAIM)

    assert queries[0] == "India AI agricultural satellite 2025"
    assert 2 <= len(queries) <= 3


def test_fallback_when_model_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(slm_client, "generate_chat", broken)

    queries = generator.generate_queries(CLAIM)

    assert queries
    assert any("2025" in q for q in queries)
    assert any("India" in q for q in queries)


def test_fallback_when_model_returns_garbage(monkeypatch):
    monkeypatch.setattr(
        slm_client, "generate_chat", _fake_model(raw="Sure! Here are the queries:")
    )
    queries = generator.generate_queries(CLAIM)
    assert queries
    assert all("here are" not in q.lower() for q in queries)


def test_parse_queries_formats():
    assert parse_queries('{"queries": ["a b", "c d"]}') == ["a b", "c d"]
    assert parse_queries('```json\n{"queries": ["a b"]}\n```') == ["a b"]
    assert parse_queries('["x y", "z w"]') == ["x y", "z w"]
    assert parse_queries("1. first query\n2. second query") == [
        "first query",
        "second query",
    ]


def test_parse_entities_fixes_case_but_rejects_invented_names():
    response = json.dumps({
        "entities": ["Rahul Gandhi", "India", "Narendra Modi"],
        "queries": ["Rahul Gandhi"],
    })
    claim = "rahul gandhi is prime minister of india"

    assert parse_entities(response, claim) == ["Rahul Gandhi", "India"]


def test_search_plan_adds_entity_query(monkeypatch):
    monkeypatch.setattr(slm_client, "generate_chat", _fake_model(raw=json.dumps({
        "entities": ["Rahul Gandhi", "India"],
        "queries": [
            "Who is the Prime Minister of India",
            "Prime Minister of India Rahul Gandhi",
        ],
    })))

    plan = generator.generate_search_plan("rahul gandhi is prime minister of india")

    assert plan["entities"] == ["Rahul Gandhi", "India"]
    assert plan["queries"][:2] == [
        "Who is the Prime Minister of India",
        "Prime Minister of India Rahul Gandhi",
    ]
    assert "Rahul Gandhi" in plan["queries"]
    assert "India" not in plan["queries"]  # one-word entity skipped


def test_empty_claim_returns_no_queries():
    assert generator.generate_queries("") == []
    assert generator.generate_queries(None) == []


def test_important_dates_and_entities_are_extracted():
    features = generator.extract_claim_features(CLAIM)
    assert "2025" in features["dates"]
    assert "India" in features["entities"]


@pytest.mark.skipif(
    os.environ.get("FACTLENS_RUN_SLM") != "1",
    reason="set FACTLENS_RUN_SLM=1 to run the real Hugging Face model",
)
def test_real_slm_generates_grounded_queries():
    queries = generator.generate_queries_with_llm(
        "India launched its first AI-powered agricultural satellite in 2025, "
        "developed by ISRO in collaboration with a private company.",
        max_queries=5,
    )

    assert 3 <= len(queries) <= 5
    assert any("ISRO" in q for q in queries)
    assert all("2026" not in q for q in queries)
