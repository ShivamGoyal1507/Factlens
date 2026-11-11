
import pytest

from src.query_generation.complexity import (
    estimate_claim_complexity,
    get_max_queries,
)
from src.query_generation.validator import (
    is_valid_query,
    is_grounded_query,
)


def test_simple_claim_has_one_query_limit():
    claim = "The Earth revolves around the Sun."

    complexity = estimate_claim_complexity(claim)

    assert complexity == "simple"
    assert get_max_queries(complexity) == 1


def test_complex_claim_has_five_query_limit():
    claim = (
        "India launched its first AI-powered agricultural satellite "
        "in 2025, developed by ISRO in collaboration with a private company."
    )

    analysis = {
        "entities": ["India", "ISRO"],
        "keywords": ["AI-powered", "agricultural satellite"],
        "actions": ["launched", "developed"],
        "dates": ["2025"],
        "relationships": [
            "ISRO collaborated with a private company"
        ],
        "components": [
            "India launched its first AI-powered agricultural satellite in 2025.",
            "ISRO developed the satellite.",
            "ISRO collaborated with a private company."
        ],
    }

    complexity = estimate_claim_complexity(claim, analysis)

    assert complexity == "complex"
    assert get_max_queries(complexity) == 5


def test_validator_rejects_prompt_fragments():
    assert not is_valid_query("Return JSON only")
    assert not is_valid_query("Original claim:")


def test_validator_accepts_search_query():
    assert is_valid_query(
        "India agricultural satellite launch 2025"
    )


def test_grounding_rejects_invented_year():
    claim = "India launched a satellite in 2025."

    query = "India satellite launch in 2026"

    assert not is_grounded_query(query, claim)


def test_grounding_accepts_supported_year():
    claim = "India launched a satellite in 2025."

    query = "India satellite launch 2025"

    assert is_grounded_query(query, claim)


def test_grounding_rejects_unrelated_query():
    claim = "India launched a satellite in 2025."

    query = "Brazil football world cup"

    assert not is_grounded_query(query, claim)
