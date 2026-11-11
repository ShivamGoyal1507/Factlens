
from src.query_generation.analyzer import (
    _parse_analysis,
    _fallback_analysis,
    _validate_analysis,
    analyze_claim,
)


def test_parse_valid_json():
    response = """
    {
        "subject": "Apple",
        "actions": ["acquired"],
        "objects": ["a startup"],
        "entities": ["Apple"],
        "dates": ["2025"],
        "relationships": [],
        "components": ["Apple acquired a startup in 2025."]
    }
    """

    result = _parse_analysis(response)

    assert result is not None
    assert result["subject"] == "Apple"
    assert result["actions"] == ["acquired"]
    assert result["dates"] == ["2025"]


def test_parse_invalid_json():
    result = _parse_analysis("This is not JSON")

    assert result is None


def test_fallback_preserves_original_claim():
    claim = "The Earth revolves around the Sun."

    result = _fallback_analysis(claim)

    assert result["components"] == [claim]


def test_validate_removes_duplicate_components():
    claim = "Apple acquired a startup."

    data = {
        "subject": "Apple",
        "actions": ["acquired"],
        "objects": ["a startup"],
        "entities": ["Apple"],
        "dates": [],
        "relationships": [],
        "components": [
            claim,
            claim,
        ],
    }

    result = _validate_analysis(data, claim)

    assert result["components"] == [claim]


def test_empty_claim():
    result = analyze_claim("")

    assert result["components"] == []


def test_fallback_returns_expected_structure():
    result = _fallback_analysis(
        "Apple acquired a startup in 2025."
    )

    expected_keys = {
        "subject",
        "actions",
        "objects",
        "entities",
        "keywords",
        "dates",
        "relationships",
        "components",
    }

    assert set(result.keys()) == expected_keys
    assert "2025" in result["dates"]
