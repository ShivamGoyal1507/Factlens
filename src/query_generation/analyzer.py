import json
import re


def _clean_text(value):
    """Normalize whitespace in a string."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _as_string_list(value):
    """Convert a JSON field into a clean list of strings."""
    if not isinstance(value, list):
        return []

    return [
        _clean_text(item)
        for item in value
        if isinstance(item, str) and _clean_text(item)
    ]


def _parse_analysis(response):
    """Parse the JSON object returned by the SLM."""
    if not isinstance(response, str):
        return None

    response = response.strip()

    # Extract JSON if the model adds surrounding text.
    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end <= start:
        return None

    try:
        data = json.loads(response[start:end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    expected_keys = [
        "subject",
        "actions",
        "objects",
        "entities",
        "keywords",
        "dates",
        "relationships",
        "components",
    ]

    result = {}

    for key in expected_keys:
        if key == "subject":
            result[key] = _clean_text(data.get(key, ""))
        else:
            result[key] = _as_string_list(data.get(key, []))

    return result


def _fallback_analysis(claim):
    """
    Conservative fallback when the model is unavailable
    or returns invalid JSON.

    It does not attempt to invent a decomposition.
    """
    claim = _clean_text(claim)

    years = re.findall(
        r"\b(?:19|20|21)\d{2}\b",
        claim
    )

    entities = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        claim
    )

    entities = list(dict.fromkeys(entities))

    action_pattern = (
        r"\b(?:launched|launch|acquired|acquire|announced|announce|"
        r"developed|develop|approved|approve|released|release|"
        r"created|create|signed|sign|won|built|tested|introduced)\b"
    )

    actions = re.findall(
        action_pattern,
        claim,
        flags=re.IGNORECASE
    )

    return {
    "subject": entities[0] if entities else "",
    "actions": list(dict.fromkeys(actions)),
    "objects": [],
    "entities": entities,
    "keywords": [],
    "dates": list(dict.fromkeys(years)),
    "relationships": [],
    "components": [claim] if claim else [],
    }


def _validate_analysis(data, original_claim):
    """Ensure that the analyzer returns a consistent structure."""
    if not isinstance(data, dict):
        return _fallback_analysis(original_claim)

    result = {
    "subject": _clean_text(data.get("subject", "")),
    "actions": _as_string_list(data.get("actions", [])),
    "objects": _as_string_list(data.get("objects", [])),
    "entities": _as_string_list(data.get("entities", [])),
    "keywords": _as_string_list(data.get("keywords", [])),
    "dates": _as_string_list(data.get("dates", [])),
    "relationships": _as_string_list(data.get("relationships", [])),
    "components": _as_string_list(data.get("components", [])),
    }

    # Do not allow an empty component list.
    if not result["components"]:
        result["components"] = [original_claim]

    # Remove duplicate components while preserving order.
    result["components"] = list(
        dict.fromkeys(result["components"])
    )

    # A component should not be completely unrelated in length.
    # Keep the original claim as a safe fallback if output is unusable.
    if not result["subject"] and not result["entities"]:
        result["subject"] = ""

    return result


def analyze_claim(claim):
    """
    Analyze a factual claim with the shared SLM (see slm_client.py).

    Returns:
        dict: subject, actions, objects, entities, keywords, dates,
              relationships and factual components.
    """
    if not isinstance(claim, str) or not claim.strip():
        return _fallback_analysis("")

    claim = _clean_text(claim)

    try:
        from src.query_generation import slm_client
        from src.query_generation.prompts import build_analysis_messages

        result = slm_client.generate_chat(
            build_analysis_messages(claim),
            max_new_tokens=256,
        )
        parsed = _parse_analysis(result["text"])

        if parsed is None:
            return _fallback_analysis(claim)

        return _validate_analysis(parsed, claim)

    except Exception as error:
        print(f"[CLAIM ANALYZER] Model analysis failed: {error}")
        return _fallback_analysis(claim)
