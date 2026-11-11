
import re


def _clean_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _extract_named_entities(claim):
    patterns = [
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
        r"\b(?:ISRO|NASA|WHO|NATO|UN|EU)\b",
    ]

    entities = []

    for pattern in patterns:
        entities.extend(re.findall(pattern, claim))

    return list(dict.fromkeys(entities))


def claim_complexity_score(claim, analysis=None):
    text = _clean_text(claim)

    if not text:
        return 0

    lower = text.lower()
    score = 0

    # Multiple clauses or independent assertions.
    if re.search(
        r"\b(and|but|while|although|whereas|because|after|before)\b",
        lower,
    ):
        score += 1

    # Named entities.
    score += min(3, len(_extract_named_entities(text)))

    # Explicit dates.
    dates = re.findall(r"\b(?:19|20|21)\d{2}\b", text)
    score += min(2, len(set(dates)))

    # Actions.
    actions = re.findall(
        r"\b(launched|announced|developed|approved|signed|"
        r"tested|built|released|created|acquired|won|"
        r"introduced|collaborated|partnered)\b",
        lower,
    )
    score += min(2, len(set(actions)))

    # Important qualifiers.
    if re.search(
        r"\b(first|only|largest|smallest|oldest|latest)\b",
        lower,
    ):
        score += 1

    # Structured analysis.
    if isinstance(analysis, dict):
        components = analysis.get("components", [])
        relationships = analysis.get("relationships", [])

        if isinstance(components, list):
            score += min(3, max(0, len(components) - 1))

        if isinstance(relationships, list) and relationships:
            score += 1

    return score


def estimate_claim_complexity(claim, analysis=None):
    score = claim_complexity_score(claim, analysis)

    if score <= 2:
        return "simple"

    if score <= 5:
        return "moderate"

    return "complex"


def get_max_queries(complexity):
    limits = {
        "simple": 1,
        "moderate": 3,
        "complex": 5,
    }

    return limits.get(complexity, 1)


def is_complex_claim(claim):
    return estimate_claim_complexity(claim) in {
        "moderate",
        "complex",
    }


__all__ = [
    "claim_complexity_score",
    "estimate_claim_complexity",
    "get_max_queries",
    "is_complex_claim",
]
