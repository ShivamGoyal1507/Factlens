"""Chat prompts used by the SLM-based query generation stage."""

import json


QUERY_SYSTEM_PROMPT = """You write web search queries for a fact-checking system.
Given a claim, write short keyword-style search queries (like what a journalist would type into Google or a news search) that would find evidence to verify or refute it.
Rules:
- Each query checks a different factual part of the claim (who, what, when, where, numbers, "first/only" superlatives).
- Use only names, dates, numbers and terms that appear in the claim. Never invent facts or answer the claim.
- Keep negations and qualifiers that change meaning.
- 3 to 10 words per query. No explanations.
Return JSON only: {"queries": ["...", "..."]}"""


# Few-shot examples steer small models toward the exact output format.
QUERY_FEW_SHOT = [
    (
        "Tesla acquired Twitter for $44 billion in 2022.",
        3,
        [
            "Twitter acquisition $44 billion 2022",
            "who acquired Twitter in 2022",
            "Tesla Twitter acquisition",
        ],
    ),
    (
        "The Eiffel Tower is in Paris.",
        1,
        ["Eiffel Tower location Paris"],
    ),
]


def _query_user_message(claim, max_queries):
    return f"Claim: {claim}\nMaximum queries: {max_queries}"


def build_query_messages(claim, max_queries):
    """Build chat messages asking the SLM for up to `max_queries` queries."""
    messages = [{"role": "system", "content": QUERY_SYSTEM_PROMPT}]

    for example_claim, example_max, example_queries in QUERY_FEW_SHOT:
        messages.append(
            {"role": "user", "content": _query_user_message(example_claim, example_max)}
        )
        messages.append(
            {"role": "assistant", "content": json.dumps({"queries": example_queries})}
        )

    messages.append({"role": "user", "content": _query_user_message(claim, max_queries)})
    return messages


CLAIM_ANALYSIS_SYSTEM_PROMPT = """You analyse factual claims for a fact-checking system.
Extract only information explicitly stated in the claim. Do not add background facts.
Return exactly one JSON object with these fields:
{"subject": "", "actions": [], "objects": [], "entities": [], "keywords": [], "dates": [], "relationships": [], "components": []}
- entities: named people, organisations, places and products.
- dates: exactly as written in the claim.
- components: the claim split into independent factual assertions.
Use empty arrays when nothing applies. Return JSON only."""


def build_analysis_messages(claim):
    return [
        {"role": "system", "content": CLAIM_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Claim: {claim}"},
    ]


__all__ = [
    "QUERY_SYSTEM_PROMPT",
    "QUERY_FEW_SHOT",
    "CLAIM_ANALYSIS_SYSTEM_PROMPT",
    "build_query_messages",
    "build_analysis_messages",
]
