
CLAIM_ANALYSIS_PROMPT = """
Task: Analyze the factual claim and extract only information
explicitly stated in it.

Return exactly one JSON object with these fields:

{{
    "subject": "main subject",
    "actions": [],
    "objects": [],
    "entities": [],
    "keywords": [],
    "dates": [],
    "relationships": [],
    "components": []
}}

Rules:
1. Extract every action stated in the claim.
2. Extract named people, organisations, places and products.
3. Do not treat generic descriptions as named entities.
4. Preserve important technical terms and qualifiers.
5. Preserve negation and the meaning of relationships.
6. Dates must remain exactly as stated.
7. Split independent factual assertions into separate components.
8. Do not add capabilities, outcomes, explanations or background facts.
9. Use empty arrays when no information is available.
10. Return valid JSON only. No markdown or explanations.

Claim:
{claim}

JSON:
"""


QUERY_GENERATION_PROMPT = """
Task: Generate search queries for a fact-checking system.

You will receive:
1. The original factual claim.
2. A structured analysis of the claim.
3. Its complexity classification.

Generate only the search queries necessary to investigate
the factual components.

Rules:
1. Every query must investigate an aspect of the original claim.
2. Use the supplied entities, keywords, dates, actions and relationships.
3. Preserve the original meaning, including negation and qualifiers.
4. Do not invent names, dates, events, capabilities or outcomes.
5. Do not introduce outside facts.
6. Each query must investigate a distinct factual aspect.
7. Do not produce multiple queries that investigate the same aspect.
8. Do not generate queries merely to reach the maximum count.
9. Queries must be concise, grammatical and suitable for a search engine.
10. Use natural search-engine phrasing rather than awkward keyword fragments.
11. Prefer specific questions when they help investigate a factual component.
12. Return valid JSON only, using exactly this structure:

{{"queries": ["query one", "query two"]}}

Original claim:
{claim}

Claim analysis:
{analysis}

Complexity:
{complexity}

Maximum number of queries:
{max_queries}

JSON:
"""
