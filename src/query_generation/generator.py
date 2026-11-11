"""
FACTLENS query generation.

Pipeline:
    1. Estimate claim complexity -> maximum number of queries (1/3/5).
    2. Ask the SLM (Qwen2.5-1.5B-Instruct by default) for search queries.
    3. Validate, ground (no invented numbers/terms) and de-duplicate them.
    4. Top up with deterministic keyword queries if the model returned
       too few, or use them entirely if the model is unavailable.

Public entry point: generate_queries(claim) -> list[str]
"""

import re

from src.query_generation.complexity import estimate_claim_complexity, get_max_queries
from src.query_generation.validator import is_valid_query, remove_duplicate_queries

try:
    from src.evidence_retrieval.retriever import retrieve_evidence_for_claim
except Exception:  # pragma: no cover - optional dependency guard
    def retrieve_evidence_for_claim(claim, queries=None, claim_made_at=None):
        return []


MIN_QUERIES = 3  # target minimum for moderate/complex claims

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "has", "have", "had", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "this", "these", "those", "to", "was", "were", "with", "about",
    "into", "after", "before", "during", "under", "over", "between", "within",
    "through", "while", "which", "who", "whom", "will", "would", "can", "could",
    "their", "his", "her", "they", "them", "than", "then", "also", "very",
}


def _normalize_query(query):
    if not isinstance(query, str):
        return ""
    return " ".join(query.strip().split()).strip("\"'[](){}")


# ---------------------------------------------------------------------------
# Lightweight feature extraction (used by fallback and exposed publicly)
# ---------------------------------------------------------------------------

def _extract_years(claim):
    if not isinstance(claim, str):
        return []
    return list(dict.fromkeys(re.findall(r"\b(?:19|20|21)\d{2}\b", claim)))


def _extract_named_entities(claim):
    if not isinstance(claim, str):
        return []

    entities = re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b", claim)
    entities += [t for t in re.findall(r"\b[A-Z]{2,}\b", claim)]

    unique, seen = [], set()
    for entity in entities:
        key = entity.lower()
        if key in _STOPWORDS or key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique[:12]


def _extract_keywords(claim):
    tokens = re.findall(r"\$?[A-Za-z0-9][A-Za-z0-9\-.%]*", claim or "")
    keywords = []
    for token in tokens:
        token = token.rstrip(".")
        if token.lower() in _STOPWORDS or len(token) < 2:
            continue
        keywords.append(token)
    return list(dict.fromkeys(keywords))


def extract_claim_features(claim):
    if not isinstance(claim, str):
        return {"keywords": [], "entities": [], "dates": [], "events": [], "relationships": []}

    cleaned = _normalize_query(claim)
    events = re.findall(
        r"\b(?:launched|launch|announced|opened|signed|tested|developed|reported|"
        r"approved|built|released|created|introduced|acquired|won|attended|said)\b",
        cleaned,
        flags=re.IGNORECASE,
    )
    relationships = []
    if re.search(r"\b(first|only|largest|smallest|oldest|latest)\b", cleaned, re.I):
        relationships.append("superlative qualifier")
    if re.search(r"\b(not|never|no)\b", cleaned, re.I):
        relationships.append("negation")

    return {
        "keywords": _extract_keywords(cleaned)[:12],
        "entities": _extract_named_entities(cleaned),
        "dates": _extract_years(cleaned),
        "events": list(dict.fromkeys(e.lower() for e in events))[:8],
        "relationships": relationships,
    }


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def _near_duplicate(a, b, threshold=0.8):
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _filter_and_prune_queries(queries, limit=5):
    kept = []
    for item in queries:
        query = _normalize_query(item)
        if not query or not is_valid_query(query):
            continue
        if any(_near_duplicate(query, existing) for existing in kept):
            continue
        kept.append(query)
    return remove_duplicate_queries(kept)[:limit]


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _keyword_fallback_queries(claim, limit=5):
    """Keyword-style queries built only from words in the claim."""
    text = _normalize_query(claim)
    if not text:
        return []

    features = extract_claim_features(text)
    keywords = features["keywords"]
    entities = features["entities"]
    years = features["dates"]
    events = features["events"]

    candidates = []

    # 1. Compact keyword version of the full claim.
    if keywords:
        candidates.append(" ".join(keywords[:8]))

    # 2. Main entity + action + year.
    if entities:
        parts = [entities[0]] + events[:1] + years[:1]
        candidates.append(" ".join(parts))

    # 3. Remaining entities with the year (e.g. partner organisations).
    for entity in entities[1:3]:
        candidates.append(" ".join([entity] + keywords[:3] + years[:1]))

    # 4. The claim itself as a last resort.
    candidates.append(text.rstrip(".!?"))

    return _filter_and_prune_queries(candidates, limit=limit)


# ---------------------------------------------------------------------------
# Model-based generation
# ---------------------------------------------------------------------------

def generate_queries_with_llm(claim, max_queries=5):
    """Generate queries with the SLM only. Returns [] on any failure."""
    if not isinstance(claim, str) or not claim.strip():
        return []
    try:
        from src.query_generation.subquery_generator import generate_subqueries

        return generate_subqueries(_normalize_query(claim), max_queries=max_queries)
    except Exception as error:
        print(f"[QUERY GENERATION] SLM generation failed: {error}")
        return []


# Backwards-compatible alias (the SmolLM2 client was replaced).
generate_queries_with_smol = generate_queries_with_llm


def generate_queries(claim, claim_made_at=None):
    """
    Generate search queries for a claim.

    Returns a list of 1-5 distinct query strings.
    `claim_made_at` is accepted for API compatibility; date filtering
    happens in evidence retrieval.
    """
    if not isinstance(claim, str) or not claim.strip():
        return []

    claim = _normalize_query(claim)
    complexity = estimate_claim_complexity(claim)
    max_queries = get_max_queries(complexity)
    target = min(MIN_QUERIES, max_queries)

    queries = generate_queries_with_llm(claim, max_queries=max_queries)

    if len(queries) < target:
        if queries:
            print("[QUERY GENERATION] Topping up with keyword fallback queries.")
        else:
            print("[QUERY GENERATION] Using keyword fallback queries.")
        queries = queries + _keyword_fallback_queries(claim)

    queries = _filter_and_prune_queries(queries, limit=max_queries)
    return queries or [claim]


def generate_queries_with_metadata(claim, claim_made_at=None):
    queries = generate_queries(claim, claim_made_at=claim_made_at)
    return [{"query": query, "claim_made_at": claim_made_at} for query in queries]


def generate_evidence_for_claim(claim, claim_made_at=None):
    if not isinstance(claim, str) or not claim.strip():
        return []

    queries = generate_queries(claim, claim_made_at=claim_made_at)
    if not queries:
        return []

    return retrieve_evidence_for_claim(claim, queries=queries, claim_made_at=claim_made_at)
