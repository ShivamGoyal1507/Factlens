import re


INVALID_QUERY_PATTERNS = (
    "key events/actions",
    "important relationships",
    "named entities",
    "important keywords",
    "original claim",
    "claim content dates",
    "search for subqueries",
    "output only a json array",
    "requirements:",
    "not specified",
    "here are",
    "subqueries:",
    "important events",
    "important information",
    "output:",
    "key points",
    "evidence request",
)


def _normalize_text(value):
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def _looks_like_prompt_fragment(query):
    lower = _normalize_text(query).lower()
    if any(pattern in lower for pattern in INVALID_QUERY_PATTERNS):
        return True
    if re.match(r"^(?:key|important|named|original|search|output|evidence|requirements)\b", lower):
        return True
    if lower.startswith(("search for ", "subqueries:", "here are", "answer:", "query:")):
        return True
    if re.match(r"^(?:name|description|keywords|entities|dates|relationships|components)\s*:\s*", lower):
        return True
    if re.search(r'^[A-Za-z_]+"?\s*:\s*"?', lower):
        return True
    return False



def is_valid_query(candidate):
    query = _normalize_text(candidate)

    if not query or len(query) < 5:
        return False

    lower = query.lower()

    # Reject prompt fragments and model formatting errors.
    if _looks_like_prompt_fragment(lower):
        return False

    if lower.endswith(":"):
        return False

    # Reject strings containing only punctuation or symbols.
    if not re.search(r"[A-Za-z0-9]", query):
        return False

    # Reject very short, incomplete fragments.
    if len(re.findall(r"[A-Za-z0-9]+", query)) < 2:
        return False

    return True

def is_grounded_query(query, claim, analysis=None, min_overlap=0.15):
    """
    Check whether a generated query has sufficient lexical overlap
    with the original claim or its extracted analysis.
    """
    if not isinstance(query, str) or not isinstance(claim, str):
        return False

    query_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", query.lower()))
    claim_tokens = set(re.findall(r"\b[a-zA-Z0-9]+\b", claim.lower()))

    if not query_tokens or not claim_tokens:
        return False

    allowed_tokens = set(claim_tokens)

    if isinstance(analysis, dict):
        for key in ("keywords", "entities", "subject", "actions", "objects"):
            value = analysis.get(key, [])

            if isinstance(value, str):
                value = [value]

            if isinstance(value, list):
                for item in value:
                    allowed_tokens.update(
                        re.findall(r"\b[a-zA-Z0-9]+\b", str(item).lower())
                    )

    overlap = query_tokens.intersection(allowed_tokens)

    return len(overlap) / len(query_tokens) >= min_overlap

def remove_duplicate_queries(queries):
    seen = set()
    results = []
    for item in queries:
        query = _normalize_text(item)
        if not query or not is_valid_query(query):
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(query)
    return results


def reject_prompt_outputs(payload):
    if not isinstance(payload, list):
        return []
    return remove_duplicate_queries(payload)


__all__ = [
    "INVALID_QUERY_PATTERNS",
    "is_valid_query",
    "is_grounded_query",
    "remove_duplicate_queries",
    "reject_prompt_outputs",
]
