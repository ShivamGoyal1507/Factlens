"""SLM-based search-query generation (Qwen2.5-Instruct by default)."""

import json
import re

from src.query_generation import slm_client
from src.query_generation.prompts import build_query_messages
from src.query_generation.validator import (
    is_grounded_query,
    is_valid_query,
    remove_duplicate_queries,
)


def _clean(text):
    return " ".join(str(text).strip().split()).strip("\"'`")


def parse_queries(response):
    """
    Extract a list of query strings from the model response.

    Accepts {"queries": [...]}, a bare JSON list, or (as a last resort)
    one query per line.
    """
    if isinstance(response, dict):
        response = response.get("text", "")
    if not isinstance(response, str) or not response.strip():
        return []

    text = response.strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)

    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start == -1 or end <= start:
            continue
        try:
            data = json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            data = data.get("queries")
        if isinstance(data, list):
            return [_clean(q) for q in data if isinstance(q, str) and q.strip()]

    # Fallback: one query per line (numbered or bulleted lists).
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
        line = _clean(line).rstrip(",")
        if line and not line.endswith(":") and "{" not in line and "}" not in line:
            lines.append(line)
    return lines


def _tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _near_duplicate(a, b, threshold=0.8):
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def filter_queries(candidates, claim, analysis=None, max_queries=5):
    """Validate, ground and de-duplicate model output."""
    kept = []
    for query in remove_duplicate_queries(candidates):
        if not is_valid_query(query):
            continue
        if not is_grounded_query(query, claim, analysis):
            continue
        if any(_near_duplicate(query, existing) for existing in kept):
            continue
        kept.append(query)
        if len(kept) >= max_queries:
            break
    return kept


def generate_subqueries(claim, max_queries=5, analysis=None, max_new_tokens=192):
    """
    Generate up to `max_queries` grounded search queries for `claim`.

    Raises on model-loading/generation errors so the caller can fall back.
    """
    if not isinstance(claim, str) or not claim.strip():
        return []
    if max_queries < 1:
        return []

    messages = build_query_messages(claim, max_queries)
    result = slm_client.generate_chat(messages, max_new_tokens=max_new_tokens)
    candidates = parse_queries(result)

    return filter_queries(candidates, claim, analysis, max_queries)


__all__ = ["parse_queries", "filter_queries", "generate_subqueries"]
