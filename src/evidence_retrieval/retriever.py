import os

from src.evidence_retrieval.sources.wikipedia import (
    search_wikipedia,
    search_wikipedia_for_claim,
)
from src.evidence_retrieval.sources.newsapi import search_newsapi
from src.evidence_retrieval.utils.normalizer import normalize_documents
from src.evidence_retrieval.utils.deduplication import remove_duplicates as _remove_duplicates
from src.evidence_retrieval.config import get_default_claim_timezone
from src.evidence_retrieval.utils.date_filter import (
    document_is_temporally_eligible,
    get_claim_date_precision,
    get_temporal_cutoff_for_claim,
)


def safe_retrieve(source_name, search_function, query):
    """
    Retrieve evidence from a source without stopping
    the entire pipeline if that source fails.
    """
    try:
        results = search_function(query)
        if not isinstance(results, list):
            results = []

        normalized_results = normalize_documents(results, source_hint=source_name)
        if not QUIET:
            print(f"{source_name}: Retrieved {len(normalized_results)} articles")
        return normalized_results

    except Exception as error:
        print(f"{source_name} retrieval failed: {error}")
        return []

def _resolve_claim_cutoff_source(claim_made_at=None):
    """Return an explicit claim-made cutoff only when the caller provides it."""
    if isinstance(claim_made_at, str) and claim_made_at.strip():
        return claim_made_at.strip()

    if claim_made_at is not None:
        return str(claim_made_at)

    return None


def filter_by_claim_date(documents, query=None, claim_made_at=None):
    """
    Keep only documents that were published on or before the claim cutoff.
    Temporal filtering is enabled only when the caller explicitly provides
    claim_made_at. A search query alone never activates filtering.
    """
    if not isinstance(documents, list):
        return []

    cutoff_source = _resolve_claim_cutoff_source(claim_made_at=claim_made_at)
    if not cutoff_source:
        return list(documents)

    cutoff = get_temporal_cutoff_for_claim(cutoff_source)
    if cutoff is None:
        return list(documents)

    precision = get_claim_date_precision(cutoff_source)
    filtered = []
    for doc in documents:
        if document_is_temporally_eligible(doc, cutoff, claim_precision=precision):
            filtered.append(doc)

    return filtered


def retrieve_evidence(query, claim_made_at=None):
    """
    Retrieve evidence from Wikipedia and NewsAPI.

    Args:
        query (str): Search query.
        claim_made_at (str | None): Explicit claim-made date used to activate
            temporal filtering. Search-query dates alone never trigger filtering.

    Returns:
        list: Combined evidence documents.
    """
    if not isinstance(query, str) or not query.strip():
        print("Invalid query. Please provide a non-empty search query.")
        return []

    wikipedia_results = safe_retrieve("Wikipedia", search_wikipedia, query)
    newsapi_results = (
        safe_retrieve("NewsAPI", search_newsapi, query) if NEWS_ENABLED else []
    )

    all_results = wikipedia_results + newsapi_results
    valid_results = validate_documents(all_results)
    unique_results = remove_duplicates(valid_results)

    cutoff_source = claim_made_at if isinstance(claim_made_at, str) and claim_made_at.strip() else None
    if cutoff_source is None and claim_made_at is not None:
        cutoff_source = str(claim_made_at)

    cutoff = None
    precision = None
    if cutoff_source is not None:
        cutoff = get_temporal_cutoff_for_claim(cutoff_source)
        precision = get_claim_date_precision(cutoff_source) if cutoff is not None else None

    if cutoff is not None:
        filtered = []
        for doc in unique_results:
            if document_is_temporally_eligible(doc, cutoff, claim_precision=precision):
                filtered_doc = dict(doc)
                filtered_doc["temporal_precision"] = precision
                filtered.append(filtered_doc)
        unique_results = filtered
        print("Claim-date filtering active:", cutoff.isoformat(), "precision=", precision)

    print("\n===== EVIDENCE RETRIEVAL SUMMARY =====")
    print("Wikipedia:", len(wikipedia_results))
    print("NewsAPI:", len(newsapi_results))
    print("Total before deduplication:", len(all_results))
    print("Total after deduplication:", len(unique_results))
    print("Invalid documents removed:", len(all_results) - len(valid_results))
    print("Duplicates removed:", len(valid_results) - len(unique_results))

    return unique_results


# FACTLENS_QUIET=1 silences the per-source success lines (batch evaluation
# with several threads); failures are still printed.
QUIET = os.environ.get("FACTLENS_QUIET", "").strip().lower() in {"1", "true", "yes"}

# Set FACTLENS_DISABLE_NEWS=1 to use Wikipedia only (e.g. for FEVER benchmarks,
# whose claims are about encyclopedic facts).
NEWS_ENABLED = os.environ.get("FACTLENS_DISABLE_NEWS", "").strip().lower() not in {
    "1", "true", "yes",
}


def _retrieve_evidence_for_single_query(query):
    """Internal helper for source retrieval, validation, and deduplication."""
    if not isinstance(query, str) or not query.strip():
        print("Invalid query. Please provide a non-empty search query.")
        return []

    wikipedia_results = safe_retrieve("Wikipedia", search_wikipedia, query)
    newsapi_results = (
        safe_retrieve("NewsAPI", search_newsapi, query) if NEWS_ENABLED else []
    )

    all_results = wikipedia_results + newsapi_results
    valid_results = validate_documents(all_results)
    unique_results = remove_duplicates(valid_results)

    return unique_results


def retrieve_evidence_for_claim(claim, queries=None, claim_made_at=None, entities=None):
    """
    Retrieve evidence for a claim using multiple generated subqueries while
    preserving the original claim-made date as the sole temporal cutoff.

    Wikipedia pages the claim is about (title mentioned in the claim) are
    retrieved first from the claim itself, independent of the generated
    queries; that is where FEVER-style gold evidence lives.
    """
    if not isinstance(claim, str) or not claim.strip():
        return []

    if queries is None:
        from src.query_generation.generator import generate_queries
        queries = generate_queries(claim)

    if not isinstance(queries, list) or not queries:
        return []

    all_documents = validate_documents(safe_retrieve(
        "Wikipedia (claim pages)",
        lambda _query: search_wikipedia_for_claim(claim, entities=entities),
        claim,
    ))
    for query in queries:
        all_documents.extend(_retrieve_evidence_for_single_query(query))

    deduplicated = remove_duplicates(all_documents)

    cutoff = None
    cutoff_precision = None
    cutoff_source = None

    if isinstance(claim_made_at, str) and claim_made_at.strip():
        cutoff_source = claim_made_at.strip()
    elif claim_made_at is not None:
        cutoff_source = str(claim_made_at)

    if cutoff_source is not None:
        cutoff = get_temporal_cutoff_for_claim(cutoff_source)
        cutoff_precision = get_claim_date_precision(cutoff_source) if cutoff is not None else None

    if cutoff is not None:
        filtered = []
        for doc in deduplicated:
            if document_is_temporally_eligible(doc, cutoff, claim_precision=cutoff_precision):
                filtered_doc = dict(doc)
                filtered_doc["temporal_precision"] = cutoff_precision
                filtered.append(filtered_doc)
        deduplicated = filtered

    return deduplicated


def remove_duplicates(documents):
    """
    Backward-compatible wrapper for tests/imports.
    """
    return _remove_duplicates(documents)


def validate_documents(documents):
    """
    Validate retrieved evidence documents.

    Required:
    - Non-empty title
    - Non-empty URL
    - Non-empty source
    - Text must be a string (empty text is allowed)
    - date and seen_date must be None or non-empty strings
    """
    valid_documents = []

    for doc in documents:
        if not isinstance(doc, dict):
            continue

        title = doc.get("title")
        url = doc.get("url")
        source = doc.get("source")
        date = doc.get("date")
        seen_date = doc.get("seen_date")
        text = doc.get("text", "")

        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(source, str) or not source.strip():
            continue

        if date is not None and (not isinstance(date, str) or not date.strip()):
            continue
        if seen_date is not None and (not isinstance(seen_date, str) or not seen_date.strip()):
            continue

        if text is None:
            text = ""
        if not isinstance(text, str):
            continue

        valid_documents.append({
            "title": title.strip(),
            "url": url.strip(),
            "source": source.strip(),
            "date": date.strip() if isinstance(date, str) else None,
            "seen_date": seen_date.strip() if isinstance(seen_date, str) else None,
            "text": text,
        })

    return valid_documents


def get_text_ready_documents(documents):
    """
    Return only documents that contain usable text
    for the ranking module.
    """
    text_ready_documents = []

    for doc in documents:
        text = doc.get("text", "")
        if isinstance(text, str) and text.strip():
            text_ready_documents.append(doc)

    return text_ready_documents


def prepare_for_ranking(documents):
    """
    Prepare retrieved documents for the downstream ranking pipeline.

    This adapter intentionally keeps the original retrieval output unchanged and
    returns a new list of copies. It filters out documents without usable text,
    adds a `published_date` alias for PassageSplitter, and preserves the original
    `date` and `seen_date` fields.
    """
    if not isinstance(documents, list):
        return []

    ranking_ready = []

    for doc in documents:
        if not isinstance(doc, dict):
            continue

        text = doc.get("text", "")
        title = doc.get("title")

        if not isinstance(text, str) or not text.strip():
            continue

        url = doc.get("url")
        source = doc.get("source")

        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(source, str) or not source.strip():
            continue

        prepared = dict(doc)
        prepared["text"] = text
        prepared["published_date"] = doc.get("date")
        ranking_ready.append(prepared)

    return ranking_ready