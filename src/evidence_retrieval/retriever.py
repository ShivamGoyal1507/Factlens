from src.evidence_retrieval.sources.wikipedia import search_wikipedia
from src.evidence_retrieval.sources.gdelt import search_gdelt
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
        print(f"{source_name}: Retrieved {len(normalized_results)} articles")
        return normalized_results

    except Exception as error:
        print(f"{source_name} retrieval failed: {error}")
        return []


def filter_by_claim_date(documents, query=None):
    """
    Keep only documents that were published on or before the claim cutoff.
    If no valid claim-date constraint is present, keep the original documents.
    """
    if not isinstance(documents, list):
        return []

    if not query:
        return list(documents)

    cutoff = get_temporal_cutoff_for_claim(query)
    if cutoff is None:
        return list(documents)

    filtered = []
    for doc in documents:
        if document_is_temporally_eligible(doc, cutoff):
            filtered.append(doc)

    return filtered


def retrieve_evidence(query):
    """
    Retrieve evidence from Wikipedia, GDELT and NewsAPI.

    Args:
        query (str): Search query.

    Returns:
        list: Combined evidence documents.
    """
    if not isinstance(query, str) or not query.strip():
        print("Invalid query. Please provide a non-empty search query.")
        return []

    wikipedia_results = safe_retrieve("Wikipedia", search_wikipedia, query)
    gdelt_results = safe_retrieve("GDELT", search_gdelt, query)
    newsapi_results = safe_retrieve("NewsAPI", search_newsapi, query)

    all_results = wikipedia_results + gdelt_results + newsapi_results
    valid_results = validate_documents(all_results)
    unique_results = remove_duplicates(valid_results)

    cutoff = get_temporal_cutoff_for_claim(query)
    if cutoff is not None:
        precision = get_claim_date_precision(query)
        filtered = []
        for doc in unique_results:
            if document_is_temporally_eligible(doc, cutoff):
                filtered_doc = dict(doc)
                filtered_doc["temporal_precision"] = precision
                filtered.append(filtered_doc)
        unique_results = filtered
        print("Claim-date filtering active:", cutoff.isoformat(), "precision=", precision)

    print("\n===== EVIDENCE RETRIEVAL SUMMARY =====")
    print("Wikipedia:", len(wikipedia_results))
    print("GDELT:", len(gdelt_results))
    print("NewsAPI:", len(newsapi_results))
    print("Total before deduplication:", len(all_results))
    print("Total after deduplication:", len(unique_results))
    print("Invalid documents removed:", len(all_results) - len(valid_results))
    print("Duplicates removed:", len(valid_results) - len(unique_results))

    return unique_results


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
        if not isinstance(text, str) or not text.strip():
            continue

        title = doc.get("title")
        url = doc.get("url")
        source = doc.get("source")

        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(url, str) or not url.strip():
            continue
        if not isinstance(source, str) or not source.strip():
            continue

        prepared = dict(doc)
        prepared["published_date"] = doc.get("date")
        ranking_ready.append(prepared)

    return ranking_ready