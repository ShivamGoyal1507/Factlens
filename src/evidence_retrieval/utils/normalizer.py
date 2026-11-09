from src.evidence_retrieval.utils.date_filter import (
    normalize_publication_date,
    normalize_seen_date,
)


def _clean_string(value, default=""):
    if isinstance(value, str):
        return value.strip()
    return default


def normalize_document(raw_document, source_hint=None):
    """
    Normalize a raw document into the shared schema:
    title, url, source, date, seen_date, text
    """
    if not isinstance(raw_document, dict):
        return None

    source = _clean_string(raw_document.get("source"), "")
    if not source and source_hint:
        source = _clean_string(source_hint, "")

    text = raw_document.get("text", "")
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = ""

    normalized = {
        "title": _clean_string(raw_document.get("title"), ""),
        "url": _clean_string(raw_document.get("url"), ""),
        "source": source,
        "date": normalize_publication_date(raw_document.get("date")),
        "seen_date": normalize_seen_date(raw_document.get("seen_date")),
        "text": text,
    }

    return normalized


def normalize_documents(documents, source_hint=None):
    normalized_documents = []

    if not isinstance(documents, list):
        return normalized_documents

    for doc in documents:
        normalized = normalize_document(doc, source_hint=source_hint)
        if normalized is not None:
            normalized_documents.append(normalized)

    return normalized_documents