def remove_duplicates(documents):
    """
    Remove duplicate articles based on URL.
    Behavior is intentionally preserved:
    - First seen URL is kept
    - Documents with empty URL are kept
    """
    unique_documents = []
    seen_urls = set()

    for doc in documents:
        url = doc.get("url", "").strip()

        if not url:
            unique_documents.append(doc)
            continue

        if url not in seen_urls:
            seen_urls.add(url)
            unique_documents.append(doc)

    return unique_documents