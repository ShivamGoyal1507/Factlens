
import requests

from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def search_gdelt(query, limit=10):
    """
    Search GDELT DOC 2.0 for news articles.
    Returns metadata-only evidence documents.
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": limit,
        "sort": "HybridRel",
    }

    headers = {
        "User-Agent": "FACTLENS/1.0 (academic claim verification project)"
    }

    try:
        response = request_with_retry(
            GDELT_API,
            params=params,
            headers=headers,
            timeout=30,
            max_retries=3,
        )
        response.raise_for_status()

        try:
            data = response.json()
        except ValueError as error:
            print(
                "GDELT returned invalid JSON. "
                f"Content-Type: {response.headers.get('Content-Type')}, "
                f"Response preview: {response.text[:200]}"
            )
            raise ValueError("Invalid JSON response from GDELT") from error

        if not isinstance(data, dict):
            print("GDELT returned an unexpected response format.")
            return []

        articles = data.get("articles", [])

        if not isinstance(articles, list):
            print("GDELT response does not contain a valid articles list.")
            return []

        documents = []

        for article in articles:
            if not isinstance(article, dict):
                continue

            doc = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": "GDELT",
                "date": None,
                "seen_date": article.get("seendate"),
                "text": "",
            }

            normalized = normalize_document(
                doc,
                source_hint="GDELT",
            )

            if normalized is not None:
                documents.append(normalized)

        return documents

    except (requests.RequestException, ValueError) as error:
        print(f"GDELT retrieval failed: {error}")
        return []
