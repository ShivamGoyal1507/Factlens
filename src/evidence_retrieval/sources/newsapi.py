import requests

from src.evidence_retrieval.config import get_newsapi_api_key
from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


NEWSAPI_URL = "https://newsapi.org/v2/everything"


def search_newsapi(query, limit=5):
    api_key = get_newsapi_api_key()
    if not api_key:
        print("Error: Please set NEWSAPI_API_KEY (or NEWSAPI_KEY) in environment.")
        return []

    params = {
        "q": query,
        "pageSize": min(limit, 100),
        "sortBy": "relevancy",
        "language": "en",
    }

    headers = {
        "X-Api-Key": api_key,
    }

    try:
        response = request_with_retry(
            NEWSAPI_URL,
            params=params,
            headers=headers,
            timeout=30,
            max_retries=3,
        )
        response.raise_for_status()
        data = response.json()

        documents = []
        for article in data.get("articles", []):
            text = article.get("content") or article.get("description") or ""

            normalized = normalize_document(
                {
                    "title": article.get("title") or "",
                    "url": article.get("url") or "",
                    "source": "NewsAPI",
                    "date": article.get("publishedAt"),
                    "seen_date": None,
                    "text": text,
                },
                source_hint="NewsAPI",
            )

            if normalized is not None:
                documents.append(normalized)

        return documents

    except (requests.RequestException, ValueError) as error:
        print(f"NewsAPI retrieval failed: {error}")
        return []