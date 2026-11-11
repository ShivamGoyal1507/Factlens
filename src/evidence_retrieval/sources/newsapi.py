import re

import requests

from src.evidence_retrieval.config import get_newsapi_api_key
from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


NEWSAPI_URL = "https://newsapi.org/v2/everything"

# NewsAPI truncates `content` to ~200 chars and appends "… [+1234 chars]".
_TRUNCATION_MARKER = re.compile(r"\s*(?:…|\.\.\.)?\s*\[\+\d+ chars\]\s*$")


def article_text(article):
    """
    Evidence text for a NewsAPI article: description plus the content
    snippet (truncation marker removed, repeated text dropped). NewsAPI only
    returns snippets, so both fields together carry the most usable facts.
    """
    description = (article.get("description") or "").strip()
    content = _TRUNCATION_MARKER.sub("", article.get("content") or "").strip()
    content = re.sub(r"<[^>]+>", " ", content)  # some feeds embed HTML
    content = re.sub(r"\s+", " ", content).strip()

    parts = []
    for part in (description, content):
        if not part:
            continue
        if any(part[:80].lower() in existing.lower() for existing in parts):
            continue
        parts.append(part if part.endswith((".", "!", "?")) else part + ".")
    return " ".join(parts)

# HTTP 429 from NewsAPI means the plan's request quota is used up (free plan:
# 100 requests/day), so further calls in this process would only fail too.
_quota_exhausted = False


def reset_newsapi_quota_flag():
    global _quota_exhausted
    _quota_exhausted = False


def search_newsapi(query, limit=5):
    global _quota_exhausted
    if _quota_exhausted:
        return []

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
            max_retries=0,  # single attempt: no repeated API hits on failure
        )
        if response.status_code == 429:
            _quota_exhausted = True
            print(
                "NewsAPI request quota exhausted (HTTP 429); skipping NewsAPI "
                "for the rest of this run."
            )
            return []
        response.raise_for_status()
        data = response.json()

        documents = []
        for article in data.get("articles", []):
            text = article_text(article)

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