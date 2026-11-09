import requests

from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "FACTLENS/1.0 (academic claim verification project)"
}


def search_wikipedia(query, limit=5):
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
    }

    try:
        response = request_with_retry(
            WIKIPEDIA_API,
            params=params,
            headers=HEADERS,
            timeout=20,
            max_retries=3,
        )
        response.raise_for_status()
        data = response.json()

        search_results = data.get("query", {}).get("search", [])
        if not search_results:
            return []

        page_ids = "|".join(str(article["pageid"]) for article in search_results)

        extract_params = {
            "action": "query",
            "pageids": page_ids,
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "format": "json",
        }

        extract_response = request_with_retry(
            WIKIPEDIA_API,
            params=extract_params,
            headers=HEADERS,
            timeout=20,
            max_retries=3,
        )
        extract_response.raise_for_status()
        extract_data = extract_response.json()

        pages = extract_data.get("query", {}).get("pages", {})
        documents = []

        for article in search_results:
            page_id = str(article["pageid"])
            page = pages.get(page_id, {})
            title = page.get("title", article.get("title", ""))

            normalized = normalize_document(
                {
                    "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + requests.utils.quote(title.replace(" ", "_")),
                    "source": "Wikipedia",
                    "date": None,
                    "seen_date": None,
                    "text": page.get("extract", ""),
                },
                source_hint="Wikipedia",
            )

            if normalized is not None:
                documents.append(normalized)

        return documents

    except (requests.RequestException, ValueError) as error:
        print(f"Wikipedia retrieval failed: {error}")
        return []
