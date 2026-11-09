from src.evidence_retrieval.sources.newsapi import search_newsapi
import pytest

pytestmark = pytest.mark.live


@pytest.mark.live
def test_newsapi_live():
    pytest.skip("Live API test skipped by default; set FACTLENS_LIVE_API=1 to run")


def test_newsapi():

    query = "India artificial intelligence technology"

    results = search_newsapi(query, limit=5)

    print("\n===== NEWSAPI RETRIEVAL TEST =====")
    print("Query:", query)
    print("Total results:", len(results))

    for index, doc in enumerate(results, start=1):

        print(f"\nArticle {index}")
        print("Title:", doc["title"])
        print("URL:", doc["url"])
        print("Source:", doc["source"])
        print("Date:", doc["date"])
        print("Seen Date:", doc["seen_date"])
        print("Text:", doc["text"])


if __name__ == "__main__":
    test_newsapi()
