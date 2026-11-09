from src.evidence_retrieval.sources.wikipedia import search_wikipedia
import pytest

pytestmark = pytest.mark.live


@pytest.mark.live
def test_newsapi_live():
    pytest.skip("Live API test skipped by default; set FACTLENS_LIVE_API=1 to run")

def test_wikipedia():

    query = "Chandrayaan-3 Moon landing date"

    results = search_wikipedia(query)

    print("\n===== WIKIPEDIA TEST =====")

    print("Query:", query)
    print("Total results:", len(results))

    for i, doc in enumerate(results, 1):

        print(f"\nArticle {i}")
        print("Title:", doc["title"])
        print("URL:", doc["url"])
        print("Source:", doc["source"])
        print("Date:", doc["date"])

        print("Seen Date:", doc["seen_date"])

        print("Text:", doc["text"])


if __name__ == "__main__":
    test_wikipedia()