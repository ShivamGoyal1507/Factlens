
from pydoc import doc

from src.evidence_retrieval.sources.gdelt import search_gdelt
import pytest

pytestmark = pytest.mark.live


@pytest.mark.live
def test_newsapi_live():
    pytest.skip("Live API test skipped by default; set FACTLENS_LIVE_API=1 to run")

def test_gdelt():

    query = "India technology news sourcelang:English"

    results = search_gdelt(query, limit=5)

    print("\n===== GDELT RETRIEVAL TEST =====")
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
    test_gdelt()
