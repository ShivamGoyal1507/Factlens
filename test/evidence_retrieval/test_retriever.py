
from src.evidence_retrieval.retriever import (
    retrieve_evidence,
    remove_duplicates,
    validate_documents,
    get_text_ready_documents
)


def test_retriever():

    query = "India artificial intelligence technology"

    results = retrieve_evidence(query)

    print("\n===== FINAL RETRIEVED DOCUMENTS =====")

    for index, doc in enumerate(results, start=1):

        print(f"\nDocument {index}")
        print("Title:", doc["title"])
        print("Source:", doc["source"])
        print("Date:", doc["date"])
        print("URL:", doc["url"])


if __name__ == "__main__":
    test_retriever()


from src.evidence_retrieval.retriever import (
    retrieve_evidence,
    remove_duplicates
)


def test_deduplication():

    documents = [
        {
            "title": "Article One",
            "url": "https://example.com/article1",
            "source": "GDELT"
        },
        {
            "title": "Article One",
            "url": "https://example.com/article1",
            "source": "NewsAPI"
        },
        {
            "title": "Article Two",
            "url": "https://example.com/article2",
            "source": "Wikipedia"
        }
    ]

    unique_documents = remove_duplicates(documents)

    print("\n===== DEDUPLICATION TEST =====")
    print("Before:", len(documents))
    print("After:", len(unique_documents))

    for doc in unique_documents:
        print(doc["title"], doc["url"])

    assert len(unique_documents) == 2

    print("Deduplication test passed!")




def test_input_validation():

    print("\n===== INPUT VALIDATION TEST =====")

    test_queries = [
        "",
        "   ",
        None,
        123,
        "India artificial intelligence"
    ]

    for query in test_queries:

        print("\nTesting query:", repr(query))

        results = retrieve_evidence(query)

        print("Results:", len(results))



def test_result_validation():

    documents = [

        # Valid document
        {
            "title": "AI in India",
            "url": "https://example.com/1",
            "source": "NewsAPI",
            "date": None,
            "seen_date": None,
            "text": "India is developing AI technology."
        },

        # Missing title
        {
            "title": "",
            "url": "https://example.com/2",
            "source": "NewsAPI",
            "date": None,
            "seen_date": None,
            "text": "Some article text."
        },

        # Missing URL
        {
            "title": "AI Research",
            "url": "",
            "source": "GDELT",
            "date": None,
            "seen_date": "20261020T100000Z",
            "text": "Some article text."
        },

        # Missing text
        {
            "title": "AI Development",
            "url": "https://example.com/4",
            "source": "Wikipedia",
            "date": None,
            "seen_date": None,
            "text": ""
        },

        # Valid document with missing dates
        {
            "title": "AI Industry",
            "url": "https://example.com/5",
            "source": "Wikipedia",
            "date": None,
            "seen_date": None,
            "text": "Information about the AI industry."
        }
    ]

    valid_documents = validate_documents(documents)

    print("\n===== RESULT VALIDATION TEST =====")
    print("Documents before:", len(documents))
    print("Documents after:", len(valid_documents))

    for doc in valid_documents:
        print("Valid:", doc["title"])

    assert len(valid_documents) == 3

    print("Result validation test passed!")


def test_text_ready_documents():

    documents = [
        {
            "title": "Wikipedia Article",
            "url": "https://example.com/1",
            "source": "Wikipedia",
            "text": "Artificial intelligence is a field of computer science."
        },
        {
            "title": "GDELT Article",
            "url": "https://example.com/2",
            "source": "GDELT",
            "text": ""
        },
        {
            "title": "News Article",
            "url": "https://example.com/3",
            "source": "NewsAPI",
            "text": "AI is transforming different industries."
        }
    ]

    text_ready = get_text_ready_documents(documents)

    assert len(text_ready) == 2

    print("\n===== TEXT-READY DOCUMENT TEST =====")
    print("Documents before:", len(documents))
    print("Documents ready for ranking:", len(text_ready))

    for doc in text_ready:
        print("Valid for ranking:", doc["title"])

    print("Text-ready document test passed!")





if __name__ == "__main__":
    test_input_validation()
    test_deduplication()
    test_result_validation()
    test_text_ready_documents()