
import argparse

from src.evidence_retrieval.retriever import retrieve_evidence


def main():
    parser = argparse.ArgumentParser(
        description="FACTLENS - Evidence-based fact verification"
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Claim or query to retrieve supporting evidence for"
    )

    args = parser.parse_args()

    query = args.query

    if not query:
        query = input("Enter your query: ").strip()

    if not query:
        print("Please enter a valid query.")
        return

    print(f"\nSearching evidence for: {query}\n")

    results = retrieve_evidence(query)

    if not results:
        print("No evidence found.")
        return

    print("\nRetrieved Evidence:")

    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}")
        print("Title:", doc.get("title"))
        print("Source:", doc.get("source"))
        print("URL:", doc.get("url"))
        print("Date:", doc.get("date"))
        print("Has text:", bool(doc.get("text", "").strip()))


if __name__ == "__main__":
    main()
