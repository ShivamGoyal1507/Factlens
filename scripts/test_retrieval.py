

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence_retrieval.retriever import retrieve_evidence



def test_retrieval():
    query = "Artificial intelligence developments in India"

    print("\nQuery:", query)
    print("\nStarting Evidence Retrieval...")

    results = retrieve_evidence(query)

    if not results:
        print("\nNo evidence retrieved.")
        return

    print("\n========== RETRIEVED EVIDENCE ==========")

    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}")
        print("Title:", doc.get("title"))
        print("Source:", doc.get("source"))
        print("URL:", doc.get("url"))
        print("Date:", doc.get("date"))
        print("Seen Date:", doc.get("seen_date"))
        print("Has Text:", bool(doc.get("text", "").strip()))

    print("\n========== SUMMARY ==========")
    print("Total documents retrieved:", len(results))
    print(
        "Documents with text:",
        sum(bool(doc.get("text", "").strip()) for doc in results)
    )


if __name__ == "__main__":
    test_retrieval()
