import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.ranking.passage_splitter import PassageSplitter
from src.ranking.bm25_ranker import BM25Ranker
from src.ranking.dense_ranker import DenseRanker
from src.ranking.candidate_merger import CandidateMerger
from src.ranking.cross_encoder_reranker import CrossEncoderReranker
from src.ranking.evidence_selector import EvidenceSelector


def main():
    claim = "India's economy grew in 2025."

    articles = [
        {
            "title": "Economic Growth Report",
            "url": "https://example.com/economy",
            "source": "Example News",
            "published_date": "2026-01-15",
            "text": (
                "India's economy expanded during the reported period. "
                "The report described an increase in economic activity. "
                "Several industries contributed to the expansion. "
                "The government also announced new infrastructure projects."
            )
        },
        {
            "title": "Economic Outlook",
            "url": "https://example.org/outlook",
            "source": "Example Research",
            "published_date": "2026-02-01",
            "text": (
                "Economic growth is measured using changes in real GDP. "
                "Annual growth rates can differ across reporting agencies. "
                "Economic forecasts are not the same as final measurements."
            )
        },
        {
            "title": "Sports News",
            "url": "https://example.net/sports",
            "source": "Example Sports",
            "published_date": "2026-02-05",
            "text": (
                "The national cricket team won an international match. "
                "The players celebrated the victory with their supporters."
            )
        }
    ]

    splitter = PassageSplitter(
        sentences_per_passage=3,
        overlap=1
    )

    passages = splitter.split_articles(articles)

    if not passages:
        raise RuntimeError("No passages were generated")

    print(f"Generated passages: {len(passages)}")

    bm25_ranker = BM25Ranker()
    bm25_ranker.fit(passages)
    bm25_results = bm25_ranker.rank(claim, top_k=20)

    print(f"BM25 results: {len(bm25_results)}")

    dense_ranker = DenseRanker()
    dense_ranker.fit(passages)
    dense_results = dense_ranker.rank(claim, top_k=20)

    print(f"Dense results: {len(dense_results)}")

    merger = CandidateMerger(rrf_k=60)
    merged_results = merger.merge(
        bm25_results,
        dense_results,
        top_k=40
    )

    print(f"Merged candidates: {len(merged_results)}")

    reranker = CrossEncoderReranker()
    reranked_results = reranker.rerank(
        claim,
        merged_results,
        top_k=20
    )

    print(f"Reranked candidates: {len(reranked_results)}")

    selector = EvidenceSelector(
        max_per_source=1,
        min_score=None,
        deduplicate=True,
    )
    final_evidence = selector.select(
        reranked_results,
        top_k=5,
    )

    print(f"\nFinal evidence: {len(final_evidence)}")
    print(final_evidence)
    for evidence in final_evidence:
        print(f"\nRank: {evidence['evidence_rank']}")
        print(f"Title: {evidence.get('title', '')}")
        print(f"Source: {evidence.get('source', '')}")
        print(f"URL: {evidence.get('url', '')}")
        print(f"Rerank score: {evidence.get('rerank_score')}")
        print(f"Text: {evidence['text']}")


if __name__ == "__main__":
    main()