"""
Try the evidence-ranking pipeline on your own claim.

Pipeline:
    articles -> passages -> BM25 + dense -> RRF merge -> cross-encoder rerank -> select top-k

Usage (from the project root, in the `factlens` conda env):

    python scripts/test_evidence_ranking.py                    # CLAIM + sample ARTICLES below (offline)
    python scripts/test_evidence_ranking.py "Your claim here"  # your claim, sample ARTICLES
    python scripts/test_evidence_ranking.py --live             # CLAIM -> query generation -> live retrieval -> ranking
    python scripts/test_evidence_ranking.py --live "Your claim here"
"""

import os
import sys
import time
from pathlib import Path

# Avoid the macOS "libomp already initialized" crash with torch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# torch MUST be imported before spacy / sentence_transformers. Otherwise they
# load a different OpenMP runtime first and loading the Qwen query model
# segfaults on macOS (exit code 139).
import torch  # noqa: E402,F401

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ranking.passage_splitter import PassageSplitter  # noqa: E402
from src.ranking.bm25_ranker import BM25Ranker  # noqa: E402
from src.ranking.dense_ranker import DenseRanker  # noqa: E402
from src.ranking.candidate_merger import CandidateMerger  # noqa: E402
from src.ranking.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402
from src.ranking.evidence_selector import EvidenceSelector  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these to test your own claim / evidence
# ---------------------------------------------------------------------------
CLAIM = "India's economy grew in 2025."

# Used in offline mode. Each article needs at least: title, url, source, text.
ARTICLES = [
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
        ),
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
        ),
    },
    {
        "title": "Sports News",
        "url": "https://example.net/sports",
        "source": "Example Sports",
        "published_date": "2026-02-05",
        "text": (
            "The national cricket team won an international match. "
            "The players celebrated the victory with their supporters."
        ),
    },
]

# Ranking settings
SENTENCES_PER_PASSAGE = 3
PASSAGE_OVERLAP = 1
TOP_K_FIRST_STAGE = 20   # BM25 / dense candidates each
TOP_K_MERGED = 40        # after RRF merge
TOP_K_RERANKED = 20      # after cross-encoder
TOP_K_FINAL = 5          # final evidence
MAX_PER_SOURCE = 1       # None = no limit per source
MIN_RERANK_SCORE = None  # e.g. 0.0 to drop weak matches


def get_live_articles(claim):
    """Generate queries for the claim, retrieve evidence, prepare it for ranking."""
    from src.query_generation import generate_queries
    from src.evidence_retrieval.retriever import (
        prepare_for_ranking,
        retrieve_evidence_for_claim,
    )

    print("\n[1] Generating queries...")
    queries = generate_queries(claim)
    for i, query in enumerate(queries, 1):
        print(f"    {i}. {query}")

    print("\n[2] Retrieving evidence...")
    documents = retrieve_evidence_for_claim(claim, queries=queries)
    articles = prepare_for_ranking(documents)
    print(f"    Retrieved documents: {len(documents)} (with usable text: {len(articles)})")
    return articles


def rank_evidence(claim, articles):
    timings = {}

    t = time.perf_counter()
    splitter = PassageSplitter(
        sentences_per_passage=SENTENCES_PER_PASSAGE,
        overlap=PASSAGE_OVERLAP,
    )
    passages = splitter.split_articles(articles)
    timings["split"] = time.perf_counter() - t
    print(f"\nPassages            : {len(passages)} from {len(articles)} articles")

    if not passages:
        print("No passages were generated - nothing to rank.")
        return []

    t = time.perf_counter()
    bm25_results = BM25Ranker().fit(passages).rank(claim, top_k=TOP_K_FIRST_STAGE)
    timings["bm25"] = time.perf_counter() - t
    print(f"BM25 results        : {len(bm25_results)}")

    t = time.perf_counter()
    dense_results = DenseRanker().fit(passages).rank(claim, top_k=TOP_K_FIRST_STAGE)
    timings["dense"] = time.perf_counter() - t
    print(f"Dense results       : {len(dense_results)}")

    merged = CandidateMerger(rrf_k=60).merge(bm25_results, dense_results, top_k=TOP_K_MERGED)
    print(f"Merged candidates   : {len(merged)}")

    t = time.perf_counter()
    reranked = CrossEncoderReranker().rerank(claim, merged, top_k=TOP_K_RERANKED)
    timings["rerank"] = time.perf_counter() - t
    print(f"Reranked candidates : {len(reranked)}")

    selector = EvidenceSelector(
        max_per_source=MAX_PER_SOURCE,
        min_score=MIN_RERANK_SCORE,
        deduplicate=True,
    )
    final = selector.select(reranked, top_k=TOP_K_FINAL)

    print("\nTimings: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()))
    return final


def print_evidence(final):
    print(f"\n========== FINAL EVIDENCE ({len(final)}) ==========")
    for ev in final:
        ranks = ev.get("retrieval_ranks", {})
        print(f"\nRank {ev['evidence_rank']}  |  rerank score: {ev.get('rerank_score', 0):.3f}")
        print(f"  Title  : {ev.get('title', '')}")
        print(f"  Source : {ev.get('source', '')}   Date: {ev.get('published_date')}")
        print(f"  URL    : {ev.get('url', '')}")
        print(
            f"  Found by: {', '.join(ev.get('retrieval_methods', []))} "
            f"(bm25 rank {ranks.get('bm25', '-')}, dense rank {ranks.get('dense', '-')}, "
            f"rrf {ev.get('rrf_score', 0):.4f})"
        )
        print(f"  Text   : {ev.get('text', '')}")


def main():
    args = sys.argv[1:]
    live = "--live" in args
    args = [a for a in args if a != "--live"]
    claim = " ".join(args).strip() or CLAIM

    print(f"Claim: {claim}")
    print(f"Mode : {'live retrieval' if live else 'offline sample articles'}")

    articles = get_live_articles(claim) if live else ARTICLES
    if not articles:
        print("\nNo articles to rank.")
        return

    final = rank_evidence(claim, articles)
    print_evidence(final)


if __name__ == "__main__":
    main()
