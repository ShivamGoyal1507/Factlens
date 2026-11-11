"""FACTLENS end-to-end claim verification entry point.

Pipeline:
    claim -> query generation (Qwen2.5-1.5B-Instruct)
          -> evidence retrieval for every query (Wikipedia, NewsAPI)
          -> sentence ranking (BM25 + dense -> RRF -> cross-encoder
             -> entity / claim-page priors -> verifier relevance)
          -> evidence selection
          -> verification (DeBERTa LoRA, per passage, then aggregated)
          -> explanation (reuses the Qwen2.5-1.5B-Instruct query model)
"""

from __future__ import annotations

import os

# Must run before ANY torch / numpy / spacy / sentence-transformers import.
# torch and spaCy/scikit-learn each ship their own OpenMP runtime on macOS;
# without this flag the second one aborts the process ("OMP: Error #15").
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Import torch first so its OpenMP runtime is the one that gets loaded.
# Importing spacy / sentence-transformers first can segfault model loading.
import torch  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
from functools import lru_cache  # noqa: E402
from typing import Any  # noqa: E402

from src.classifier.verifier import ClaimVerifier  # noqa: E402
from src.evidence_retrieval.retriever import (  # noqa: E402
    prepare_for_ranking,
    retrieve_evidence_for_claim,
)
from src.query_generation import generate_search_plan, normalize_claim  # noqa: E402
from src.query_generation.generator import _extract_named_entities  # noqa: E402


# Models are loaded once per process (batch evaluation calls run_pipeline
# many times; reloading DeBERTa and the encoders per claim dominated runtime).
@lru_cache(maxsize=None)
def _dense_model(name: str):
    from src.ranking.dense_ranker import DenseRanker

    return DenseRanker(model_name=name)


@lru_cache(maxsize=None)
def _cross_encoder(name: str):
    from src.ranking.cross_encoder_reranker import CrossEncoderReranker

    return CrossEncoderReranker(model_name=name)


@lru_cache(maxsize=1)
def _verifier() -> ClaimVerifier:
    return ClaimVerifier()


# The query-generation LLM is not safe to call from several threads at once;
# prepare_claim() may run in worker threads (batch evaluation), so it is
# serialised. Retrieval itself runs concurrently.
_QUERY_MODEL_LOCK = threading.Lock()


def rank_evidence(
    normalized_claim: str,
    queries: list[str],
    entities: list[str],
    ranking_documents: list[dict[str, Any]],
    *,
    passage_sentences: int = 3,
    passage_overlap: int = 1,
    bm25_top_k: int = 30,
    dense_top_k: int = 30,
    merged_top_k: int = 60,
    reranked_top_k: int = 30,
    evidence_top_k: int = 5,
    max_evidence_per_source: int | None = 10,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    verifier_rerank_weight: float = 5.0,
    verifier_rerank_pool: int = 30,
) -> dict[str, Any]:
    """Split retrieved documents into passages, rank them and select evidence."""
    from src.ranking.bm25_ranker import BM25Ranker
    from src.ranking.candidate_merger import CandidateMerger
    from src.ranking.claim_page_scorer import ClaimPageScorer
    from src.ranking.entity_scorer import EntityScorer
    from src.ranking.evidence_selector import EvidenceSelector
    from src.ranking.passage_splitter import PassageSplitter
    from src.ranking.verifier_reranker import VerifierReranker

    splitter = PassageSplitter(
        sentences_per_passage=passage_sentences,
        overlap=passage_overlap,
    )
    passages = splitter.split_articles(ranking_documents)
    if not passages:
        raise RuntimeError("No passages could be generated from retrieved evidence.")

    # 3. Ranking. Every stage uses the claim AND the generated queries, so a
    #    passage answering a neutral query ("who is the prime minister of
    #    India") is found even if it shares few words with the claim.
    # Entity-only queries ("Jensen Huang") are for retrieval only: as ranking
    # texts they push generic biographies to the top even though they say
    # nothing about the claim.
    entity_keys = {entity.lower() for entity in entities}
    ranking_queries = [q for q in queries if q.lower() not in entity_keys]
    ranking_texts = list(dict.fromkeys([normalized_claim] + ranking_queries))

    bm25_ranker = BM25Ranker().fit(passages)
    dense_ranker = _dense_model(dense_model_name).fit(passages)

    rankings = []
    for index, text in enumerate(ranking_texts):
        label = "claim" if index == 0 else f"q{index}"
        rankings.append((f"bm25:{label}", bm25_ranker.rank(text, top_k=bm25_top_k)))
        rankings.append((f"dense:{label}", dense_ranker.rank(text, top_k=dense_top_k)))

    bm25_results = rankings[0][1]
    dense_results = rankings[1][1]

    merged_results = CandidateMerger().merge_many(
        rankings,
        top_k=merged_top_k,
    )
    if not merged_results:
        raise RuntimeError("The ranking modules produced no candidates.")

    # Cross-encoder: best score over claim + queries for every merged
    # candidate, then entity-aware adjustment (boost passages about the
    # claim's entities, penalise namesakes such as Rajiv Gandhi for a Rahul
    # Gandhi claim), then a prior for sentences on pages the claim names
    # (where FEVER evidence lives), then the verifier's own judgement of
    # whether each top sentence settles the claim (1 - P(NEI)).
    cross_encoded = _cross_encoder(cross_encoder_model_name).rerank_multi(
        ranking_texts,
        merged_results,
        top_k=len(merged_results),
    )
    entity_scored = EntityScorer().score(cross_encoded, entities)
    page_scored = ClaimPageScorer().score(entity_scored, normalized_claim)
    verifier = _verifier()
    reranked_results = VerifierReranker(
        verifier,
        weight=verifier_rerank_weight,
        pool_size=verifier_rerank_pool,
    ).rerank(normalized_claim, page_scored)[:reranked_top_k]
    if not reranked_results:
        raise RuntimeError("The reranker produced no valid candidates.")

    final_evidence = EvidenceSelector(
        max_per_source=max_evidence_per_source,
        deduplicate=True,
        group_by="url",  # limit per article, not per source name
        score_field="final_score",
    ).select(
        reranked_results,
        top_k=evidence_top_k,
    )
    if not final_evidence:
        raise RuntimeError("Evidence selection produced no usable evidence.")

    return {
        "passages": passages,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
        "merged_results": merged_results,
        "reranked_results": reranked_results,
        "final_evidence": final_evidence,
    }


def prepare_claim(
    claim: str,
    *,
    use_query_generation: bool = True,
    claim_made_at: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Network stage: query generation and evidence retrieval.

    Thread-safe, so batch evaluation can retrieve evidence for upcoming
    claims in worker threads while the GPU stage (finish_claim) runs.
    """
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")

    claim = claim.strip()

    # 1. Query generation: neutral + claim-specific queries, entity queries,
    #    and the claim's named entities (used later for entity-aware ranking).
    #    Falls back to the raw claim (and capitalised spans as entities) if
    #    generation is disabled or yields nothing.
    if use_query_generation:
        with _QUERY_MODEL_LOCK:
            plan = generate_search_plan(claim)
    else:
        plan = {"queries": [], "entities": []}
    queries = plan["queries"] or [claim]
    entities = plan["entities"] or _extract_named_entities(claim)
    if verbose:
        print("\n===== GENERATED QUERIES =====")
        for index, query in enumerate(queries, 1):
            print(f"{index}. {query}")
        print("Entities:", ", ".join(entities) if entities else "(none)")

    # The verifier was trained on FEVER claims (proper capitalisation, final
    # period) and is case-sensitive, so normalise the claim for ranking,
    # verification and explanation.
    normalized_claim = normalize_claim(claim, entities)
    if verbose:
        print("Normalized claim:", normalized_claim)

    # 2. Evidence retrieval: the Wikipedia pages the claim mentions, then
    #    every query; merged and de-duplicated.
    retrieved_documents = retrieve_evidence_for_claim(
        normalized_claim,
        queries=queries,
        claim_made_at=claim_made_at,
        entities=entities,
    )
    ranking_documents = prepare_for_ranking(retrieved_documents)
    return {
        "claim": claim,
        "normalized_claim": normalized_claim,
        "queries": queries,
        "entities": entities,
        "retrieved_documents": retrieved_documents,
        "ranking_documents": ranking_documents,
    }


def finish_claim(
    prepared: dict[str, Any],
    *,
    passage_sentences: int = 3,
    passage_overlap: int = 1,
    bm25_top_k: int = 30,
    dense_top_k: int = 30,
    merged_top_k: int = 60,
    reranked_top_k: int = 30,
    evidence_top_k: int = 5,
    max_evidence_per_source: int | None = 10,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    include_explanation: bool = True,
    verifier_rerank_weight: float = 5.0,
    verifier_rerank_pool: int = 30,
) -> dict[str, Any]:
    """Model stage: ranking, verification and (optionally) explanation."""
    claim = prepared["claim"]
    normalized_claim = prepared["normalized_claim"]
    queries = prepared["queries"]
    entities = prepared["entities"]
    retrieved_documents = prepared["retrieved_documents"]
    ranking_documents = prepared["ranking_documents"]

    if not ranking_documents:
        # Every source failed or was rate-limited: without evidence the only
        # honest verdict is NEI, so report that instead of crashing.
        print(
            "\nNo evidence could be retrieved (all sources failed or were "
            "rate-limited). Returning NEI."
        )
        return _no_evidence_result(
            claim, normalized_claim, queries, entities, len(retrieved_documents)
        )

    ranked = rank_evidence(
        normalized_claim,
        queries,
        entities,
        ranking_documents,
        passage_sentences=passage_sentences,
        passage_overlap=passage_overlap,
        bm25_top_k=bm25_top_k,
        dense_top_k=dense_top_k,
        merged_top_k=merged_top_k,
        reranked_top_k=reranked_top_k,
        evidence_top_k=evidence_top_k,
        max_evidence_per_source=max_evidence_per_source,
        dense_model_name=dense_model_name,
        cross_encoder_model_name=cross_encoder_model_name,
        verifier_rerank_weight=verifier_rerank_weight,
        verifier_rerank_pool=verifier_rerank_pool,
    )
    passages = ranked["passages"]
    bm25_results = ranked["bm25_results"]
    dense_results = ranked["dense_results"]
    merged_results = ranked["merged_results"]
    reranked_results = ranked["reranked_results"]
    final_evidence = ranked["final_evidence"]
    verifier = _verifier()

    # 4. Verification: claim vs. each passage separately (as the FEVER model
    #    was trained), then aggregated. Concatenating all passages truncates
    #    at 256 tokens and hides the decisive passage.
    verification = verifier.verify_evidence(
        normalized_claim,
        [item["text"] for item in final_evidence],
    )
    for item, prediction in zip(final_evidence, verification["per_evidence"]):
        item["verifier_verdict"] = prediction["verdict"]
        item["verifier_probabilities"] = prediction["probabilities"]

    # 5. Explanation from the passages that decided the verdict (all
    #    passages for NEI), reusing the already-loaded query model.
    explanation = None
    if include_explanation:
        from src.explanation_generation.explainer import VerdictExplainer
        from src.query_generation.slm_client import MODEL_NAME as QUERY_MODEL_NAME

        explanation_verdicts = {
            "SUPPORTS": "Supported",
            "REFUTES": "Refuted",
            "NEI": "Not Enough Information",
        }
        decisive = verification["decisive_evidence"]
        explanation_evidence = (
            [final_evidence[i]["text"] for i in decisive]
            if decisive
            else [item["text"] for item in final_evidence]
        )
        explanation = VerdictExplainer(model_name=QUERY_MODEL_NAME).explain(
            claim=normalized_claim,
            verdict=explanation_verdicts[verification["verdict"]],
            evidences=explanation_evidence[:3],
        )

    return {
        "claim": claim,
        "normalized_claim": normalized_claim,
        "queries": queries,
        "entities": entities,
        "evidence": final_evidence,
        "retrieved_titles": list(dict.fromkeys(
            doc.get("title", "") for doc in ranking_documents
        )),
        "verification": verification,
        "explanation": explanation,
        "counts": {
            "queries": len(queries),
            "retrieved_documents": len(retrieved_documents),
            "ranking_documents": len(ranking_documents),
            "passages": len(passages),
            "bm25_candidates": len(bm25_results),
            "dense_candidates": len(dense_results),
            "merged_candidates": len(merged_results),
            "reranked_candidates": len(reranked_results),
            "selected_evidence": len(final_evidence),
            "explanation_generated": explanation is not None,
        },
    }


def run_pipeline(
    claim: str,
    *,
    passage_sentences: int = 3,
    passage_overlap: int = 1,
    bm25_top_k: int = 30,
    dense_top_k: int = 30,
    merged_top_k: int = 60,
    reranked_top_k: int = 30,
    evidence_top_k: int = 5,
    max_evidence_per_source: int | None = 10,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    include_explanation: bool = True,
    use_query_generation: bool = True,
    claim_made_at: str | None = None,
    verifier_rerank_weight: float = 5.0,
    verifier_rerank_pool: int = 30,
) -> dict[str, Any]:
    """Run query generation, retrieval, ranking, verification and explanation."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if evidence_top_k < 1:
        raise ValueError("evidence_top_k must be positive")
    if max_evidence_per_source is not None and max_evidence_per_source < 1:
        raise ValueError(
            "max_evidence_per_source must be positive or None"
        )

    claim = claim.strip()
    prepared = prepare_claim(
        claim,
        use_query_generation=use_query_generation,
        claim_made_at=claim_made_at,
    )
    return finish_claim(
        prepared,
        passage_sentences=passage_sentences,
        passage_overlap=passage_overlap,
        bm25_top_k=bm25_top_k,
        dense_top_k=dense_top_k,
        merged_top_k=merged_top_k,
        reranked_top_k=reranked_top_k,
        evidence_top_k=evidence_top_k,
        max_evidence_per_source=max_evidence_per_source,
        dense_model_name=dense_model_name,
        cross_encoder_model_name=cross_encoder_model_name,
        include_explanation=include_explanation,
        verifier_rerank_weight=verifier_rerank_weight,
        verifier_rerank_pool=verifier_rerank_pool,
    )


def _no_evidence_result(
    claim: str,
    normalized_claim: str,
    queries: list[str],
    entities: list[str],
    retrieved_count: int,
) -> dict[str, Any]:
    """Pipeline result when retrieval produced no usable evidence."""
    return {
        "claim": claim,
        "normalized_claim": normalized_claim,
        "queries": queries,
        "entities": entities,
        "evidence": [],
        "retrieved_titles": [],
        "verification": {
            "claim": normalized_claim,
            "verdict": "NEI",
            "confidence": 1.0,
            "probabilities": {"SUPPORTS": 0.0, "REFUTES": 0.0, "NEI": 1.0},
            "decisive_evidence": [],
            "per_evidence": [],
            "reason": "no_evidence_retrieved",
        },
        "explanation": (
            "No evidence could be retrieved for this claim (the evidence "
            "sources failed or rate-limited the request), so it cannot be "
            "verified. Try again in a few minutes."
        ),
        "counts": {
            "queries": len(queries),
            "retrieved_documents": retrieved_count,
            "ranking_documents": 0,
            "passages": 0,
            "selected_evidence": 0,
            "explanation_generated": False,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FACTLENS - generate queries, retrieve, rank, and verify a claim",
    )
    parser.add_argument(
        "claim",
        nargs="?",
        help="Claim to verify",
    )
    parser.add_argument(
        "--evidence-top-k",
        type=int,
        default=5,
        help="Number of selected evidence passages (default: 5)",
    )
    parser.add_argument(
        "--max-evidence-per-source",
        type=int,
        default=2,
        help="Maximum selected passages per article (URL); use 0 for unlimited",
    )
    parser.add_argument(
        "--without-explanation",
        action="store_true",
        help="Skip explanation generation",
    )
    parser.add_argument(
        "--without-query-generation",
        action="store_true",
        help="Search with the raw claim instead of generated queries",
    )
    parser.add_argument(
        "--claim-made-at",
        default=None,
        help="Date the claim was made (e.g. 2024-06-15); newer evidence is filtered out",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    claim = args.claim or input("Enter your claim: ").strip()
    if not claim:
        raise SystemExit("Please provide a non-empty claim.")
    if args.evidence_top_k < 1:
        raise SystemExit("--evidence-top-k must be positive.")
    if args.max_evidence_per_source < 0:
        raise SystemExit("--max-evidence-per-source cannot be negative.")

    result = run_pipeline(
        claim,
        evidence_top_k=args.evidence_top_k,
        max_evidence_per_source=(
            None if args.max_evidence_per_source == 0
            else args.max_evidence_per_source
        ),
        include_explanation=not args.without_explanation,
        use_query_generation=not args.without_query_generation,
        claim_made_at=args.claim_made_at,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
