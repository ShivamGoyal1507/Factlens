"""FACTLENS end-to-end claim verification entry point.

Pipeline:
    claim -> query generation (Qwen2.5-1.5B-Instruct)
          -> evidence retrieval for every query (Wikipedia, GDELT, NewsAPI)
          -> passage ranking (BM25 + dense -> RRF -> cross-encoder)
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
from typing import Any  # noqa: E402

from src.classifier.verifier import ClaimVerifier  # noqa: E402
from src.evidence_retrieval.retriever import (  # noqa: E402
    prepare_for_ranking,
    retrieve_evidence_for_claim,
)
from src.query_generation import generate_search_plan, normalize_claim  # noqa: E402


def run_pipeline(
    claim: str,
    *,
    passage_sentences: int = 3,
    passage_overlap: int = 1,
    bm25_top_k: int = 20,
    dense_top_k: int = 20,
    merged_top_k: int = 40,
    reranked_top_k: int = 20,
    evidence_top_k: int = 5,
    max_evidence_per_source: int | None = 10,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    include_explanation: bool = True,
    use_query_generation: bool = True,
    claim_made_at: str | None = None,
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

    from src.ranking.bm25_ranker import BM25Ranker
    from src.ranking.candidate_merger import CandidateMerger
    from src.ranking.cross_encoder_reranker import CrossEncoderReranker
    from src.ranking.dense_ranker import DenseRanker
    from src.ranking.entity_scorer import EntityScorer
    from src.ranking.evidence_selector import EvidenceSelector
    from src.ranking.passage_splitter import PassageSplitter

    claim = claim.strip()

    # 1. Query generation: neutral + claim-specific queries, entity queries,
    #    and the claim's named entities (used later for entity-aware ranking).
    #    Falls back to the raw claim if generation is disabled or yields nothing.
    plan = (
        generate_search_plan(claim)
        if use_query_generation
        else {"queries": [], "entities": []}
    )
    queries = plan["queries"] or [claim]
    entities = plan["entities"]
    print("\n===== GENERATED QUERIES =====")
    for index, query in enumerate(queries, 1):
        print(f"{index}. {query}")
    print("Entities:", ", ".join(entities) if entities else "(none)")

    # The verifier was trained on FEVER claims (proper capitalisation, final
    # period) and is case-sensitive, so normalise the claim for ranking,
    # verification and explanation.
    normalized_claim = normalize_claim(claim, entities)
    print("Normalized claim:", normalized_claim)

    # 2. Evidence retrieval for every query, merged and de-duplicated.
    retrieved_documents = retrieve_evidence_for_claim(
        claim,
        queries=queries,
        claim_made_at=claim_made_at,
    )
    ranking_documents = prepare_for_ranking(retrieved_documents)
    if not ranking_documents:
        raise RuntimeError(
            "Evidence retrieval returned no documents with usable text."
        )

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
    dense_ranker = DenseRanker(model_name=dense_model_name).fit(passages)

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
    # Gandhi claim), then keep the top candidates.
    cross_encoded = CrossEncoderReranker(
        model_name=cross_encoder_model_name,
    ).rerank_multi(
        ranking_texts,
        merged_results,
        top_k=len(merged_results),
    )
    reranked_results = EntityScorer().score(cross_encoded, entities)[:reranked_top_k]
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

    # 4. Verification: claim vs. each passage separately (as the FEVER model
    #    was trained), then aggregated. Concatenating all passages truncates
    #    at 256 tokens and hides the decisive passage.
    verification = ClaimVerifier().verify_evidence(
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
