"""FACTLENS end-to-end claim verification entry point.

Query generation is intentionally outside this pipeline. The supplied claim is
retrieved, ranked, selected, and passed to the trained verifier.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


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
    max_evidence_per_source: int | None = 1,
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    include_explanation: bool = True,
) -> dict[str, Any]:
    """Run retrieval, ranking, verification, and explanation generation."""
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim must be a non-empty string")
    if evidence_top_k < 1:
        raise ValueError("evidence_top_k must be positive")
    if max_evidence_per_source is not None and max_evidence_per_source < 1:
        raise ValueError(
            "max_evidence_per_source must be positive or None"
        )

    # Keep duplicate OpenMP runtimes from aborting environments that load both
    # PyTorch and the sentence-transformers stack.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    try:
        from src.classifier.verifier import ClaimVerifier
        from src.evidence_retrieval.retriever import (
            prepare_for_ranking,
            retrieve_evidence,
        )
        from src.ranking.bm25_ranker import BM25Ranker
        from src.ranking.candidate_merger import CandidateMerger
        from src.ranking.cross_encoder_reranker import CrossEncoderReranker
        from src.ranking.dense_ranker import DenseRanker
        from src.ranking.evidence_selector import EvidenceSelector
        from src.ranking.passage_splitter import PassageSplitter
    except ModuleNotFoundError as error:
        missing_module = error.name or "a required module"
        raise RuntimeError(
            f"FACTLENS cannot start because '{missing_module}' is not "
            "installed. Activate the target environment and run "
            "'python -m pip install -r requirements.txt', then install the "
            "spaCy model with 'python -m spacy download en_core_web_sm'."
        ) from error

    retrieved_documents = retrieve_evidence(claim.strip())
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

    bm25_results = BM25Ranker().fit(passages).rank(
        claim,
        top_k=bm25_top_k,
    )
    dense_results = DenseRanker(model_name=dense_model_name).fit(passages).rank(
        claim,
        top_k=dense_top_k,
    )

    merged_results = CandidateMerger().merge(
        bm25_results,
        dense_results,
        top_k=merged_top_k,
    )
    if not merged_results:
        raise RuntimeError("The ranking modules produced no candidates.")

    reranked_results = CrossEncoderReranker(
        model_name=cross_encoder_model_name,
    ).rerank(
        claim,
        merged_results,
        top_k=reranked_top_k,
    )
    if not reranked_results:
        raise RuntimeError("The reranker produced no valid candidates.")

    final_evidence = EvidenceSelector(
        max_per_source=max_evidence_per_source,
        deduplicate=True,
    ).select(
        reranked_results,
        top_k=evidence_top_k,
    )
    if not final_evidence:
        raise RuntimeError("Evidence selection produced no usable evidence.")

    verification = ClaimVerifier().verify(claim, final_evidence)
    explanation = None

    if include_explanation:
        from src.explanation_generation.explainer import VerdictExplainer

        explanation_verdicts = {
            "SUPPORTS": "Supported",
            "REFUTES": "Refuted",
            "NEI": "Not Enough Information",
        }
        explanation = VerdictExplainer().explain(
            claim=claim,
            verdict=explanation_verdicts[verification["verdict"]],
            evidences=[item["text"] for item in final_evidence],
        )

    return {
        "claim": claim.strip(),
        "evidence": final_evidence,
        "verification": verification,
        "explanation": explanation,
        "counts": {
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
        description="FACTLENS - retrieve, rank, and verify a claim",
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
        default=1,
        help="Maximum selected passages per source; use 0 for unlimited",
    )
    parser.add_argument(
        "--without-explanation",
        action="store_true",
        help="Skip explanation generation",
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
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
