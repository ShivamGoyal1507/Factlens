"""
Re-rank evidence candidates with the FEVER verifier itself.

The verifier was trained to output NEI for sentences that do not settle the
claim and SUPPORTS/REFUTES for gold evidence, so 1 - P(NEI) is a
FEVER-trained "is this decisive evidence?" score - exactly what sentence
selection needs and what the MS MARCO cross-encoder (a QA relevance model)
does not measure.

final_score += weight * (P(SUPPORTS) + P(REFUTES))

Only the top `pool_size` candidates are scored, so the topical rankers and
page priors still decide what is considered; the verifier decides which of
those actually address the claim. The per-candidate predictions are kept
("verifier_prediction") so the final verification step can reuse them.
"""


class VerifierReranker:
    def __init__(self, verifier, weight=5.0, pool_size=30, score_field="final_score"):
        self.verifier = verifier
        self.weight = weight
        self.pool_size = pool_size
        self.score_field = score_field

    def rerank(self, claim, candidates):
        candidates = [dict(c) for c in candidates]
        pool = [
            c for c in candidates[: self.pool_size]
            if isinstance(c.get("text"), str) and c["text"].strip()
        ]
        rest = candidates[self.pool_size:]
        if not pool:
            return candidates

        predictions = self.verifier.verify_batch(
            [{"claim": claim, "evidence": [c["text"]]} for c in pool]
        )
        for candidate, prediction in zip(pool, predictions):
            probabilities = prediction["probabilities"]
            relevance = float(probabilities["SUPPORTS"]) + float(probabilities["REFUTES"])
            candidate["verifier_relevance"] = relevance
            candidate["verifier_prediction"] = prediction
            candidate[self.score_field] = (
                float(candidate.get(self.score_field, 0.0) or 0.0)
                + self.weight * relevance
            )

        pool.sort(key=lambda item: item[self.score_field], reverse=True)
        ranked = pool + rest
        for rank, item in enumerate(ranked, start=1):
            item["final_rank"] = rank
        return ranked


__all__ = ["VerifierReranker"]
