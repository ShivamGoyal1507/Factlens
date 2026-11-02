import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=16,
        device=None
    ):
        self.model = CrossEncoder(
            model_name,
            device=device
        )
        self.batch_size = batch_size

    def rerank(self, claim, candidates, top_k=20):
        if top_k <= 0:
            return []

        if not isinstance(claim, str) or not claim.strip():
            return []

        if not candidates:
            return []

        valid_candidates = [
            dict(candidate)
            for candidate in candidates
            if isinstance(candidate.get("text"), str)
            and candidate["text"].strip()
        ]

        if not valid_candidates:
            return []

        pairs = [
            (claim, candidate["text"])
            for candidate in valid_candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 100
        )

        scores = np.asarray(scores)

        if scores.ndim > 1:
            scores = scores.reshape(len(valid_candidates), -1)[:, 0]

        for candidate, score in zip(valid_candidates, scores):
            candidate["rerank_score"] = float(score)

        valid_candidates.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        results = valid_candidates[:top_k]

        for rank, candidate in enumerate(results, start=1):
            candidate["rerank_rank"] = rank

        return results