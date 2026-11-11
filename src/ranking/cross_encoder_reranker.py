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
            (claim, candidate.get("rank_text") or candidate["text"])
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

    def rerank_multi(self, texts, candidates, top_k=20):
        """
        Score each candidate against several texts (the claim plus generated
        queries) and keep the best score.

        A passage that answers a neutral query such as "who is the prime
        minister of India" is crucial evidence for "Rahul Gandhi is prime
        minister of India" even though it scores low against the claim.

        Adds "rerank_score" (max), "claim_rerank_score" (texts[0]) and
        "best_query".
        """
        texts = [
            t.strip() for t in (texts or [])
            if isinstance(t, str) and t.strip()
        ]
        if top_k <= 0 or not texts or not candidates:
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
            (text, candidate.get("rank_text") or candidate["text"])
            for candidate in valid_candidates
            for text in texts
        ]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=len(pairs) > 200,
            )
        )
        if scores.ndim > 1:
            scores = scores.reshape(len(pairs), -1)[:, 0]
        scores = scores.reshape(len(valid_candidates), len(texts))

        for candidate, row in zip(valid_candidates, scores):
            best = int(np.argmax(row))
            candidate["rerank_score"] = float(row[best])
            candidate["claim_rerank_score"] = float(row[0])
            candidate["best_query"] = texts[best]

        valid_candidates.sort(key=lambda item: item["rerank_score"], reverse=True)
        results = valid_candidates[:top_k]
        for rank, candidate in enumerate(results, start=1):
            candidate["rerank_rank"] = rank
        return results
