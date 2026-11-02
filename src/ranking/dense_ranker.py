import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRanker:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        device=None
    ):
        self.model = SentenceTransformer(
            model_name,
            device=device
        )
        self.batch_size = batch_size
        self.passages = []
        self.embeddings = None

    def fit(self, passages):
        self.passages = list(passages)

        texts = [
            passage.get("text", "")
            if isinstance(passage.get("text", ""), str)
            else ""
            for passage in self.passages
        ]

        if not texts:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return self

        self.embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )

        return self

    def rank(self, query, top_k=20):
        if top_k <= 0:
            return []

        if not self.passages or self.embeddings is None:
            return []

        if not isinstance(query, str) or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = np.dot(self.embeddings, query_embedding)

        order = np.argsort(-scores, kind="stable")[:top_k]
        results = []

        for index in order:
            result = dict(self.passages[index])
            result["dense_score"] = float(scores[index])
            result["dense_rank"] = len(results) + 1
            results.append(result)

        return results