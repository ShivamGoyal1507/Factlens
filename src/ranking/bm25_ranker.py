import re
import numpy as np
from rank_bm25 import BM25Okapi


class BM25Ranker:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.passages = []
        self.tokenized_passages = []
        self.bm25 = None

    @staticmethod
    def tokenize(text):
        if not isinstance(text, str):
            return []

        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, passages):
        self.passages = list(passages)
        self.tokenized_passages = [
            self.tokenize(passage.get("text", ""))
            for passage in self.passages
        ]

        if not self.passages:
            self.bm25 = None
            return self

        self.bm25 = BM25Okapi(
            self.tokenized_passages,
            k1=self.k1,
            b=self.b
        )

        return self

    def rank(self, query, top_k=20):
        if top_k <= 0:
            return []

        if self.bm25 is None or not self.passages:
            return []

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = np.asarray(
            self.bm25.get_scores(query_tokens),
            dtype=np.float64
        )

        order = np.argsort(-scores, kind="stable")[:top_k]
        results = []

        for index in order:
            result = dict(self.passages[index])
            result["bm25_score"] = float(scores[index])
            result["bm25_rank"] = len(results) + 1
            results.append(result)

        return results

    