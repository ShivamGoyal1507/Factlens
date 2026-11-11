import hashlib
import re


class CandidateMerger:
    def __init__(self, rrf_k=60):
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        self.rrf_k = rrf_k

    @staticmethod
    def normalize_text(text):
        if not isinstance(text, str):
            return ""

        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def get_key(passage):
        passage_id = passage.get("passage_id")

        if passage_id:
            return str(passage_id)

        text = CandidateMerger.normalize_text(
            passage.get("text", "")
        )

        url = passage.get("url", "")

        return hashlib.sha256(
            f"{url}|{text}".encode("utf-8")
        ).hexdigest()[:16]

    def merge(self, bm25_results, dense_results, top_k=40):
        return self.merge_many(
            [("bm25", bm25_results), ("dense", dense_results)],
            top_k=top_k,
        )

    def merge_many(self, rankings, top_k=40):
        """
        Reciprocal-rank-fuse any number of rankings.

        rankings: list of (method_name, results) pairs, e.g.
            [("bm25", ...), ("dense", ...), ("bm25:q1", ...), ...]
        """
        if top_k <= 0:
            return []

        combined = {}

        for method, results in rankings:
            seen_in_ranking = set()

            for rank, passage in enumerate(results, start=1):
                key = self.get_key(passage)

                if key in seen_in_ranking:
                    continue

                seen_in_ranking.add(key)

                if key not in combined:
                    combined[key] = dict(passage)
                    combined[key]["passage_id"] = key
                    combined[key]["rrf_score"] = 0.0
                    combined[key]["retrieval_methods"] = []
                    combined[key]["retrieval_ranks"] = {}

                combined[key]["rrf_score"] += (
                    1.0 / (self.rrf_k + rank)
                )

                if method not in combined[key]["retrieval_methods"]:
                    combined[key]["retrieval_methods"].append(method)

                combined[key]["retrieval_ranks"][method] = rank

                for field, value in passage.items():
                    if combined[key].get(field) in (None, ""):
                        combined[key][field] = value

        merged = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        for rank, passage in enumerate(merged, start=1):
            passage["rrf_rank"] = rank

        return merged[:top_k]