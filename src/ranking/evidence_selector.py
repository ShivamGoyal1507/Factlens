import re


class EvidenceSelector:
    def __init__(
        self,
        max_per_source=None,
        min_score=None,
        deduplicate=True,
        group_by="source",
        score_field="rerank_score",
    ):
        """
        group_by:
            "source" - limit counts per source name ("Wikipedia", "NewsAPI").
                       Note: all Wikipedia articles then share ONE slot.
            "url"    - limit counts per article/document.
        score_field: score used for min_score filtering.
        """
        if group_by not in {"source", "url"}:
            raise ValueError("group_by must be 'source' or 'url'")
        if max_per_source is not None and max_per_source < 1:
            raise ValueError("max_per_source must be positive or None")

        self.max_per_source = max_per_source
        self.min_score = min_score
        self.deduplicate = deduplicate
        self.group_by = group_by
        self.score_field = score_field

    @staticmethod
    def normalize_text(text):
        if not isinstance(text, str):
            return ""

        return re.sub(r"\s+", " ", text.lower()).strip()

    def get_group_key(self, passage):
        if self.group_by == "url":
            url = passage.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip().lower()
        return self.get_source_key(passage)

    @staticmethod
    def get_source_key(passage):
        source = passage.get("source")

        if isinstance(source, str) and source.strip():
            return source.strip().lower()

        url = passage.get("url")

        if isinstance(url, str) and url.strip():
            return url.strip().lower()

        return "unknown"

    def select(self, candidates, top_k=5):
        if top_k <= 0:
            return []

        selected = []
        seen_texts = set()
        source_counts = {}

        for candidate in candidates:
            text = candidate.get("text", "")

            if not isinstance(text, str) or not text.strip():
                continue

            normalized_text = self.normalize_text(text)

            if self.deduplicate and (
                not normalized_text or normalized_text in seen_texts
            ):
                continue

            if self.min_score is not None:
                score = candidate.get(self.score_field)

                if score is None or score < self.min_score:
                    continue

            source_key = self.get_group_key(candidate)

            if self.max_per_source is not None:
                if source_counts.get(source_key, 0) >= self.max_per_source:
                    continue

            selected_candidate = dict(candidate)
            selected_candidate["evidence_rank"] = len(selected) + 1
            selected.append(selected_candidate)

            if self.deduplicate:
                seen_texts.add(normalized_text)

            source_counts[source_key] = source_counts.get(source_key, 0) + 1

            if len(selected) >= top_k:
                break

        return selected
