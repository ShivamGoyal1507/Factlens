"""
Entity-aware adjustment of cross-encoder scores.

The cross-encoder scores topical relevance, so for "Rahul Gandhi is prime
minister of India" it ranks Rajiv Gandhi's page ("served as the prime
minister of India") above Rahul Gandhi's own page. This step corrects that:

* bonus  - the passage's page is about a claim entity (entity in title),
           or the passage text mentions it;
* penalty - the page is about a *namesake*: its title shares part of a
           multi-word claim entity ("Gandhi") but not the full name
           ("Rahul Gandhi"), e.g. "Rajiv Gandhi", "Nehru-Gandhi family";
           or it is the full name plus a generational suffix
           ("Barack Obama Sr.", "... Jr.", "... II").

final_score = rerank_score + entity_score
"""

import re

_STOPWORDS = {
    "the", "of", "and", "in", "on", "for", "a", "an", "to", "at", "by",
    "de", "la", "le", "von", "van", "da",
}


def _contains(text, phrase):
    if not text or not phrase:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(phrase.lower()) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text.lower()) is not None


# "Barack Obama Sr.", "Martin Luther King Jr.", "Henry VIII" ... are other people.
_GENERATIONAL_SUFFIX = re.compile(
    r"^\s*,?\s*(?:sr|jr|senior|junior|i{1,3}|iv|v|vi{1,3})\b\.?",
    re.IGNORECASE,
)


def _is_generational_namesake(title, entity):
    """True when the title is the entity name followed by Sr./Jr./II etc."""
    match = re.search(
        r"(?<![A-Za-z0-9])" + re.escape(entity.lower()) + r"(?![A-Za-z0-9])",
        title.lower(),
    )
    if not match:
        return False
    return bool(_GENERATIONAL_SUFFIX.match(title[match.end():]))


def _name_tokens(entity):
    return [
        t for t in re.findall(r"[A-Za-z0-9]+", entity.lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


class EntityScorer:
    def __init__(
        self,
        title_bonus=2.0,
        text_bonus=1.0,
        namesake_penalty=4.0,
        max_bonus=4.0,
        base_field="rerank_score",
    ):
        self.title_bonus = title_bonus
        self.text_bonus = text_bonus
        self.namesake_penalty = namesake_penalty
        self.max_bonus = max_bonus
        self.base_field = base_field

    def entity_score(self, passage, entities):
        title = passage.get("title", "") or ""
        text = passage.get("text", "") or ""

        bonus = 0.0
        penalty = 0.0
        matched = []

        for entity in entities:
            if _contains(title, entity):
                bonus += self.title_bonus
                matched.append(entity)
            elif _contains(text, entity):
                bonus += self.text_bonus
                matched.append(entity)

        bonus = min(bonus, self.max_bonus)

        # Namesake check for multi-word entities (people, organisations).
        for entity in entities:
            tokens = _name_tokens(entity)
            if len(tokens) < 2:
                continue
            if _is_generational_namesake(title, entity):
                penalty = self.namesake_penalty
                bonus = 0.0
                break
            if _contains(title, entity):
                continue
            title_tokens = set(re.findall(r"[a-z0-9]+", title.lower()))
            if title_tokens & set(tokens):
                penalty = self.namesake_penalty
                break

        return bonus - penalty, matched

    def score(self, candidates, entities):
        """Return candidates sorted by final_score (adds entity fields)."""
        entities = [e for e in (entities or []) if isinstance(e, str) and e.strip()]
        scored = []

        for candidate in candidates:
            item = dict(candidate)
            base = float(item.get(self.base_field, 0.0) or 0.0)
            adjustment, matched = (
                self.entity_score(item, entities) if entities else (0.0, [])
            )
            item["entity_score"] = adjustment
            item["matched_entities"] = matched
            item["final_score"] = base + adjustment
            scored.append(item)

        scored.sort(key=lambda item: item["final_score"], reverse=True)
        for rank, item in enumerate(scored, start=1):
            item["final_rank"] = rank
        return scored


__all__ = ["EntityScorer"]
