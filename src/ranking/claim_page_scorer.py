"""
Page- and position-level priors for FEVER-style evidence.

In FEVER the evidence for a claim almost always comes from a Wikipedia page
whose title the claim mentions ("Vedam (film)" for "Vedam is a 2010 Indian
drama film"), and very often from the page's first sentence, which defines
the subject. The cross-encoder knows neither, so it happily ranks a
topically similar sentence from an unrelated page first.

final_score += page_bonus      if the page's base title is in the claim
             + lead_bonus      if it is also the page's first sentence
"""

from src.evidence_retrieval.sources.wikipedia import title_in_claim


class ClaimPageScorer:
    def __init__(self, page_bonus=1.5, lead_bonus=0.75, score_field="final_score"):
        self.page_bonus = page_bonus
        self.lead_bonus = lead_bonus
        self.score_field = score_field

    def score(self, candidates, claim):
        """Return candidates sorted by the adjusted score (adds page_prior)."""
        scored = []
        for candidate in candidates:
            item = dict(candidate)
            prior = 0.0
            if item.get("source") == "Wikipedia" and title_in_claim(item.get("title", ""), claim):
                prior += self.page_bonus
                if item.get("start_sentence") == 0:
                    prior += self.lead_bonus
            item["page_prior"] = prior
            item[self.score_field] = float(item.get(self.score_field, 0.0) or 0.0) + prior
            scored.append(item)

        scored.sort(key=lambda item: item[self.score_field], reverse=True)
        for rank, item in enumerate(scored, start=1):
            item["final_rank"] = rank
        return scored


__all__ = ["ClaimPageScorer"]
