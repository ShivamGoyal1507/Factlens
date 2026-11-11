from src.evidence_retrieval.sources.wikipedia import (
    base_title,
    claim_mentions,
    title_in_claim,
)
from src.ranking.claim_page_scorer import ClaimPageScorer
from src.ranking.verifier_reranker import VerifierReranker


def test_base_title_strips_disambiguator():
    assert base_title("Vedam (film)") == "Vedam"
    assert base_title("Death Note (2006 TV series)") == "Death Note"
    assert base_title("India") == "India"


def test_title_in_claim_uses_word_boundaries():
    claim = "Vedam is a 2010 Indian drama film."
    assert title_in_claim("Vedam (film)", claim)
    assert not title_in_claim("India", claim)  # "Indian" is not "India"
    assert not title_in_claim("Raavanan (2010 film)", claim)


def test_claim_mentions_starts_with_subject():
    mentions = claim_mentions("Sidse Babett Knudsen is Swedish.")
    assert mentions[0] == "Sidse Babett Knudsen"
    assert "Swedish" in mentions

    mentions = claim_mentions("The Beatles were a rock band.")
    assert "Beatles" in mentions


def _sentence(title, text, score, start=1, source="Wikipedia"):
    return {
        "title": title,
        "text": text,
        "final_score": score,
        "start_sentence": start,
        "source": source,
        "url": f"https://en.wikipedia.org/wiki/{title}",
    }


def test_claim_page_scorer_prefers_pages_named_in_claim():
    claim = "Numenius lived in Egypt."
    ranked = ClaimPageScorer().score(
        [
            _sentence("Moses", "Moses lived in Egypt.", 3.0),
            _sentence(
                "Numenius",
                "Numenius was a Greek philosopher who lived in Syria.",
                2.0,
                start=0,
            ),
        ],
        claim,
    )
    assert ranked[0]["title"] == "Numenius"
    assert ranked[0]["page_prior"] == 2.25
    assert ranked[1]["page_prior"] == 0.0


class _FakeVerifier:
    """Decisive (REFUTES) for sentences mentioning Syria, NEI otherwise."""

    def verify_batch(self, items):
        results = []
        for item in items:
            decisive = "Syria" in item["evidence"][0]
            probabilities = (
                {"SUPPORTS": 0.01, "REFUTES": 0.97, "NEI": 0.02}
                if decisive
                else {"SUPPORTS": 0.02, "REFUTES": 0.03, "NEI": 0.95}
            )
            verdict = max(probabilities, key=probabilities.get)
            results.append({
                "verdict": verdict,
                "confidence": probabilities[verdict],
                "probabilities": probabilities,
            })
        return results


def test_verifier_reranker_promotes_decisive_sentence():
    candidates = [
        _sentence("Numenius", "Numenius wrote several books.", 4.0),
        _sentence("Numenius", "He lived in Apamea in Syria.", 2.0),
        _sentence("Other", "Unscored tail candidate.", 1.0),
    ]
    ranked = VerifierReranker(_FakeVerifier(), weight=5.0, pool_size=2).rerank(
        "Numenius lived in Egypt.", candidates
    )
    assert ranked[0]["text"] == "He lived in Apamea in Syria."
    assert ranked[0]["verifier_relevance"] > 0.9
    assert ranked[-1]["text"] == "Unscored tail candidate."
    assert "verifier_relevance" not in ranked[-1]
    assert [item["final_rank"] for item in ranked] == [1, 2, 3]


def test_plural_mentions_match_singular_titles():
    claim = "Martial arts films exclude fights between multiple characters."
    assert title_in_claim("Martial arts film", claim)
    assert "Martial arts film" in claim_mentions(claim)


def test_fuzzy_title_match_accepts_typos_not_namesakes():
    assert title_in_claim("Burj Khalifa", "57 elevators reside in the Burj Kahlifa.")
    assert not title_in_claim("Rajiv Gandhi", "Rahul Gandhi is prime minister of India.")
    assert not title_in_claim("George H. W. Bush", "George W. Bush was president.")
    assert not title_in_claim("India", "Vedam is a 2010 Indian drama film.")


def test_clean_full_text_drops_headings_and_reference_sections():
    from src.evidence_retrieval.sources.wikipedia import clean_full_text

    text = (
        "Jeb Bush is a politician.\n\n== Early life ==\nBush grew up in Houston.\n"
        "== References ==\nSome citation."
    )
    cleaned = clean_full_text(text)
    assert "grew up in Houston" in cleaned
    assert "Early life" not in cleaned
    assert "citation" not in cleaned
