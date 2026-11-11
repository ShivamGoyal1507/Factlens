from src.ranking.candidate_merger import CandidateMerger
from src.ranking.entity_scorer import EntityScorer
from src.ranking.evidence_selector import EvidenceSelector


def _p(title, text, score, url=None, source="Wikipedia"):
    return {
        "title": title,
        "text": text,
        "rerank_score": score,
        "url": url or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        "source": source,
    }


def test_entity_scorer_prefers_subject_over_namesake():
    candidates = [
        _p("Rajiv Gandhi", "Rajiv Gandhi served as the prime minister of India.", 8.8),
        _p("Rahul Gandhi", "Rahul Gandhi is the leader of the Opposition in India.", 7.3),
        _p("Prime Minister of India", "Narendra Modi is the current prime minister of India.", 6.6),
    ]

    ranked = EntityScorer().score(candidates, ["Rahul Gandhi", "India"])
    titles = [c["title"] for c in ranked]

    assert titles[0] == "Rahul Gandhi"
    assert titles[-1] == "Rajiv Gandhi"
    assert ranked[-1]["entity_score"] < 0


def test_entity_scorer_without_entities_keeps_order():
    candidates = [_p("A", "x", 2.0), _p("B", "y", 5.0)]
    ranked = EntityScorer().score(candidates, [])
    assert [c["title"] for c in ranked] == ["B", "A"]
    assert all(c["entity_score"] == 0 for c in ranked)


def test_selector_group_by_url_allows_several_wikipedia_articles():
    candidates = [
        _p("Rahul Gandhi", "text one", 9.0),
        _p("Prime Minister of India", "text two", 8.0),
        _p("Prime Minister of India", "text three", 7.0),
    ]

    by_source = EvidenceSelector(max_per_source=1).select(candidates, top_k=5)
    by_url = EvidenceSelector(max_per_source=1, group_by="url").select(candidates, top_k=5)

    assert len(by_source) == 1
    assert [c["title"] for c in by_url] == ["Rahul Gandhi", "Prime Minister of India"]


def test_merge_many_fuses_multiple_rankings():
    a = {"passage_id": "a", "text": "a"}
    b = {"passage_id": "b", "text": "b"}
    c = {"passage_id": "c", "text": "c"}

    merged = CandidateMerger().merge_many(
        [("bm25:claim", [a, b]), ("bm25:q1", [c, b]), ("dense:q1", [b])],
        top_k=10,
    )

    assert merged[0]["passage_id"] == "b"
    assert {m["passage_id"] for m in merged} == {"a", "b", "c"}


def test_merge_still_supports_two_lists():
    a = {"passage_id": "a", "text": "a"}
    merged = CandidateMerger().merge([a], [a], top_k=5)
    assert merged[0]["retrieval_methods"] == ["bm25", "dense"]
