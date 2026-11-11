from src.evidence_retrieval import retriever


def test_gdelt_headline_becomes_passage_text():
    docs = [
        {
            "title": "Modi sworn in as India's prime minister for a third term",
            "url": "https://example.com/modi",
            "source": "GDELT",
            "date": None,
            "seen_date": "20240609T120000Z",
            "text": "",
        },
        {
            "title": "Empty NewsAPI item",
            "url": "https://example.com/empty",
            "source": "NewsAPI",
            "date": None,
            "seen_date": None,
            "text": "",
        },
    ]

    prepared = retriever.prepare_for_ranking(docs)

    assert len(prepared) == 1
    assert prepared[0]["text"] == docs[0]["title"]
    assert prepared[0]["text_from_title"] is True
    assert docs[0]["text"] == ""  # original document untouched


def test_only_first_queries_hit_gdelt(monkeypatch):
    seen = []

    def fake_single(query, include_gdelt=True):
        seen.append((query, include_gdelt))
        return []

    monkeypatch.setattr(retriever, "_retrieve_evidence_for_single_query", fake_single)
    monkeypatch.setattr(retriever, "GDELT_MAX_QUERIES_PER_CLAIM", 2)

    retriever.retrieve_evidence_for_claim("claim", queries=["q1", "q2", "q3", "q4"])

    assert seen == [("q1", True), ("q2", True), ("q3", False), ("q4", False)]
