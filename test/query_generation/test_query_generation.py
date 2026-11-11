import src.query_generation.generator as generator

CLAIM = "India launched its first AI-powered agricultural satellite in 2025."


def test_generate_queries_are_real_search_queries():
    queries = generator.generate_queries(CLAIM)

    assert 3 <= len(queries) <= 5
    assert all(isinstance(q, str) and q.strip() for q in queries)
    assert all("important keywords" not in q.lower() for q in queries)
    assert all("named entities" not in q.lower() for q in queries)
    assert all("key events" not in q.lower() for q in queries)
    assert all("search for subqueries" not in q.lower() for q in queries)


def test_feature_descriptions_are_rejected():
    invalid = [
        "Key events/actions: launched",
        "Important relationships: first occurrence or first launch; AI or artificial intelligence involved",
        "Named entities: India",
        "Search for subqueries related to India",
    ]

    filtered = generator._filter_and_prune_queries(invalid)
    assert filtered == []


def test_duplicate_queries_are_removed():
    queries = generator.generate_queries(CLAIM)
    assert len(queries) == len(set(q.lower() for q in queries))


def test_important_dates_and_entities_are_preserved():
    features = generator.extract_claim_features(CLAIM)
    assert "2025" in features["dates"]
    assert any("India" in entity for entity in features["entities"])

    queries = generator.generate_queries(CLAIM)
    assert any("2025" in q for q in queries)
    assert any("India" in q for q in queries)


def test_fallback_supplements_invalid_model_output(monkeypatch):
    monkeypatch.setattr(generator, "generate_queries_with_llm", lambda claim: [
        "Key events/actions: launched",
        "Important relationships: AI",
        "Search for subqueries",
    ])

    queries = generator.generate_queries(CLAIM)
    assert 3 <= len(queries) <= 5
    assert all("key events" not in q.lower() for q in queries)
    assert all("search for subqueries" not in q.lower() for q in queries)


def test_final_result_has_three_to_five_valid_queries_when_possible():
    queries = generator.generate_queries(CLAIM)
    assert 3 <= len(queries) <= 5


def test_no_real_flant5_loading_in_unit_tests(monkeypatch):
    calls = []

    def fake_loader():
        calls.append("called")
        return None

    monkeypatch.setattr(generator, "_load_flant5_model", fake_loader)
    queries = generator.generate_queries(CLAIM)

    assert queries
    assert calls == [] or calls == ["called"]