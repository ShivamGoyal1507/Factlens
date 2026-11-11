from unittest.mock import MagicMock

from src.evidence_retrieval.sources import newsapi, wikipedia


def _response(status, payload=None, headers=None):
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.json.return_value = payload or {}
    if status >= 400:
        import requests

        response.raise_for_status.side_effect = requests.HTTPError(str(status))
    return response


def test_wikipedia_429_starts_cooldown_and_skips_further_calls(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(kwargs.get("params"))
        return _response(429, headers={"Retry-After": "120"})

    monkeypatch.setattr(wikipedia, "request_with_retry", fake_request)

    assert wikipedia.search_wikipedia("Rahul Gandhi") == []
    assert wikipedia.search_wikipedia("Prime Minister of India") == []
    assert wikipedia.search_wikipedia_for_claim("Rahul Gandhi is prime minister of India.") == []
    assert len(calls) == 1  # only the first request hit the API
    assert 100 < wikipedia._cooldown_remaining() <= 120


def test_wikipedia_identical_requests_are_cached(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(kwargs.get("params"))
        return _response(200, {"query": {"search": [{"title": "Rahul Gandhi"}]}})

    monkeypatch.setattr(wikipedia, "request_with_retry", fake_request)

    assert wikipedia._search_titles("Rahul Gandhi", 5) == ["Rahul Gandhi"]
    assert wikipedia._search_titles("Rahul Gandhi", 5) == ["Rahul Gandhi"]
    assert len(calls) == 1


def test_newsapi_stops_after_quota_429(monkeypatch):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(1)
        return _response(429)

    monkeypatch.setattr(newsapi, "request_with_retry", fake_request)
    monkeypatch.setattr(newsapi, "get_newsapi_api_key", lambda: "test-key")

    assert newsapi.search_newsapi("Rahul Gandhi") == []
    assert newsapi.search_newsapi("India") == []
    assert len(calls) == 1


def test_newsapi_article_text_combines_fields_and_strips_marker():
    text = newsapi.article_text({
        "description": "Narendra Modi was sworn in as prime minister",
        "content": "NEW DELHI: Narendra Modi took oath on Sunday for a third term… [+2345 chars]",
    })
    assert text == (
        "Narendra Modi was sworn in as prime minister. "
        "NEW DELHI: Narendra Modi took oath on Sunday for a third term."
    )
    assert newsapi.article_text({"description": None, "content": None}) == ""
