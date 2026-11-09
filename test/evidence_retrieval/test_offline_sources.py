import requests
import pytest
from unittest.mock import Mock
from datetime import datetime, timezone

from src.evidence_retrieval.sources.wikipedia import search_wikipedia
from src.evidence_retrieval.sources.gdelt import search_gdelt
from src.evidence_retrieval.sources.newsapi import search_newsapi
from src.evidence_retrieval.retriever import (
    validate_documents,
    remove_duplicates,
    get_text_ready_documents,
    filter_by_claim_date,
    prepare_for_ranking,
)
from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.date_filter import (
    document_is_temporally_eligible,
    extract_claim_date_constraint,
    get_temporal_cutoff_for_claim,
)


def _mock_response(status_code=200, payload=None, headers=None, exc=None):
    response = Mock()
    response.status_code = status_code
    response.headers = headers or {}
    response.json.return_value = payload or {}
    response.raise_for_status.side_effect = exc
    return response


def test_wikipedia_success(monkeypatch):
    payload = {
        "query": {
            "search": [{"pageid": 42, "title": "AI"}]
        }
    }
    extract_payload = {
        "query": {
            "pages": {
                "42": {
                    "title": "AI",
                    "extract": "Artificial intelligence is a field of computer science."
                }
            }
        }
    }

    responses = [
        _mock_response(200, payload),
        _mock_response(200, extract_payload),
    ]

    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.wikipedia.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))

    results = search_wikipedia("AI", limit=1)
    assert len(results) == 1
    assert results[0]["title"] == "AI"
    assert results[0]["source"] == "Wikipedia"
    assert "Artificial intelligence" in results[0]["text"]


def test_gdelt_success(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "AI growth",
                "url": "https://example.com/ai",
                "seendate": "20240101T120000Z",
            }
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        return _mock_response(200, payload)

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.gdelt.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))

    results = search_gdelt("AI", limit=5)
    assert len(results) == 1
    assert results[0]["source"] == "GDELT"
    assert results[0]["seen_date"] == "20240101T120000Z"
    assert results[0]["date"] is None
    assert results[0]["text"] == ""


def test_newsapi_success(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "AI article",
                "url": "https://example.com/news",
                "publishedAt": "2024-01-02T00:00:00Z",
                "content": "AI is becoming widespread.",
                "description": "description",
            }
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        return _mock_response(200, payload)

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.newsapi.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))
    monkeypatch.setattr("src.evidence_retrieval.sources.newsapi.get_newsapi_api_key", lambda: "test-key")

    results = search_newsapi("AI", limit=5)
    assert len(results) == 1
    assert results[0]["source"] == "NewsAPI"
    assert results[0]["date"] == "2024-01-02T00:00:00Z"
    assert "AI is becoming widespread" in results[0]["text"]


def test_malformed_response_is_ignored(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        response = _mock_response(200, {})
        response.json.side_effect = ValueError("bad json")
        return response

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.wikipedia.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))

    assert search_wikipedia("AI") == []


def test_timeout_is_handled(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        raise requests.Timeout("timeout")

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.wikipedia.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))

    assert search_wikipedia("AI") == []


def test_http_error_retry_behavior(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            response = _mock_response(429, {}, {"Retry-After": "0"})
            response.raise_for_status.side_effect = requests.HTTPError("429")
            return response
        return _mock_response(200, {"articles": []})

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.newsapi.request_with_retry", lambda *args, **kwargs: request_with_retry(*args, **kwargs))
    monkeypatch.setattr("src.evidence_retrieval.sources.newsapi.get_newsapi_api_key", lambda: "test-key")

    results = search_newsapi("AI")
    assert results == []
    assert calls["count"] >= 3


def test_metadata_only_gdelt_document_is_kept(monkeypatch):
    payload = {
        "articles": [
            {
                "title": "GDELT metadata only",
                "url": "https://example.com/gdelt",
                "seendate": "20240101T120000Z",
            }
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=None, max_retries=None, status_codes_to_retry=None, session=None, **kwargs):
        return _mock_response(200, payload)

    monkeypatch.setattr("src.evidence_retrieval.http_retry.requests.get", fake_get)
    monkeypatch.setattr("src.evidence_retrieval.sources.gdelt.request_with_retry", lambda *args, **kwargs: fake_get(*args, **kwargs))

    results = search_gdelt("AI")
    assert len(results) == 1
    assert results[0]["text"] == ""
    assert results[0]["seen_date"] == "20240101T120000Z"
    assert results[0]["date"] is None


def test_validation_and_deduplication():
    documents = [
        {
            "title": "AI in India",
            "url": "https://example.com/1",
            "source": "NewsAPI",
            "date": "2024-01-01",
            "seen_date": None,
            "text": "India is developing AI.",
        },
        {
            "title": "",
            "url": "https://example.com/2",
            "source": "Wikipedia",
            "date": None,
            "seen_date": None,
            "text": "invalid",
        },
        {
            "title": "AI duplicate",
            "url": "https://example.com/1",
            "source": "GDELT",
            "date": None,
            "seen_date": "20240101T120000Z",
            "text": "",
        },
        {
            "title": "Metadata doc",
            "url": "https://example.com/3",
            "source": "GDELT",
            "date": None,
            "seen_date": "20240102T120000Z",
            "text": "",
        },
    ]

    valid = validate_documents(documents)
    assert len(valid) == 3
    unique = remove_duplicates(valid)
    assert len(unique) == 2

    text_ready = get_text_ready_documents(valid)
    assert len(text_ready) == 1
    assert text_ready[0]["title"] == "AI in India"


def test_missing_dates_and_empty_text_are_allowed():
    documents = [
        {
            "title": "AI growth",
            "url": "https://example.com/4",
            "source": "Wikipedia",
            "date": None,
            "seen_date": None,
            "text": "",
        },
        {
            "title": "AI report",
            "url": "https://example.com/5",
            "source": "GDELT",
            "date": None,
            "seen_date": "20240103T050000Z",
            "text": "",
        },
    ]

    valid = validate_documents(documents)
    assert len(valid) == 2
    assert valid[0]["text"] == ""
    assert valid[1]["seen_date"] == "20240103T050000Z"


def test_exact_claim_timestamp_filters_before_and_after():
    query = "Claim made at 2026-11-01T10:30:00+05:30"
    claim_cutoff = get_temporal_cutoff_for_claim(query)
    assert claim_cutoff is not None
    assert claim_cutoff.isoformat() == "2026-11-01T05:00:00+00:00"

    documents = [
        {"title": "Before", "url": "https://example.com/before", "source": "NewsAPI", "date": "2026-11-01T03:59:59Z", "seen_date": None, "text": "before"},
        {"title": "Exact", "url": "https://example.com/exact", "source": "NewsAPI", "date": "2026-11-01T05:00:00Z", "seen_date": None, "text": "exact"},
        {"title": "One second after", "url": "https://example.com/after", "source": "NewsAPI", "date": "2026-11-01T05:00:01Z", "seen_date": None, "text": "after"},
        {"title": "Later same day", "url": "https://example.com/later_same_day", "source": "NewsAPI", "date": "2026-11-01T12:00:00Z", "seen_date": None, "text": "later"},
    ]

    filtered = filter_by_claim_date(documents, query)
    assert [doc["title"] for doc in filtered] == ["Before", "Exact"]


def test_date_only_claim_filters_by_end_of_claim_day():
    query = "Claim made on 2026-11-01"
    cutoff = get_temporal_cutoff_for_claim(query)
    assert cutoff is not None
    assert cutoff.isoformat() == "2026-11-01T23:59:59.999999+00:00"

    documents = [
        {"title": "On claim date", "url": "https://example.com/on-date", "source": "NewsAPI", "date": "2026-11-01T12:00:00Z", "seen_date": None, "text": "on date"},
        {"title": "Next day", "url": "https://example.com/next-day", "source": "NewsAPI", "date": "2026-11-02T00:00:00Z", "seen_date": None, "text": "next day"},
        {"title": "Same day but later", "url": "https://example.com/same-day-later", "source": "NewsAPI", "date": "2026-11-01T23:59:59.999999Z", "seen_date": None, "text": "later same day"},
    ]

    filtered = filter_by_claim_date(documents, query)
    assert [doc["title"] for doc in filtered] == ["On claim date", "Same day but later"]


def test_missing_or_invalid_publication_timestamps_are_excluded_when_filtering_active():
    query = "Claim date: 2026-11-01"
    documents = [
        {"title": "Missing date", "url": "https://example.com/missing", "source": "NewsAPI", "date": None, "seen_date": None, "text": "missing"},
        {"title": "Invalid date", "url": "https://example.com/invalid", "source": "NewsAPI", "date": "not-a-date", "seen_date": None, "text": "invalid"},
        {"title": "Valid date", "url": "https://example.com/valid", "source": "NewsAPI", "date": "2026-11-01T12:00:00Z", "seen_date": None, "text": "ok"},
    ]

    filtered = filter_by_claim_date(documents, query)
    assert [doc["title"] for doc in filtered] == ["Valid date"]


def test_gdelt_seen_date_is_never_used_as_publication_timestamp():
    query = "Claim made on 2026-11-01"
    documents = [
        {"title": "GDELT metadata only", "url": "https://example.com/gdelt", "source": "GDELT", "date": None, "seen_date": "20261101T120000Z", "text": ""},
        {"title": "Real article", "url": "https://example.com/gdelt-real", "source": "GDELT", "date": "2026-11-01T10:00:00Z", "seen_date": "20261101T120000Z", "text": "content"},
    ]

    filtered = filter_by_claim_date(documents, query)
    assert [doc["title"] for doc in filtered] == ["Real article"]


def test_wikipedia_revision_times_are_never_used_as_publication_timestamps():
    query = "Claim made on 2026-11-01"
    documents = [
        {"title": "Wikipedia page", "url": "https://en.wikipedia.org/wiki/Test", "source": "Wikipedia", "date": None, "seen_date": None, "text": "content"},
        {"title": "Wikipedia updated after claim", "url": "https://en.wikipedia.org/wiki/Test2", "source": "Wikipedia", "date": "2026-11-02T00:00:00Z", "seen_date": None, "text": "content"},
    ]

    filtered = filter_by_claim_date(documents, query)
    assert [doc["title"] for doc in filtered] == []


def test_newsapi_published_at_is_normalized_and_used():
    document = {
        "title": "NewsAPI doc",
        "url": "https://example.com/newsapi",
        "source": "NewsAPI",
        "date": "2024-01-02T00:00:00Z",
        "seen_date": None,
        "text": "content",
    }

    cutoff = get_temporal_cutoff_for_claim("Claim made at 2024-01-02T00:00:00Z")
    assert cutoff is not None
    assert document_is_temporally_eligible(document, cutoff)


def test_ambiguous_or_unsupported_claim_dates_do_not_trigger_filtering():
    assert get_temporal_cutoff_for_claim("The claim was mentioned in 2026") is None
    assert get_temporal_cutoff_for_claim("Some rumor happened before summer") is None
    constraint = extract_claim_date_constraint("Some rumor happened before summer")
    assert constraint["precision"] == "unavailable"


def test_no_claim_date_constraint_preserves_existing_retrieval_behavior():
    documents = [
        {"title": "Doc 1", "url": "https://example.com/1", "source": "NewsAPI", "date": "2026-11-02T00:00:00Z", "seen_date": None, "text": "text"},
        {"title": "Doc 2", "url": "https://example.com/2", "source": "NewsAPI", "date": "2026-11-01T00:00:00Z", "seen_date": None, "text": "text"},
    ]
    assert filter_by_claim_date(documents, "AI breakthroughs in India") == documents


def test_prepare_for_ranking_remains_compatible_after_temporal_filtering():
    documents = [
        {"title": "Doc", "url": "https://example.com/doc", "source": "NewsAPI", "date": "2026-11-01T12:00:00Z", "seen_date": None, "text": "body text"},
    ]
    prepared = prepare_for_ranking(documents)
    assert len(prepared) == 1
    assert prepared[0]["published_date"] == "2026-11-01T12:00:00Z"
    assert prepared[0]["date"] == "2026-11-01T12:00:00Z"
    assert prepared[0]["text"] == "body text"


@pytest.mark.live
def test_live_newsapi_placeholder():
    pytest.skip("Live API test skipped by default; enable with --run-live")