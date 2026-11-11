import requests

from src.evidence_retrieval.sources import gdelt


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"articles": []}
        self.headers = {}
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _counting(monkeypatch, response=None, error=None):
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(kwargs)
        if error is not None:
            raise error
        return response or _Response()

    monkeypatch.setattr(gdelt, "request_with_retry", fake_request)
    return calls


def test_requests_are_not_retried(monkeypatch):
    calls = _counting(monkeypatch)
    gdelt.search_gdelt("AI")
    assert calls[0]["max_retries"] == 0


def test_http_429_starts_cooldown_and_skips_next_calls(monkeypatch):
    calls = _counting(monkeypatch, response=_Response(status_code=429))

    assert gdelt.search_gdelt("first") == []
    assert gdelt.search_gdelt("second") == []
    assert gdelt.search_gdelt("third") == []

    assert len(calls) == 1  # only the first query actually reached GDELT


def test_connection_reset_starts_cooldown(monkeypatch):
    calls = _counting(
        monkeypatch,
        error=requests.ConnectionError("Connection reset by peer"),
    )

    assert gdelt.search_gdelt("first") == []
    assert gdelt.search_gdelt("second") == []
    assert len(calls) == 1


def test_disable_switch(monkeypatch):
    calls = _counting(monkeypatch)
    monkeypatch.setenv("FACTLENS_DISABLE_GDELT", "1")

    assert gdelt.search_gdelt("AI") == []
    assert calls == []


def test_requests_are_spaced_by_min_interval(monkeypatch):
    _counting(monkeypatch)
    sleeps = []
    clock = {"now": 1000.0}

    monkeypatch.setattr(gdelt, "MIN_INTERVAL_SECONDS", 5.5)
    monkeypatch.setattr(gdelt.time, "time", lambda: clock["now"])
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: sleeps.append(s))

    gdelt.search_gdelt("first")
    clock["now"] += 1.0
    gdelt.search_gdelt("second")

    assert sleeps == [4.5]


def test_queries_are_limited_to_english_sources(monkeypatch):
    calls = _counting(monkeypatch)
    gdelt.search_gdelt("Rahul Gandhi")
    assert calls[0]["params"]["query"] == "Rahul Gandhi sourcelang:english"


def test_cooldown_is_shared_across_runs_via_state_file(monkeypatch, tmp_path):
    monkeypatch.setattr(gdelt, "STATE_FILE", tmp_path / "gdelt_state.json")
    calls = _counting(monkeypatch, response=_Response(status_code=429))

    gdelt.search_gdelt("first run")
    assert len(calls) == 1

    # Simulate a brand-new `python main.py` process: memory is empty,
    # only the state file remains.
    gdelt._state.update(gdelt._EMPTY_STATE)
    assert gdelt.search_gdelt("second run") == []
    assert len(calls) == 1


def test_cooldown_doubles_on_repeated_blocks_and_resets_on_success(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(gdelt.time, "time", lambda: clock["now"])
    monkeypatch.setattr(gdelt, "COOLDOWN_SECONDS", 300.0)
    _counting(monkeypatch, response=_Response(status_code=429))

    gdelt.search_gdelt("a")
    assert gdelt._state["blocked_until"] == 1300.0

    clock["now"] = 1301.0
    gdelt.search_gdelt("b")
    assert gdelt._state["blocked_until"] == 1301.0 + 600.0

    clock["now"] = 5000.0
    _counting(monkeypatch)  # GDELT answers normally again
    gdelt.search_gdelt("c")
    assert gdelt._state["block_count"] == 0
