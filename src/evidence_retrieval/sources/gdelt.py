
import json
import os
import threading
import time
from pathlib import Path

import requests

from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT allows one request every 5 seconds per IP. Exceeding it returns
# HTTP 429, or over HTTPS simply resets the connection
# ("Connection reset by peer"), and every further request while blocked
# extends the block.
MIN_INTERVAL_SECONDS = float(os.environ.get("FACTLENS_GDELT_MIN_INTERVAL", "5.5"))

# Cooldown after a block; doubles on each consecutive block (5, 10, 20, 40,
# 60 min) and resets after a successful request.
COOLDOWN_SECONDS = float(os.environ.get("FACTLENS_GDELT_COOLDOWN", "300"))
MAX_COOLDOWN_SECONDS = float(os.environ.get("FACTLENS_GDELT_MAX_COOLDOWN", "3600"))

# Only English-language sources: other languages cannot be matched against
# English claims by the rankers or the verifier.
QUERY_SUFFIX = os.environ.get("FACTLENS_GDELT_QUERY_SUFFIX", "sourcelang:english")

# Limiter state is saved to disk so separate `python main.py` runs share it.
# Without this every new run pings a GDELT that is still blocking us, which
# keeps the block alive indefinitely.
STATE_FILE = Path(
    os.environ.get(
        "FACTLENS_GDELT_STATE_FILE",
        Path.home() / ".cache" / "factlens" / "gdelt_state.json",
    )
)

_EMPTY_STATE = {"last_request": 0.0, "blocked_until": 0.0, "block_count": 0}
_state = dict(_EMPTY_STATE)
_lock = threading.Lock()


def gdelt_enabled():
    """GDELT can be switched off with FACTLENS_DISABLE_GDELT=1."""
    return os.environ.get("FACTLENS_DISABLE_GDELT", "").strip().lower() not in {
        "1", "true", "yes",
    }


def _load_state():
    if STATE_FILE is None:
        return
    try:
        data = json.loads(Path(STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        for key in _EMPTY_STATE:
            value = data.get(key)
            if isinstance(value, (int, float)):
                # Keep the most restrictive value seen by any process.
                _state[key] = max(_state[key], value)


def _save_state():
    if STATE_FILE is None:
        return
    try:
        path = Path(STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_state), encoding="utf-8")
    except OSError:
        pass


def reset_rate_limiter():
    """Clear limiter state in memory and on disk."""
    with _lock:
        _state.update(_EMPTY_STATE)
        if STATE_FILE is not None:
            try:
                Path(STATE_FILE).unlink()
            except OSError:
                pass


def gdelt_status():
    """Return a short human-readable limiter status."""
    with _lock:
        _load_state()
        remaining = _state["blocked_until"] - time.time()
    if remaining > 0:
        return f"cooling down for {int(remaining)}s more (block #{_state['block_count']})"
    return "ready"


def _wait_for_slot():
    """
    Block until a request is allowed. Returns False when GDELT is
    in cooldown and should be skipped.
    """
    with _lock:
        _load_state()
        now = time.time()
        if now < _state["blocked_until"]:
            return False

        wait = _state["last_request"] + MIN_INTERVAL_SECONDS - now
        if wait > 0:
            time.sleep(wait)

        _state["last_request"] = time.time()
        _save_state()
        return True


def _start_cooldown(reason):
    with _lock:
        _state["block_count"] = int(_state["block_count"]) + 1
        cooldown = min(
            COOLDOWN_SECONDS * (2 ** (_state["block_count"] - 1)),
            MAX_COOLDOWN_SECONDS,
        )
        _state["blocked_until"] = time.time() + cooldown
        _save_state()
    print(
        f"GDELT is rate-limiting this IP ({reason}); skipping GDELT for "
        f"{int(cooldown)}s. Other sources continue normally."
    )


def _record_success():
    with _lock:
        if _state["block_count"] or _state["blocked_until"]:
            _state["block_count"] = 0
            _state["blocked_until"] = 0.0
            _save_state()


def search_gdelt(query, limit=10):
    """
    Search GDELT DOC 2.0 for news articles.
    Returns metadata-only evidence documents.
    """
    if not isinstance(query, str) or not query.strip():
        return []

    full_query = query.strip()
    if QUERY_SUFFIX and QUERY_SUFFIX not in full_query:
        full_query = f"{full_query} {QUERY_SUFFIX}"

    params = {
        "query": full_query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": limit,
        "sort": "HybridRel",
    }

    headers = {
        "User-Agent": "FACTLENS/1.0 (academic claim verification project)"
    }

    if not gdelt_enabled():
        return []

    if not _wait_for_slot():
        print(f"GDELT skipped ({gdelt_status()}).")
        return []

    try:
        # No automatic retries: every retry inside GDELT's 5-second window
        # counts as another violation and extends the block.
        response = request_with_retry(
            GDELT_API,
            params=params,
            headers=headers,
            timeout=30,
            max_retries=0,
        )

        if response.status_code == 429:
            _start_cooldown("HTTP 429")
            return []

        response.raise_for_status()
        _record_success()

        try:
            data = response.json()
        except ValueError as error:
            print(
                "GDELT returned invalid JSON. "
                f"Content-Type: {response.headers.get('Content-Type')}, "
                f"Response preview: {response.text[:200]}"
            )
            raise ValueError("Invalid JSON response from GDELT") from error

        if not isinstance(data, dict):
            print("GDELT returned an unexpected response format.")
            return []

        articles = data.get("articles", [])

        if not isinstance(articles, list):
            print("GDELT response does not contain a valid articles list.")
            return []

        documents = []

        for article in articles:
            if not isinstance(article, dict):
                continue

            doc = {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "source": "GDELT",
                "date": None,
                "seen_date": article.get("seendate"),
                "text": "",
            }

            normalized = normalize_document(
                doc,
                source_hint="GDELT",
            )

            if normalized is not None:
                documents.append(normalized)

        return documents

    except requests.ConnectionError as error:
        # GDELT resets HTTPS connections from rate-limited clients.
        _start_cooldown("connection reset")
        return []

    except (requests.RequestException, ValueError) as error:
        print(f"GDELT retrieval failed: {error}")
        return []
