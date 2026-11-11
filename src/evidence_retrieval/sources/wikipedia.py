import os
import re
import threading
import time
from difflib import SequenceMatcher
from functools import lru_cache

import requests

import src.evidence_retrieval.config  # noqa: F401  (loads .env before reading FACTLENS_CONTACT)
from src.evidence_retrieval.http_retry import request_with_retry
from src.evidence_retrieval.utils.normalizer import normalize_document


WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Wikimedia's API rate limits (2026) allow only 10 requests/min for clients
# it cannot identify, but 200 requests/min for a User-Agent that follows its
# policy: "<client>/<version> (<contact info>) <library>/<version>", where
# contact info is a URL and/or email. Without the contact part a single claim
# exhausts the budget and every later call gets HTTP 429.
# FACTLENS_CONTACT overrides the contact (e.g. "https://your-site; you@uni.edu").
_CONTACT = (
    os.environ.get("FACTLENS_CONTACT", "").strip()
    or "https://github.com/ShivamGoyal1507/Factlens"
)
HEADERS = {
    "User-Agent": (
        f"FactLensBot/1.0 ({_CONTACT}; academic claim verification project) "
        f"python-requests/{requests.__version__}"
    ),
    "Accept-Encoding": "gzip",
}

# FEVER evidence comes from the introductory section of Wikipedia pages,
# which is what the verifier was trained on, so intros are fetched by default.
_MAX_TITLES_PER_REQUEST = 20  # MediaWiki limit for extracts with exintro

_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")


def _page_url(title):
    return "https://en.wikipedia.org/wiki/" + requests.utils.quote(title.replace(" ", "_"))


def base_title(title):
    """'Vedam (film)' -> 'Vedam'."""
    return _PARENTHETICAL.sub("", title or "").strip()


_FUZZY_MIN_RATIO = 0.9
_FUZZY_MIN_CHARS = 8


def title_in_claim(title, claim, fuzzy=True):
    """
    True when the page's base title is mentioned in the claim (plural ok).

    With fuzzy=True a misspelt mention also counts ("Burj Kahlifa" for
    "Burj Khalifa"): a span of the claim with the same number of words must be
    at least 90% similar. Namesakes stay apart ("Rajiv Gandhi" vs
    "Rahul Gandhi" is 75%).
    """
    base = base_title(title)
    if not base or not claim:
        return False
    pattern = (
        r"(?<![A-Za-z0-9])" + re.escape(base.lower()) + r"(?:s|es)?(?![A-Za-z0-9])"
    )
    if re.search(pattern, claim.lower()) is not None:
        return True
    if not fuzzy or len(base) < _FUZZY_MIN_CHARS:
        return False

    target = base.lower()
    size = len(target.split())
    words = re.findall(r"[\w'.&-]+", claim.lower())
    for start in range(0, max(0, len(words) - size) + 1):
        span = " ".join(words[start:start + size]).rstrip(".")
        if SequenceMatcher(None, target, span).ratio() >= _FUZZY_MIN_RATIO:
            return True
    return False


class WikipediaRateLimited(requests.RequestException):
    """Raised instead of calling the API while Wikipedia is rate-limiting us."""


# Wikimedia's etiquette: at most ~3 concurrent requests, stay under the rate
# limit (200/min for an identified User-Agent) and back off on HTTP 429.
# Thread-safe: send times are reserved under a lock (one request per
# _MIN_REQUEST_INTERVAL across all threads), but the HTTP round trip happens
# outside it, so up to _MAX_IN_FLIGHT requests overlap. Identical requests
# are answered from an in-process cache, and after a 429 Wikipedia is skipped
# for the Retry-After period instead of being hit again.
_MIN_REQUEST_INTERVAL = float(os.environ.get("FACTLENS_WIKI_MIN_INTERVAL", "0.35"))
_MAX_IN_FLIGHT = max(1, int(os.environ.get("FACTLENS_WIKI_MAX_CONCURRENCY", "3")))
_DEFAULT_COOLDOWN = 60.0
_CACHE_MAX = 2048

_thread_local = threading.local()
_lock = threading.Lock()
_in_flight = threading.BoundedSemaphore(_MAX_IN_FLIGHT)
_state = {"last_request": 0.0, "blocked_until": 0.0}
_cache = {}


def _session():
    """One HTTP session per thread (requests.Session is not thread-safe)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return session


def reset_wikipedia_state():
    """Clear the response cache and any rate-limit cooldown (tests, long runs)."""
    _cache.clear()
    _state["last_request"] = 0.0
    _state["blocked_until"] = 0.0


def _cooldown_remaining():
    return max(0.0, _state["blocked_until"] - time.time())


def _raise_if_cooling_down():
    remaining = _cooldown_remaining()
    if remaining:
        raise WikipediaRateLimited(
            f"Wikipedia skipped (rate-limited, {int(remaining)}s left)"
        )


def _reserve_send_slot():
    """Wait for this thread's turn so all threads together respect the rate."""
    with _lock:
        _raise_if_cooling_down()
        slot = max(time.monotonic(), _state["last_request"] + _MIN_REQUEST_INTERVAL)
        _state["last_request"] = slot
    delay = slot - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    _raise_if_cooling_down()  # a 429 may have arrived while waiting


def _start_cooldown(response):
    try:
        seconds = float(response.headers.get("Retry-After", _DEFAULT_COOLDOWN))
    except (TypeError, ValueError):
        seconds = _DEFAULT_COOLDOWN
    seconds = max(seconds, 1.0)
    _state["blocked_until"] = time.time() + seconds
    print(
        f"Wikipedia is rate-limiting this IP (HTTP {response.status_code}); "
        f"skipping Wikipedia for {int(seconds)}s."
    )


def _api_get(params):
    params = {**params, "format": "json"}
    key = tuple(sorted((k, str(v)) for k, v in params.items()))
    if key in _cache:
        return _cache[key]

    # Take a concurrency slot first, then the send time: reserving the time
    # first would let requests that waited for a slot go out in a burst.
    with _in_flight:
        _reserve_send_slot()
        response = request_with_retry(
            WIKIPEDIA_API,
            params=params,
            timeout=20,
            max_retries=0,  # single attempt: no repeated API hits on failure
            session=_session(),
        )
    if response.status_code in (429, 503):
        with _lock:
            if not _cooldown_remaining():  # report once, not per thread
                _start_cooldown(response)
    response.raise_for_status()
    data = response.json()

    if len(_cache) >= _CACHE_MAX:
        _cache.clear()
    _cache[key] = data
    return data


def _search_titles(query, limit):
    """
    Wikipedia full-text search. When nothing matches (typically a misspelt
    name such as "Burj Kahlifa"), Wikipedia's own spelling suggestion is
    searched instead - one extra request, only in that case.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srinfo": "suggestion",
        "srprop": "",
    }
    data = _api_get(params)
    query_data = data.get("query", {})
    titles = [item["title"] for item in query_data.get("search", [])]
    suggestion = (query_data.get("searchinfo") or {}).get("suggestion")
    if not titles and suggestion and suggestion.strip().lower() != query.strip().lower():
        data = _api_get({**params, "srsearch": suggestion, "srinfo": ""})
        titles = [item["title"] for item in data.get("query", {}).get("search", [])]
    return titles


# Sections at the end of an article that hold no claim evidence.
_TRAILING_SECTIONS = re.compile(
    r"^==+\s*(?:See also|References|Notes|External links|Further reading|"
    r"Bibliography|Sources|Citations|Footnotes|Works cited)\s*==+\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING = re.compile(r"^==+[^=\n]*==+\s*$", re.MULTILINE)
FULL_TEXT_MAX_CHARS = int(os.environ.get("FACTLENS_WIKI_FULL_TEXT_CHARS", "20000"))


def clean_full_text(text, max_chars=FULL_TEXT_MAX_CHARS):
    """Drop reference/link sections and heading lines; cap the length."""
    match = _TRAILING_SECTIONS.search(text or "")
    if match:
        text = text[:match.start()]
    text = _HEADING.sub("", text or "")
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) > max_chars:
        cut = text.rfind(". ", 0, max_chars)
        text = text[: cut + 1 if cut > 0 else max_chars]
    return text


def fetch_full_text(title):
    """
    Full plain-text article (MediaWiki returns only one full extract per
    request, so this is one request per page). Returns "" on failure.
    """
    try:
        data = _api_get({
            "action": "query",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
            "exsectionformat": "wiki",
            "redirects": 1,
        })
    except (requests.RequestException, ValueError, KeyError):
        return ""
    for page in data.get("query", {}).get("pages", {}).values():
        if "missing" not in page:
            return clean_full_text(page.get("extract", ""))
    return ""


def fetch_pages(titles, intro_only=True):
    """
    Fetch page extracts by title (redirects followed, disambiguation pages
    dropped). Returns normalized documents in the order of `titles`.
    """
    titles = list(dict.fromkeys(t for t in titles if isinstance(t, str) and t.strip()))
    if not titles:
        return []

    pages_by_title = {}
    redirects = {}
    for start in range(0, len(titles), _MAX_TITLES_PER_REQUEST):
        chunk = titles[start:start + _MAX_TITLES_PER_REQUEST]
        params = {
            "action": "query",
            "titles": "|".join(chunk),
            "prop": "extracts|pageprops",
            "ppprop": "disambiguation",
            "explaintext": True,
            "exlimit": "max",
            "redirects": 1,
        }
        if intro_only:
            params["exintro"] = True
        data = _api_get(params)
        query = data.get("query", {})
        for mapping in query.get("normalized", []) + query.get("redirects", []):
            redirects[mapping["from"]] = mapping["to"]
        for page in query.get("pages", {}).values():
            if "missing" in page or "invalid" in page:
                continue
            if "disambiguation" in page.get("pageprops", {}):
                continue
            pages_by_title[page.get("title")] = page

    documents, seen = [], set()
    for title in titles:
        resolved = title
        for _ in range(3):  # normalized -> redirect chains
            resolved = redirects.get(resolved, resolved)
        page = pages_by_title.get(resolved)
        if page is None or resolved in seen:
            continue
        seen.add(resolved)
        document = normalize_document(
            {
                "title": resolved,
                "url": _page_url(resolved),
                "source": "Wikipedia",
                "date": None,
                "seen_date": None,
                "text": page.get("extract", ""),
            },
            source_hint="Wikipedia",
        )
        if document is not None and document["text"].strip():
            documents.append(document)
    return documents


def search_wikipedia(query, limit=5):
    try:
        titles = _search_titles(query, limit)
        return fetch_pages(titles)
    except WikipediaRateLimited:
        return []
    except (requests.RequestException, ValueError, KeyError) as error:
        print(f"Wikipedia retrieval failed: {error}")
        return []


# ---------------------------------------------------------------------------
# Claim-driven page retrieval (FEVER-style document retrieval)
# ---------------------------------------------------------------------------

_LEADING_STOP = {"the", "a", "an"}
_VERB_SPLIT = re.compile(
    r"\s+(?:is|was|are|were|has|had|have|does|did|can|could|will|would|"
    r"became|becomes|starred|stars|played|plays|won|wrote|directed|"
    r"received|released|died|lived|only|never|did not|was not|is not)\b",
    re.IGNORECASE,
)


_NLP_LOCK = threading.Lock()  # spaCy pipelines are not guaranteed thread-safe


@lru_cache(maxsize=1)
def _default_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return None


def claim_mentions(claim, entities=None, nlp=None):
    """
    Candidate page titles mentioned in a claim, most specific first:
    the claim subject, named entities, capitalised spans and noun chunks.
    Uses spaCy (en_core_web_sm) when available, regexes otherwise.
    """
    claim = (claim or "").strip().rstrip(".")
    mentions = []
    nlp = nlp if nlp is not None else _default_nlp()
    if nlp is not None:
        with _NLP_LOCK:
            doc = nlp(claim)
    else:
        doc = None

    subject = _VERB_SPLIT.split(claim, maxsplit=1)[0].strip()
    if subject and subject != claim and len(subject.split()) <= 8:
        mentions.append(subject)

    if doc is not None:
        # Grammatical subject: "Martial arts films" in "Martial arts films
        # exclude fights ..." (no verb list needed).
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ in {"nsubj", "nsubjpass"}:
                mentions.append(chunk.text)
        mentions.extend(ent.text for ent in doc.ents)

    mentions.extend(e for e in (entities or []) if isinstance(e, str))

    mentions.extend(
        re.findall(r"\b[A-Z0-9][\w'.&-]*(?:\s+(?:of|the|and|de|in|for|[A-Z0-9][\w'.&-]*))*", claim)
    )

    if doc is not None:
        mentions.extend(chunk.text for chunk in doc.noun_chunks)

    cleaned, seen = [], set()
    for mention in mentions:
        mention = re.sub(r"\s+(?:of|the|and|de|in|for)$", "", mention.strip(" ,;:'\""))
        if mention.count("(") != mention.count(")"):
            mention = mention.replace("(", "").replace(")", "").strip()
        words = mention.split()
        if words and words[0].lower() in _LEADING_STOP:
            mention = " ".join(words[1:])
        key = mention.lower()
        if len(mention) < 2 or key in seen or key in _LEADING_STOP:
            continue
        seen.add(key)
        cleaned.append(mention)
        # "Martial arts films" -> also look up "Martial arts film".
        if re.search(r"[a-z]s$", mention) and not re.search(r"(?:ss|us|is)$", mention):
            singular = mention[:-1]
            if singular.lower() not in seen:
                seen.add(singular.lower())
                cleaned.append(singular)
    return cleaned[:14]


FULL_TEXT_PAGES = int(os.environ.get("FACTLENS_WIKI_FULL_TEXT_PAGES", "2"))


def search_wikipedia_for_claim(
    claim, entities=None, nlp=None, per_mention=5, max_pages=12, max_mention_searches=3,
    full_text_pages=None,
):
    """
    Retrieve the Wikipedia pages a claim is about.

    FEVER gold evidence almost always sits on a page whose title is mentioned
    in the claim ("Vedam (film)" for "Vedam is a 2010 Indian drama film"), so:
      1. direct title lookup of every mention (one batched request,
         redirects followed);
      2. a Wikipedia search for the full claim and for the first
         `max_mention_searches` mentions (the subject and main entities),
         keeping pages whose base title appears in the claim, plus the top
         hits for the full claim.
      3. for the first `full_text_pages` pages named in the claim, the whole
         article instead of the intro: many facts ("grew up in Houston",
         "can return synonyms") are only in the article body.
    Pages whose title is in the claim come first. Requests are sequential:
    about 1 + max_mention_searches searches, one batched page fetch and
    `full_text_pages` full-article fetches.
    """
    if full_text_pages is None:
        full_text_pages = FULL_TEXT_PAGES
    try:
        mentions = claim_mentions(claim, entities=entities, nlp=nlp)
        searches = [claim] + mentions[:max_mention_searches]
        results = [_safe_search_titles(q, per_mention) for q in searches]

        claim_hits = results[0]
        candidates = list(mentions)
        for titles in results:
            candidates.extend(t for t in titles if title_in_claim(t, claim))
        candidates.extend(claim_hits[:3])

        documents = fetch_pages(candidates)
        documents.sort(key=lambda d: not title_in_claim(d["title"], claim))
        documents = documents[:max_pages]

        named = [d for d in documents if title_in_claim(d["title"], claim)]
        for document in named[:full_text_pages]:
            full_text = fetch_full_text(document["title"])
            if len(full_text) > len(document["text"]):
                document["text"] = full_text
        return documents
    except WikipediaRateLimited:
        return []
    except (requests.RequestException, ValueError, KeyError) as error:
        print(f"Wikipedia claim retrieval failed: {error}")
        return []


def _safe_search_titles(query, limit):
    try:
        return _search_titles(query, limit)
    except WikipediaRateLimited:
        return []
    except (requests.RequestException, ValueError, KeyError) as error:
        print(f"Wikipedia search failed for {query!r}: {error}")
        return []
