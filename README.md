# FACTLENS – Evidence Retrieval Module

## 1. Overview

FACTLENS is an evidence-based claim-verification project. The Evidence Retrieval module is responsible for collecting candidate evidence from multiple external sources, normalizing the results into a common document structure, validating the data, handling transient API failures, and removing duplicates.

Within this project, the Evidence Retrieval layer is intentionally scoped to retrieval-related work only. It does not implement ranking, reranking, or passage splitting.

The module retrieves from three sources:

- Wikipedia
- NewsAPI

It returns a normalized list of documents that can then be passed to downstream FACTLENS components, including the ranking pipeline.

## 2. Responsibilities

The Evidence Retrieval module performs the following tasks:

- Query multiple external sources for evidence
- Normalize heterogeneous raw results into a common schema
- Validate required metadata and content fields
- Preserve source-specific metadata such as `seen_date`
- Apply claim-date-based temporal filtering when a clear claim date is supplied
- Remove duplicate URLs from the final evidence list
- Handle transient network and HTTP failures with retry/backoff
- Keep compatibility with downstream ranking via `prepare_for_ranking()`

## 3. Actual architecture and retrieval flow

The implementation is a sequential pipeline, not a fully parallel worker pool.

```text
User Query / Claim
        |
        v
Query-level claim-date extraction
        |
        v
Wikipedia retrieval
        |
        v
NewsAPI retrieval
        |
        v
Normalize source outputs
        |
        v
Validate documents
        |
        v
Deduplicate results
        |
        v
Apply claim-date filtering when active
        |
        v
Return evidence documents
        |
        v
prepare_for_ranking() for downstream compatibility
```

The Evidence Retrieval module owns the retrieval, normalization, validation, deduplication, and temporal filtering stages. Ranking and reranking live under separate project modules.

## 4. Evidence sources and actual fields retrieved

### 4.1 Wikipedia

The implementation calls the MediaWiki API and retrieves search results plus page extracts.

It keeps the following fields in the normalized document:

- `title`
- `url`
- `source` = `Wikipedia`
- `date` = `None` unless a genuine publication timestamp is available
- `seen_date` = `None`
- `text` = extracted article text when available

Important:

- Wikipedia revision timestamps are not treated as publication dates.
- `date` is not populated from revision metadata.
- When claim-date filtering is active, Wikipedia documents without a real publication timestamp are excluded.

### 4.2 NewsAPI

The implementation calls NewsAPI’s `/v2/everything` endpoint and keeps:

- `title`
- `url`
- `source` = `NewsAPI`
- `date` = `publishedAt` normalized into the common schema
- `seen_date` = `None`
- `text` = `content` if present, otherwise `description`, otherwise empty string

Important:

- The code uses the original `publishedAt` field as the publication timestamp.
- It does not claim every result contains full article text.
- Some NewsAPI results may only include a snippet or a description.

## 5. Standard document schema

Documents from all sources are normalized into the following structure:

```python
{
    "title": "Artificial Intelligence Developments",
    "url": "https://example.com/article",
    "source": "NewsAPI",
    "date": "2026-11-01T09:30:00Z",
    "seen_date": None,
    "text": "Article content..."
}
```

Field meanings:

- `title`: article or document title
- `url`: original document URL
- `source`: source name (`Wikipedia`, `NewsAPI`)
- `date`: verified publication timestamp when available
- `seen_date`: observation timestamp from a source when it is relevant and separate from publication time
- `text`: article text or extracted content, which may be empty for metadata-only records

Missing publication dates are represented as `None`.

The implementation does not substitute `seen_date` or revision timestamps for publication dates.

## 6. Claim-date extraction and temporal filtering

The implementation supports claim-date extraction from explicit query text when the claim date/time is clear and unambiguous.

### Supported claim-date expressions

The extraction logic supports explicit forms such as:

- `Claim made on 2026-11-01`
- `Claim date: 2026-11-01`
- `Claim made at 2026-11-01T10:30:00+05:30`
- `Claim made on 1 November 2026`
- `Claim made at 10:30 AM on 1 November 2026`

It supports:

- ISO 8601 date strings and timestamps
- common unambiguous natural-language date forms
- timezone-aware timestamps when present

It intentionally does not guess when the query is ambiguous or unsupported.

### Exact timestamp filtering

When the claim includes an exact timestamp, the rule is:

```text
evidence_published_at <= claim_made_at
```

Times are converted to UTC before comparison.

Examples:

- claim: `2026-11-01T10:30:00+05:30`
- UTC claim time: `2026-11-01T05:30:00Z`

Therefore:

- evidence at `2026-11-01T03:59:59Z` -> include
- evidence at `2026-11-01T05:30:00Z` -> include
- evidence at `2026-11-01T05:30:01Z` -> exclude
- evidence on later dates -> exclude

### Date-only filtering

When only a claim date is available, the code uses the end of that date as a provisional cutoff in the configured claim timezone.

Default timezone assumption:

- `CLAIM_TIMEZONE` environment variable, if set
- otherwise `UTC`

Example:

- claim date: `2026-11-01`
- cutoff: `2026-11-01 23:59:59.999999` in the configured timezone

This is intentionally conservative and date-level only. The implementation does not claim that same-day evidence was definitely available before the claim was made.

### Temporal precision metadata

When claim-date filtering is active, the module records the precision of the extracted claim constraint. The values are:

- `exact_timestamp`
- `date_only`
- `unavailable`

This metadata describes the claim constraint, not the article publication date itself.

## 7. Missing and invalid timestamp handling

When claim-date filtering is active:

- missing publication timestamps are excluded
- invalid publication timestamps are excluded
- `None` publication dates are not replaced with `seen_date`
- `date` is never inferred from `seen_date`
- `date` is never inferred from Wikipedia revision metadata

If no claim-date constraint is extracted, the system preserves its normal retrieval behavior without temporal filtering.

## 8. Retry mechanism and HTTP handling

The implementation includes `request_with_retry()` in `src/evidence_retrieval/http_retry.py`.

It handles transient failures with exponential backoff and optional respect for the `Retry-After` header.

Retryable HTTP status codes implemented by the project:

- `429`
- `500`
- `502`
- `503`
- `504`

It also retries on temporary request-level errors such as:

- connection errors
- timeouts
- request exceptions

Failures from one source do not stop retrieval from the other source modules.

## 9. Validation and deduplication

After retrieval, documents are validated before they are considered final evidence.

Validation checks include:

- non-empty `title`
- non-empty `url`
- non-empty `source`
- `date` and `seen_date` are valid strings when present
- `text` is a string when present (empty string is allowed)

Duplicate documents are removed by URL using the deduplication helper in `src/evidence_retrieval/utils/deduplication.py`.

This preserves the first-seen URL and retains empty-URL documents as-is.

## 10. `prepare_for_ranking()` compatibility

The retrieval module provides `prepare_for_ranking(documents)` in `src/evidence_retrieval/retriever.py`.

This function:

- keeps the original retrieval documents intact
- creates ranking-ready copies
- filters out documents with empty or missing text
- adds `published_date` as an alias to `date` for downstream compatibility
- preserves the original `date` and `seen_date` fields

It is a compatibility adapter for the ranking layer and does not modify the ranking code.

## 11. Project structure

```text
Factlens/
├── .gitignore
├── README.md
├── pytest.ini
├── main.py
├── scripts/
│   └── test_retrieval.py
├── src/
│   ├── evidence_retrieval/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── http_retry.py
│   │   ├── retriever.py
│   │   ├── sources/
│   │   │   ├── __init__.py
│   │   │   ├── wikipedia.py
│   │   │   └── newsapi.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── date_filter.py
│   │       ├── deduplication.py
│   │       └── normalizer.py
│   └── ranking/
│       ├── bm25_ranker.py
│       ├── candidate_merger.py
│       ├── cross_encoder_reranker.py
│       ├── dense_ranker.py
│       ├── evidence_selector.py
│       └── passage_splitter.py
└── test/
    ├── conftest.py
    └── evidence_retrieval/
        ├── test_retriever.py
        ├── test_wikipedia.py
        ├── test_newsapi.py
        ├── test_offline_sources.py
```

## 12. Installation and configuration

### Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
python -m pip install requests python-dotenv pytest
```

The project also uses pytest markers for live tests. The current `pytest.ini` defines:

```ini
[pytest]
markers =
    live: tests that call live external APIs and are skipped by default
```

### Configure environment variables

Create a `.env` file at the project root when using NewsAPI.

Supported names:

```env
NEWSAPI_API_KEY=your_newsapi_api_key
# or
NEWSAPI_KEY=your_newsapi_api_key
```

Optional timezone configuration for claim-date filtering:

```env
CLAIM_TIMEZONE=UTC
```

The `.env` file is intentionally excluded from Git via `.gitignore`.

## 13. Running scripts and tests

### Manual retrieval check

From the project root:

```bash
python scripts/test_retrieval.py
```

This script calls `retrieve_evidence()` and prints the retrieved document metadata.

### Offline tests

The offline suite runs without live external calls:

```bash
python -m pytest -m "not live" -q
```

Live tests are marked with `@pytest.mark.live` and are skipped by default.

The existing offline tests cover:

- source retrieval behavior
- validation and deduplication
- date normalization and handling
- claim-date extraction
- exact timestamp filtering
- date-only filtering
- missing and invalid timestamp behavior
- `seen_date` separation
- Wikipedia revision timestamp handling
- NewsAPI publication timestamp handling
- downstream `prepare_for_ranking()` compatibility

## 14. Known limitations

- Claim-date filtering only activates when a clear claim time or date is extracted from the query.
- Date-only filtering is intentionally coarse and should not be treated as exact per-minute chronology.
- Wikipedia article metadata is not treated as original publication time.
- NewsAPI `publishedAt` is used only when the API supplies it.
- Retrieval quality depends on the query and on the availability of source data.
- The module does not guarantee that all retrieved texts are complete article texts or fully relevant to every query.

## 15. Scope statement

This README documents the Evidence Retrieval component only. It does not describe ranking or reranking logic, which are maintained in separate project modules.
