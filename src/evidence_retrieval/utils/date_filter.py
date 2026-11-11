from datetime import datetime, timedelta, timezone
import re

from src.evidence_retrieval.config import get_default_claim_timezone


def _as_clean_string(value):
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def normalize_publication_date(value):
    """
    Lightweight normalization:
    - keep None if missing/invalid
    - accept common ISO-like timestamps
    - preserve date/ts meaning without inventing values
    """
    text = _as_clean_string(value)
    if text is None:
        return None

    candidate = text.replace("Z", "+00:00")

    try:
        datetime.fromisoformat(candidate)
        return text
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            datetime.strptime(text, fmt)
            return text
        except ValueError:
            continue

    return None


def normalize_seen_date(value):
    """
    GDELT seen_date normalization:
    - keep None if missing/invalid
    - supports GDELT format: YYYYMMDDTHHMMSSZ
    """
    text = _as_clean_string(value)
    if text is None:
        return None

    try:
        datetime.strptime(text, "%Y%m%dT%H%M%SZ")
        return text
    except ValueError:
        pass

    candidate = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
        return text
    except ValueError:
        return None


def _parse_iso_datetime(value, timezone_name=None):
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned:
        return None

    candidate = cleaned.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        tz = _coerce_claim_timezone(timezone_name or get_default_claim_timezone())
        parsed = parsed.replace(tzinfo=tz)

    return parsed.astimezone(timezone.utc)


def _parse_gdelt_seen_date(value):
    text = _as_clean_string(value)
    if text is None:
        return None
    try:
        parsed = datetime.strptime(text, "%Y%m%dT%H%M%SZ")
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def _coerce_claim_timezone(timezone_name):
    if not timezone_name:
        return timezone.utc

    name = timezone_name.strip()
    if not name:
        return timezone.utc

    if name.upper() == "UTC":
        return timezone.utc

    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return timezone.utc


def _parse_date_only(value):
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d %B %Y", "%B %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    return None


def extract_claim_date_constraint(query):
    """
    Extract an explicit claim-date constraint from a query when clear and unambiguous.

    Returns a dict with:
      {
        "precision": "exact_timestamp" | "date_only" | "unavailable",
        "value": datetime | date | None,
        "timezone": str | None,
      }

    This intentionally does not guess when the query lacks a well-defined claim date.
    """
    if not isinstance(query, str):
        return {"precision": "unavailable", "value": None, "timezone": None}

    text = query.strip()
    if not text:
        return {"precision": "unavailable", "value": None, "timezone": None}

    bare_iso_date = re.fullmatch(r"\d{4}-\d{2}-\d{2}", text)
    if bare_iso_date:
        date_obj = _parse_date_only(text)
        if date_obj is not None:
            return {
                "precision": "date_only",
                "value": date_obj,
                "timezone": get_default_claim_timezone(),
            }

    bare_iso_timestamp = re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?", text)
    if bare_iso_timestamp:
        iso_value = text.replace("Z", "+00:00")
        parsed = _parse_iso_datetime(iso_value)
        if parsed is not None:
            return {
                "precision": "exact_timestamp",
                "value": parsed,
                "timezone": get_default_claim_timezone(),
            }

    patterns = [
        r"claim\s+date\s*[:=]\s*(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(:\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)",
        r"claim\s+made\s+(?:on|at)\s*(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(:\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)",
        r"(?:on|at)\s+(\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(:\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)",
        r"(?:on|at)\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}(?:\s+at\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)",
        r"(?:on|at)\s+(\w+\s+\d{1,2},\s*\d{4}(?:\s+at\s+\d{1,2}:\d{2}(?:\s*[APap][Mm])?)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip()
        if not value:
            continue

        iso_candidate = value.replace(" ", "T") if "T" not in value and re.search(r"\d{4}-\d{2}-\d{2}", value) else value
        iso_candidate = iso_candidate.replace(" UTC", "+00:00").replace(" GMT", "+00:00")

        if re.search(r"\d{4}-\d{2}-\d{2}", iso_candidate) and not re.search(r"T|Z|[+-]\d{2}:?\d{2}", iso_candidate):
            date_obj = _parse_date_only(iso_candidate)
            if date_obj is not None:
                return {
                    "precision": "date_only",
                    "value": date_obj,
                    "timezone": get_default_claim_timezone(),
                }

        iso_candidate = iso_candidate.replace(" ", "T") if "T" not in iso_candidate else iso_candidate
        if iso_candidate.endswith(" UTC"):
            iso_candidate = iso_candidate[:-4] + "+00:00"
        if iso_candidate.endswith(" GMT"):
            iso_candidate = iso_candidate[:-4] + "+00:00"
        if isinstance(iso_candidate, str):
            if re.search(r"\d{1,2}:\d{2}\s*[APap][Mm]", iso_candidate):
                normalized = re.sub(r"\s*([APap][Mm])", lambda m: m.group(1).upper(), iso_candidate)
                if "AM" in normalized.upper() or "PM" in normalized.upper():
                    parts = normalized.split()
                    if len(parts) >= 4 and parts[-2].lower() in {"am", "pm"}:
                        time_part = parts[-2] + " " + parts[-1]
                        normalized = " ".join(parts[:-2] + [time_part])

        parsed = _parse_iso_datetime(iso_candidate, timezone_name=get_default_claim_timezone())
        if parsed is not None:
            return {
                "precision": "exact_timestamp",
                "value": parsed,
                "timezone": get_default_claim_timezone(),
            }

        natural_date = _parse_date_only(value)
        if natural_date is not None:
            return {
                "precision": "date_only",
                "value": natural_date,
                "timezone": get_default_claim_timezone(),
            }

    return {"precision": "unavailable", "value": None, "timezone": None}


def _to_utc_datetime(value, timezone_name=None):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            aware = value.replace(tzinfo=_coerce_claim_timezone(timezone_name or get_default_claim_timezone()))
            return aware.astimezone(timezone.utc)
        return value.astimezone(timezone.utc)

    if isinstance(value, str):
        parsed = _parse_iso_datetime(value, timezone_name=timezone_name or get_default_claim_timezone())
        if parsed is not None:
            return parsed

    return None


def _date_only_end_of_day(date_value, timezone_name=None):
    tz = _coerce_claim_timezone(timezone_name or get_default_claim_timezone())
    naive = datetime.combine(date_value, datetime.max.time())
    aware = naive.replace(tzinfo=tz)
    return aware.astimezone(timezone.utc)


def get_temporal_cutoff_for_claim(query):
    """
    Return a UTC cutoff datetime if a claim date constraint is available.
    Returns None when no valid claim-date constraint was extracted.
    """
    constraint = extract_claim_date_constraint(query)
    precision = constraint.get("precision")
    value = constraint.get("value")
    if precision == "unavailable" or value is None:
        return None

    if precision == "exact_timestamp":
        return _to_utc_datetime(value, timezone_name=constraint.get("timezone"))

    if precision == "date_only":
        return _date_only_end_of_day(value, constraint.get("timezone"))

    return None


def document_is_temporally_eligible(document, claim_datetime=None, claim_precision=None):
    """
    Return True when a document's publication timestamp is on or before the claim cutoff.
    Missing/invalid publication timestamps are excluded when filtering is active.
    """
    if claim_datetime is None:
        return True

    published = document.get("date")
    if not isinstance(published, str) or not published.strip():
        return False

    document_date = _parse_date_only(published)
    if document_date is not None:
        if claim_precision == "exact_timestamp":
            return False
        tz_name = get_default_claim_timezone()
        tz = _coerce_claim_timezone(tz_name)
        cutoff_day = claim_datetime.astimezone(tz).date()
        return document_date <= cutoff_day

    document_timestamp = _parse_iso_datetime(published)
    if document_timestamp is not None:
        return document_timestamp.astimezone(timezone.utc) <= claim_datetime.astimezone(timezone.utc)

    return False


def get_claim_date_precision(query):
    """Return the temporal precision for a supported claim-date constraint."""
    constraint = extract_claim_date_constraint(query)
    if constraint is None:
        return "unavailable"
    return constraint.get("precision", "unavailable")