import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CLAIM_TIMEZONE = "UTC"


def get_default_claim_timezone():
    """
    Default timezone used when a claim date is supplied without an explicit tz.
    This is deliberately project-scoped and documented; we do not guess from the
    local machine timezone.
    """
    return os.getenv("CLAIM_TIMEZONE", DEFAULT_CLAIM_TIMEZONE).strip() or DEFAULT_CLAIM_TIMEZONE


def get_newsapi_api_key():
    """
    Read NewsAPI key from environment.
    Supported names:
    - NEWSAPI_API_KEY
    - NEWSAPI_KEY
    """
    key = (
        os.getenv("NEWSAPI_API_KEY")
        or os.getenv("NEWSAPI_KEY")
        or ""
    ).strip()

    return key or None