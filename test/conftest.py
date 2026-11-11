import os

# Import torch before anything else (see main.py): other libraries load a
# different OpenMP runtime first and Hugging Face model loading segfaults.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402,F401

from unittest.mock import patch

import pytest


class _Mocker:
    def __init__(self):
        self._patchers = []

    def patch(self, target, *args, **kwargs):
        mocker = patch(target, *args, **kwargs)
        self._patchers.append(mocker)
        return mocker.start()

    def stopall(self):
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._patchers.clear()


@pytest.fixture
def mocker():
    mocker_instance = _Mocker()
    yield mocker_instance
    mocker_instance.stopall()


@pytest.fixture(autouse=True)
def _reset_wikipedia_and_newsapi_state():
    """Wikipedia's response cache / 429 cooldown and NewsAPI's quota flag are
    process-wide; reset them so mocked responses never leak between tests."""
    from src.evidence_retrieval.sources import newsapi, wikipedia

    original_interval = wikipedia._MIN_REQUEST_INTERVAL
    wikipedia._MIN_REQUEST_INTERVAL = 0.0
    wikipedia.reset_wikipedia_state()
    newsapi.reset_newsapi_quota_flag()
    yield
    wikipedia.reset_wikipedia_state()
    newsapi.reset_newsapi_quota_flag()
    wikipedia._MIN_REQUEST_INTERVAL = original_interval
