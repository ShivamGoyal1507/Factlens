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
def _reset_gdelt_rate_limiter():
    """Keep GDELT's 5-second rate limiter from slowing down or leaking between tests."""
    from src.evidence_retrieval.sources import gdelt

    original_interval = gdelt.MIN_INTERVAL_SECONDS
    original_state_file = gdelt.STATE_FILE
    gdelt.STATE_FILE = None  # never read/write the real ~/.cache state in tests
    gdelt.MIN_INTERVAL_SECONDS = 0.0
    gdelt.reset_rate_limiter()
    yield
    gdelt.reset_rate_limiter()
    gdelt.MIN_INTERVAL_SECONDS = original_interval
    gdelt.STATE_FILE = original_state_file
