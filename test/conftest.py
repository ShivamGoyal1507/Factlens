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
