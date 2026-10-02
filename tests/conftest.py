"""Test-session hooks and shared fixtures.

* Tests marked ``omnisafe`` are skipped when the pinned stack is not installed.
* ``keys_open`` and ``pin_open``: every PENDING key is answered (configs.registered.ANSWERED_QUESTIONS,
  docs/DECISIONS.md of 2026-10-02), so a test of what the code does while a key is open pins that key open
  explicitly, whatever the amendment log has answered since. A test module may define its own ``keys_open``.
"""

from __future__ import annotations

import importlib.util
from typing import Callable

import pytest

from configs import registered as R


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("omnisafe") is not None:
        return
    skip = pytest.mark.skip(reason="needs the pinned OmniSafe stack (omnisafe is not installed)")
    for item in items:
        if "omnisafe" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def keys_open(monkeypatch) -> None:
    """Every PENDING key open (configs.registered.ANSWERED_QUESTIONS empty): the test reads what the code does only
    while a key is open (a gate holds, a verdict waits, the other reading is computed)."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())


@pytest.fixture
def pin_open(monkeypatch) -> Callable[..., None]:
    """``pin_open(*keys)``: those PENDING keys open, every other key as the repository answers it."""

    def pin(*keys: str) -> None:
        unknown = [key for key in keys if key not in R.PENDING]
        assert not unknown, f"not PENDING keys: {unknown}"
        monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.ANSWERED_QUESTIONS) - set(keys))

    return pin
