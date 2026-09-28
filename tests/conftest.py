"""Test-session hooks: tests marked ``omnisafe`` are skipped when the pinned stack is not installed."""

from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if importlib.util.find_spec("omnisafe") is not None:
        return
    skip = pytest.mark.skip(reason="needs the pinned OmniSafe stack (omnisafe is not installed)")
    for item in items:
        if "omnisafe" in item.keywords:
            item.add_marker(skip)
