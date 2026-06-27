"""Tests for the MCP search seam — offline (the real server is never launched).

We verify the two graceful paths: disabled-by-config and any-failure both return
"" so callers degrade instead of crashing. The happy path (a real MCP connection)
is exercised by the live probe, not the test suite.
"""

from types import SimpleNamespace

import pytest

from watercolor_tutor import mcp_search


def test_web_search_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_search,
        "get_settings",
        lambda: SimpleNamespace(web_search_enabled=False, mcp_search_command="x"),
    )
    assert mcp_search.web_search("anything") == ""


def test_web_search_degrades_gracefully_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_search,
        "get_settings",
        lambda: SimpleNamespace(web_search_enabled=True, mcp_search_command="x"),
    )

    async def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(mcp_search, "_search_async", boom)
    # No crash, no network — just an empty result the caller can degrade on.
    assert mcp_search.web_search("anything") == ""
