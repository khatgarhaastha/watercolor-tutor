"""Tests for the MCP search seam — offline (the real servers are never launched).

We verify graceful paths: disabled-by-config and failures return "" so callers
degrade. For image_search we also prove the image→text FALLBACK. The happy path
(a real MCP connection) is exercised by the live probe, not the suite.
"""

from types import SimpleNamespace

import pytest

from watercolor_tutor import mcp_search


def _settings(**overrides: object) -> SimpleNamespace:
    base = {
        "web_search_enabled": True,
        "mcp_search_command": "text-server",
        "image_search_enabled": True,
        "mcp_image_command": "image-server",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# --- web_search --------------------------------------------------------------


def test_web_search_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_search, "get_settings", lambda: _settings(web_search_enabled=False))
    assert mcp_search.web_search("anything") == ""


def test_web_search_degrades_gracefully_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_search, "get_settings", lambda: _settings())

    async def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("server unreachable")

    monkeypatch.setattr(mcp_search, "_call_tool_async", boom)
    assert mcp_search.web_search("anything") == ""


# --- image_search (incl. the image -> text fallback) -------------------------


def test_image_search_disabled_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_search, "get_settings", lambda: _settings(image_search_enabled=False))
    assert mcp_search.image_search("flat wash") == ""


def test_image_search_falls_back_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_search, "get_settings", lambda: _settings())

    async def fake(command: str, tool_name: str, arguments: dict) -> str:
        if tool_name == "ddg-image-search":
            raise RuntimeError("403 Ratelimit")  # image endpoint throttled
        return "TEXT REFERENCES"  # ddg-text-search fallback succeeds

    monkeypatch.setattr(mcp_search, "_call_tool_async", fake)
    assert mcp_search.image_search("flat wash") == "TEXT REFERENCES"


def test_image_search_degrades_when_both_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_search, "get_settings", lambda: _settings())

    async def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("both down")

    monkeypatch.setattr(mcp_search, "_call_tool_async", boom)
    assert mcp_search.image_search("flat wash") == ""
