"""Tests for the web_search node — offline (the MCP seam is stubbed)."""

import pytest

from watercolor_tutor import mcp_search
from watercolor_tutor.nodes.web_search import web_search
from watercolor_tutor.state import TutorState


def _state(query: str) -> TutorState:
    return TutorState(
        messages=[{"role": "user", "content": query}],
        step=1,
        awaiting_question=True,
        intent="",
        image_path="",
    )


def test_web_search_node_folds_live_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        mcp_search,
        "web_search",
        lambda *a, **k: "1. Cheap Set\n   URL: http://x.com\n   Summary: ...",
    )

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["prompt"] = messages[0]["content"]
        return "Here are a few beginner-friendly options..."

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)

    update = web_search(_state("what's a good cheap beginner set to buy?"))

    assert update["messages"][0]["role"] == "assistant"
    assert "http://x.com" in captured["prompt"]  # live results folded into the prompt
    assert "good cheap beginner set" in captured["prompt"]  # the learner's question too
    assert "step" not in update  # reply-and-stay


def test_web_search_node_degrades_when_search_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(mcp_search, "web_search", lambda *a, **k: "")  # search failed/disabled

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["prompt"] = messages[0]["content"]
        return "I couldn't fetch live results right now, but in general..."

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)

    update = web_search(_state("how much is cold-press paper?"))

    assert update["messages"][0]["role"] == "assistant"
    assert "unavailable" in captured["prompt"].lower()  # the degraded preamble was used
