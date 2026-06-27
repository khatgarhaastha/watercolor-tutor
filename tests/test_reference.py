"""Tests for the reference node — offline (the MCP image seam is stubbed)."""

import pytest

from watercolor_tutor import mcp_search
from watercolor_tutor.nodes.reference import reference
from watercolor_tutor.state import TutorState


def _state(step: int) -> TutorState:
    return TutorState(
        messages=[{"role": "user", "content": "can I see an example?"}],
        step=step,
        awaiting_question=True,
        intent="",
        image_path="",
    )


def test_reference_uses_step_query_and_folds_results(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_image_search(query: str, *a: object, **k: object) -> str:
        captured["query"] = query
        return "1. Flat wash demo\n   URL: http://x.com/wash"

    monkeypatch.setattr(mcp_search, "image_search", fake_image_search)

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["prompt"] = messages[0]["content"]
        return "Here's a reference that should show a clean wash..."

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)

    update = reference(_state(step=3))

    assert update["messages"][0]["role"] == "assistant"
    assert "flat wash" in captured["query"].lower()  # step-anchored beginner query
    assert "http://x.com/wash" in captured["prompt"]  # results folded in for selection
    assert "step" not in update  # reply-and-stay


def test_reference_degrades_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(mcp_search, "image_search", lambda *a, **k: "")  # search failed/disabled

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["prompt"] = messages[0]["content"]
        return "I couldn't fetch a reference right now..."

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)

    update = reference(_state(step=2))

    assert update["messages"][0]["role"] == "assistant"
    assert "unavailable" in captured["prompt"].lower()  # the degraded preamble was used
