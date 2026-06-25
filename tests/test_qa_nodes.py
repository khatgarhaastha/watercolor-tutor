"""Tests for the answer and advance nodes (offline; LLM stubbed for answer)."""

import pytest

from watercolor_tutor.nodes.advance import advance
from watercolor_tutor.nodes.answer import answer
from watercolor_tutor.state import TutorState


def _state(messages: list, step: int = 1) -> TutorState:
    return TutorState(messages=messages, step=step, awaiting_question=True)


def test_answer_replies_and_keeps_awaiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "Use a round brush.")
    state = _state([{"role": "user", "content": "what brush?"}], step=2)

    update = answer(state)

    assert update["messages"][0]["role"] == "assistant"
    assert update["messages"][0]["content"] == "Use a round brush."
    # answer must NOT change awaiting_question — we still wait for the learner.
    assert "awaiting_question" not in update


def test_answer_sends_question_and_step_context(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["system"] = system
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)
    state = _state([{"role": "user", "content": "how much water?"}], step=3)

    answer(state)

    # The question is sent as the leading "user" turn; step context is in system.
    assert captured["messages"][0] == {"role": "user", "content": "how much water?"}
    assert "Step 3" in captured["system"]


def test_advance_increments_step_and_clears_awaiting() -> None:
    update = advance(_state([], step=1))
    assert update["step"] == 2
    assert update["awaiting_question"] is False
