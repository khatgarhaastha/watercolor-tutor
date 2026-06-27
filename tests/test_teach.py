"""Tests for the teach node.

The teach node calls Claude, so every test stubs `llm.generate` — tests stay
fast, free, and offline. We patch the name on the `llm` module because the node
calls `llm.generate(...)` (looked up at call time), so the patch takes effect.
"""

import pytest

from watercolor_tutor.nodes.teach import teach
from watercolor_tutor.state import TutorState

# These tests exercise the step prompt + message shape, not RAG — keep RAG
# stubbed off so they stay offline. Grounding is proven in test_grounding.py.
pytestmark = pytest.mark.usefixtures("stub_rag")


def test_teach_emits_lesson_and_sets_awaiting(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "Mix paint with water.")
    initial_state["step"] = 1

    update = teach(initial_state)

    assert update["awaiting_question"] is True
    assert update["messages"][0]["role"] == "assistant"
    assert update["messages"][0]["content"] == "Mix paint with water."


def test_teach_sends_the_matching_step_prompt(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Capture what the node sends to the model so we can assert it's step 2's
    # instruction (and that the first turn uses the required "user" role).
    captured: dict = {}

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["system"] = system
        captured["messages"] = messages
        return "ok"

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)
    initial_state["step"] = 2

    teach(initial_state)

    assert captured["messages"][0]["role"] == "user"
    assert "Step 2" in captured["messages"][0]["content"]
