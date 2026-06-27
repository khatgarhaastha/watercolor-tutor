"""Tests for the Slice 2 nodes: go_back, reexplain, respond (offline)."""

import pytest

from watercolor_tutor.nodes.go_back import go_back
from watercolor_tutor.nodes.reexplain import reexplain
from watercolor_tutor.nodes.respond import respond
from watercolor_tutor.state import TutorState

# reexplain now grounds via RAG; stub it off here so these node tests stay offline
# (grounding is proven in test_grounding.py).
pytestmark = pytest.mark.usefixtures("stub_rag")


def _state(step: int = 2, intent: str = "", messages: list | None = None) -> TutorState:
    return TutorState(
        messages=messages or [{"role": "user", "content": "hi"}],
        step=step,
        awaiting_question=True,
        intent=intent,
        image_path="",
    )


# --- go_back -----------------------------------------------------------------


def test_go_back_decrements_step() -> None:
    update = go_back(_state(step=2))
    assert update["step"] == 1
    assert update["awaiting_question"] is False


def test_go_back_clamps_at_first_step() -> None:
    # Defense-in-depth: even if reached on step 1, it can't go below 1.
    assert go_back(_state(step=1))["step"] == 1


# --- reexplain ---------------------------------------------------------------


def test_reexplain_reteaches_current_step(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["messages"] = messages
        return "Here's another way to think about it."

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)
    update = reexplain(_state(step=2))

    assert update["messages"][0]["role"] == "assistant"
    assert update["awaiting_question"] is True
    # The instruction should name the current step and ask for a re-explanation.
    assert "Step 2" in captured["messages"][0]["content"]
    assert "Re-explain" in captured["messages"][0]["content"]


# --- respond -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("intent", "fragment"),
    [
        ("off_topic", "off-topic"),
        ("sharing_progress", "sharing"),
        ("skip_ahead", "FINAL step"),  # boundary message
        ("go_back", "FIRST step"),  # boundary message
    ],
)
def test_respond_uses_intent_specific_framing(
    intent: str, fragment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    def fake_generate(system: str, messages: list[dict]) -> str:
        captured["system"] = system
        return "ok"

    monkeypatch.setattr("watercolor_tutor.llm.generate", fake_generate)
    update = respond(_state(step=3, intent=intent))

    assert update["messages"][0]["role"] == "assistant"
    assert fragment in captured["system"]
    # respond never changes the step.
    assert "step" not in update
