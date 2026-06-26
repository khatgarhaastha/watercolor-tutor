"""Tests for the graph state and its reducers."""

from typing import Any

from langgraph.graph.message import add_messages

from watercolor_tutor.state import TutorState


def test_tutor_state_has_expected_keys() -> None:
    # TypedDict exposes its declared fields via __annotations__.
    assert set(TutorState.__annotations__) == {
        "messages",
        "step",
        "awaiting_question",
        "intent",
        "image_path",
    }


def test_add_messages_reducer_appends() -> None:
    # The messages field uses add_messages, which appends rather than overwrites.
    # Annotate as list[Any] so the reducer's broad message union is satisfied.
    existing: list[Any] = [{"role": "assistant", "content": "hello"}]
    new: list[Any] = [{"role": "user", "content": "hi"}]
    merged = add_messages(existing, new)
    assert isinstance(merged, list)
    assert len(merged) == 2
