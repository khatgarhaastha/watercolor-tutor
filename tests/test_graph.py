"""Tests for graph wiring and the welcome node.

These run fully offline: the welcome node is deterministic, and we stub the LLM
so invoking the graph never calls the real Anthropic API.
"""

import pytest

from watercolor_tutor.graph import compile_graph
from watercolor_tutor.nodes.welcome import welcome
from watercolor_tutor.state import TutorState


def _content(message: object) -> str:
    """Read a message's text whether it's a dict or a LangChain message object.

    The `add_messages` reducer coerces our dicts into message objects during a
    graph run, so post-invoke messages expose `.content` rather than `["content"]`.
    """
    return getattr(message, "content", None) or message["content"]  # type: ignore[index]


def test_welcome_node_returns_partial_update(initial_state: TutorState) -> None:
    update = welcome(initial_state)
    assert update["step"] == 1
    assert update["messages"][0]["role"] == "assistant"
    assert "Welcome" in update["messages"][0]["content"]


def test_graph_compiles_and_invokes(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stub the LLM so the teach node returns canned text instead of calling out.
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "Here is your lesson.")

    graph = compile_graph()
    result = graph.invoke(initial_state)

    # welcome greets and sets step=1; teach delivers the lesson and waits.
    assert result["step"] == 1
    assert result["awaiting_question"] is True

    contents = [_content(m) for m in result["messages"]]
    assert any("Welcome" in c for c in contents)
    assert any("Here is your lesson." in c for c in contents)
