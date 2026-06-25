"""Tests for graph wiring and the welcome node.

These run fully offline: the welcome node is deterministic, and we stub the LLM
so invoking the graph never calls the real Anthropic API.
"""

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

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


def test_interactive_lesson_flow(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the full pause/resume loop offline using Command(resume=...).

    No real input() and no real LLM — Command(resume=text) supplies what the
    learner would type, and the stubbed generate() stands in for Claude. This is
    the payoff of the interrupt + checkpointer design: the whole conversation is
    testable without a human or the network.
    """
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "LESSON")
    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "test"}}

    # Kick off: welcome + teach step 1, then pause at await_learner.
    state = graph.invoke(initial_state, config=config)
    assert state["step"] == 1
    assert graph.get_state(config).next  # non-empty => paused, awaiting input

    # A question routes to `answer` and pauses again, still on step 1.
    state = graph.invoke(Command(resume="what brush should I use?"), config=config)
    assert state["step"] == 1
    assert graph.get_state(config).next

    # "ready" advances to step 2, then 3 (more steps remain each time).
    state = graph.invoke(Command(resume="ready"), config=config)
    assert state["step"] == 2
    state = graph.invoke(Command(resume="all set"), config=config)
    assert state["step"] == 3

    # "ready" on the final step ends the lesson — the graph is no longer paused.
    state = graph.invoke(Command(resume="i'm good"), config=config)
    assert not graph.get_state(config).next

    # The transcript should contain the greeting and the taught lessons.
    contents = [_content(m) for m in state["messages"]]
    assert any("Welcome" in c for c in contents)
    assert contents.count("LESSON") >= 3  # one lesson per step
