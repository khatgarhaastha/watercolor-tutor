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


def _fake_classify(text: str) -> str:
    """A deterministic stand-in for the LLM classifier, keyed off the test inputs."""
    low = text.lower()
    ready = any(w in low for w in ("ready", "move on", "all set", "i'm good"))
    question = "?" in text
    if ready and question:
        return "both"
    return "ready" if ready else "question"


def test_interactive_lesson_flow(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the full pause/resume loop offline using Command(resume=...).

    Both the lesson text (generate) and the intent classifier are stubbed, so the
    whole conversation runs with no human and no network — the payoff of the
    interrupt + checkpointer design plus keeping classification in a node.
    """
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "LESSON")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", _fake_classify)
    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "test"}}

    # Kick off: welcome + teach step 1, then pause at await_learner.
    state = graph.invoke(initial_state, config=config)
    assert state["step"] == 1
    assert graph.get_state(config).next  # non-empty => paused, awaiting input

    # A pure question routes to `answer` and pauses again, still on step 1.
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

    contents = [_content(m) for m in state["messages"]]
    assert any("Welcome" in c for c in contents)
    assert contents.count("LESSON") >= 3  # one lesson per step


def test_both_intent_answers_and_advances(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v0 bug, now fixed: a reply that is BOTH a question and a ready-signal
    gets answered AND advances the step (it used to silently drop the question).
    """
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "TEXT")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", _fake_classify)
    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "both"}}

    graph.invoke(initial_state, config=config)  # teach step 1, pause
    before = len(graph.get_state(config).values["messages"])

    # "let's move on, what should I paint?" -> classified "both"
    state = graph.invoke(Command(resume="let's move on, what should I paint?"), config=config)

    assert state["step"] == 2  # advanced (answer happened first, then advance)
    assert graph.get_state(config).next  # paused again at the next step
    # The learner turn, the answer, and the next lesson were all appended.
    assert len(state["messages"]) >= before + 3
