"""Tests for graph wiring and the welcome node.

These run fully offline: the welcome node is deterministic, and we stub the LLM
so invoking the graph never calls the real Anthropic API.
"""

import base64
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from watercolor_tutor.graph import compile_graph
from watercolor_tutor.nodes.welcome import welcome
from watercolor_tutor.state import TutorState

_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

# Flow tests exercise routing, not RAG — keep RAG stubbed off so teach/reexplain/
# vision_feedback don't load the embedding model. (Grounding is proven in
# test_grounding.py.)
pytestmark = pytest.mark.usefixtures("stub_rag")


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


def _classify_v2(text: str) -> str:
    """Maps the integration test's resume phrases to the full intent set."""
    low = text.lower()
    if "confused" in low or "lost" in low:
        return "confused"
    if "skip" in low:
        return "skip_ahead"
    if "go back" in low or "revisit" in low:
        return "go_back"
    if "painted" in low:
        return "sharing_progress"
    if "favorite" in low or "weather" in low:
        return "off_topic"
    if "ready" in low:
        return "ready"
    return "question"


def test_full_intent_set_with_navigation_bounds(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise every new intent and prove the two navigation boundaries hold."""
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "TEXT")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", _classify_v2)
    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "v2"}}

    def resume(text: str) -> dict:
        return graph.invoke(Command(resume=text), config=config)

    graph.invoke(initial_state, config=config)  # teach step 1, pause

    # confused -> re-explained, still on step 1
    assert resume("I'm confused")["step"] == 1

    # BOUNDARY: go_back on the first step is blocked -> respond, still step 1
    assert resume("can we go back?")["step"] == 1

    # skip_ahead forward works: 1 -> 2
    assert resume("can we skip ahead?")["step"] == 2

    # go_back works when not on the first step: 2 -> 1
    assert resume("wait, can we go back?")["step"] == 1

    # skip up to the last step: 1 -> 2 -> 3
    assert resume("skip ahead please")["step"] == 2
    assert resume("skip ahead again")["step"] == 3

    # BOUNDARY: skip_ahead on the last step is blocked -> respond, still step 3
    assert resume("skip ahead")["step"] == 3

    # sharing_progress and off_topic both reply and stay on step 3
    assert resume("I just painted a blue sky!")["step"] == 3
    assert resume("what's your favorite color?")["step"] == 3
    assert graph.get_state(config).next  # still paused, lesson not over

    # ready on the last step ends the lesson
    resume("ok I'm ready to finish")
    assert not graph.get_state(config).next


def test_needs_web_info_routes_to_web_search(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 'needs_web_info' reply routes to web_search, folds live results, stays on-step.

    The MCP seam is stubbed — the real search server is never launched in tests.
    """
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "LIVE ANSWER")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", lambda _t: "needs_web_info")
    monkeypatch.setattr(
        "watercolor_tutor.mcp_search.web_search", lambda *a, **k: "1. A Set\n   URL: http://x"
    )

    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "web"}}
    graph.invoke(initial_state, config=config)  # teach step 1, pause

    state = graph.invoke(Command(resume="what set should I buy?"), config=config)
    contents = [_content(m) for m in state["messages"]]
    assert any("LIVE ANSWER" in c for c in contents)  # web_search synthesis appended
    assert state["step"] == 1  # stayed on the current step
    assert graph.get_state(config).next  # paused again


def test_vision_feedback_flow(
    initial_state: TutorState, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Sharing an image routes to vision_feedback (bypassing classify), stays on
    the current step, and clears the image — then normal replies classify again.
    """
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *_: "LESSON")
    monkeypatch.setattr("watercolor_tutor.llm.see", lambda *a, **k: "Nice even wash!")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", lambda _t: "question")
    image = tmp_path / "wash.png"
    image.write_bytes(_PNG_1X1)

    graph = compile_graph()
    config: RunnableConfig = {"configurable": {"thread_id": "vision"}}
    graph.invoke(initial_state, config=config)  # teach step 1, pause

    # Share an image -> vision_feedback (not classify).
    state = graph.invoke(
        Command(resume={"text": "how's this?", "image_path": str(image)}), config=config
    )
    contents = [_content(m) for m in state["messages"]]
    assert any("Nice even wash!" in c for c in contents)  # vision feedback appended
    assert state["step"] == 1  # stayed on the current step
    assert state["image_path"] == ""  # image consumed/cleared
    assert graph.get_state(config).next  # paused again

    # A normal text reply still goes through classify -> answer.
    state = graph.invoke(Command(resume="what brush?"), config=config)
    assert state["step"] == 1
    assert graph.get_state(config).next
