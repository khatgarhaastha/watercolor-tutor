"""Tests for durable persistence — a returning thread resumes its prior state.

Fully offline: the LLM (`generate`/`classify_intent`) and RAG (`stub_rag`) are
stubbed, and we use a real local SQLite file in `tmp_path`. Two separate
`SqliteSaver` blocks on the SAME file simulate an app restart.
"""

from pathlib import Path

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from watercolor_tutor.graph import compile_graph

pytestmark = pytest.mark.usefixtures("stub_rag")  # keep teaching/vision offline


@pytest.fixture(autouse=True)
def _stub_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("watercolor_tutor.llm.generate", lambda *a, **k: "LESSON")
    monkeypatch.setattr("watercolor_tutor.classifier.classify_intent", lambda *a, **k: "ready")


def _initial() -> dict:
    return {"messages": [], "step": 0, "awaiting_question": False, "intent": "", "image_path": ""}


def test_session_resumes_after_restart(tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.sqlite")
    config: RunnableConfig = {"configurable": {"thread_id": "learner-1"}}

    # --- Session 1: start the lesson and advance to step 2, then "close" the app
    # (exit the with-block, releasing the connection). ---
    with SqliteSaver.from_conn_string(db) as saver:
        saver.setup()
        graph = compile_graph(saver)
        graph.invoke(_initial(), config=config)  # welcome + teach step 1, pause
        graph.invoke(Command(resume="ready"), config=config)  # ready -> advance to step 2
        assert graph.get_state(config).values["step"] == 2

    # --- Session 2: a brand-new graph + saver on the SAME db file (a "restart").
    with SqliteSaver.from_conn_string(db) as saver:
        saver.setup()
        graph = compile_graph(saver)

        snapshot = graph.get_state(config)
        assert snapshot.values["step"] == 2  # resumed the prior state from disk!
        assert snapshot.next  # still paused at await_learner, ready to continue

        graph.invoke(Command(resume="ready"), config=config)  # continue -> step 3
        assert graph.get_state(config).values["step"] == 3


def test_different_thread_is_isolated_and_fresh(tmp_path: Path) -> None:
    db = str(tmp_path / "sessions.sqlite")
    with SqliteSaver.from_conn_string(db) as saver:
        saver.setup()
        graph = compile_graph(saver)

        # Drive one learner forward.
        learner1: RunnableConfig = {"configurable": {"thread_id": "learner-1"}}
        graph.invoke(_initial(), config=learner1)
        graph.invoke(Command(resume="ready"), config=learner1)
        assert graph.get_state(learner1).values["step"] == 2

        # A different thread_id has no saved state — a separate, fresh session.
        learner2: RunnableConfig = {"configurable": {"thread_id": "learner-2"}}
        assert not graph.get_state(learner2).values
