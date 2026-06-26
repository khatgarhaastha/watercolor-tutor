"""Tests for the conditional-edge routers and helpers (fully offline).

The routers are pure functions that read `state["intent"]`, so we exercise every
branch directly with no LLM and no graph run — the payoff of keeping the routing
decision separate from the classify node that produces the intent.
"""

from langchain_core.messages import AIMessage, HumanMessage

from watercolor_tutor.routing import (
    is_ready_signal,
    last_user_message,
    route_after_answer,
    route_after_input,
)
from watercolor_tutor.state import TutorState


def _state(messages: list | None = None, step: int = 1, intent: str = "") -> TutorState:
    return TutorState(messages=messages or [], step=step, awaiting_question=True, intent=intent)


def test_is_ready_signal_detects_ready_phrases() -> None:
    assert is_ready_signal("I'm ready")
    assert is_ready_signal("next please")
    assert is_ready_signal("Got it, thanks!")
    # Broadened keywords so a demo feels less brittle.
    assert is_ready_signal("all set")
    assert is_ready_signal("i'm good")
    assert is_ready_signal("sounds good, let's go")


def test_is_ready_signal_false_for_questions_and_statements() -> None:
    assert not is_ready_signal("what brush should I use?")
    # Known keyword-heuristic limitation (still relevant: it's the fallback).
    assert not is_ready_signal("i have all the supplies with me")


def test_last_user_message_with_dicts() -> None:
    messages = [
        {"role": "assistant", "content": "lesson"},
        {"role": "user", "content": "what paper?"},
    ]
    assert last_user_message(_state(messages)) == "what paper?"


def test_last_user_message_with_message_objects() -> None:
    # Mirrors what the add_messages reducer produces during a real graph run.
    messages = [AIMessage(content="lesson"), HumanMessage(content="what paper?")]
    assert last_user_message(_state(messages)) == "what paper?"


def test_last_user_message_none_when_no_user_turn() -> None:
    assert last_user_message(_state([{"role": "assistant", "content": "hi"}])) is None


# --- route_after_input: reads state["intent"] -------------------------------


def test_route_question_goes_to_answer() -> None:
    assert route_after_input(_state(intent="question", step=1)) == "answer"


def test_route_both_goes_to_answer_first() -> None:
    # "both" answers the question first; route_after_answer advances afterward.
    assert route_after_input(_state(intent="both", step=1)) == "answer"


def test_route_ready_with_steps_left_advances() -> None:
    assert route_after_input(_state(intent="ready", step=1)) == "advance"


def test_route_ready_on_final_step_ends() -> None:
    assert route_after_input(_state(intent="ready", step=3)) == "end"


# --- route_after_answer: the v0 "both" fix ----------------------------------


def test_route_after_answer_question_loops_back() -> None:
    assert route_after_answer(_state(intent="question", step=1)) == "await_learner"


def test_route_after_answer_both_advances() -> None:
    assert route_after_answer(_state(intent="both", step=1)) == "advance"


def test_route_after_answer_both_on_final_step_ends() -> None:
    assert route_after_answer(_state(intent="both", step=3)) == "end"
