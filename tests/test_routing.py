"""Tests for the conditional-edge router and its helpers (fully offline).

The router is a pure function, so we can exercise every branch directly with no
LLM and no graph run — that's the payoff of keeping routing free of side effects.
"""

from langchain_core.messages import AIMessage, HumanMessage

from watercolor_tutor.routing import is_ready_signal, last_user_message, route_after_input
from watercolor_tutor.state import TutorState


def _state(messages: list, step: int = 1) -> TutorState:
    return TutorState(messages=messages, step=step, awaiting_question=True)


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
    # The known heuristic limitation: this reads as a comment, not "ready".
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


def test_route_question_goes_to_answer() -> None:
    messages = [{"role": "user", "content": "what brush?"}]
    assert route_after_input(_state(messages, step=1)) == "answer"


def test_route_ready_with_steps_left_advances() -> None:
    messages = [{"role": "user", "content": "ready!"}]
    assert route_after_input(_state(messages, step=1)) == "advance"


def test_route_ready_on_final_step_ends() -> None:
    messages = [{"role": "user", "content": "ready!"}]
    assert route_after_input(_state(messages, step=3)) == "end"
