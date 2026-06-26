"""Tests for the conditional-edge routers and helpers (fully offline).

The routers are pure functions that read `state["intent"]`, so we exercise every
branch directly with no LLM and no graph run — the payoff of keeping the routing
decision separate from the classify node that produces the intent.
"""

from langchain_core.messages import AIMessage, HumanMessage

from watercolor_tutor.routing import (
    can_advance,
    can_go_back,
    clamp_step,
    is_ready_signal,
    last_user_message,
    route_after_answer,
    route_after_input,
    route_after_reply,
)
from watercolor_tutor.state import TutorState


def _state(
    messages: list | None = None, step: int = 1, intent: str = "", image_path: str = ""
) -> TutorState:
    return TutorState(
        messages=messages or [],
        step=step,
        awaiting_question=True,
        intent=intent,
        image_path=image_path,
    )


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


# --- navigation bound helpers ------------------------------------------------


def test_can_advance() -> None:
    assert can_advance(1) and can_advance(2)
    assert not can_advance(3)  # last step (TOTAL_STEPS)


def test_can_go_back() -> None:
    assert can_go_back(2) and can_go_back(3)
    assert not can_go_back(1)  # first step


def test_clamp_step_bounds() -> None:
    assert clamp_step(0) == 1  # below the floor
    assert clamp_step(4) == 3  # above the ceiling (TOTAL_STEPS)
    assert clamp_step(2) == 2  # in range, unchanged


# --- route_after_reply: image presence fork ---------------------------------


def test_route_after_reply_with_image_goes_to_vision() -> None:
    assert route_after_reply(_state(image_path="/tmp/wash.png")) == "vision_feedback"


def test_route_after_reply_without_image_classifies() -> None:
    assert route_after_reply(_state(image_path="")) == "classify"


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


def test_route_confused_goes_to_reexplain() -> None:
    assert route_after_input(_state(intent="confused", step=2)) == "reexplain"


def test_route_off_topic_and_progress_go_to_respond() -> None:
    assert route_after_input(_state(intent="off_topic", step=2)) == "respond"
    assert route_after_input(_state(intent="sharing_progress", step=2)) == "respond"


def test_route_skip_ahead_advances_when_steps_remain() -> None:
    assert route_after_input(_state(intent="skip_ahead", step=1)) == "advance"


def test_route_skip_ahead_on_last_step_is_blocked() -> None:
    # BOUNDARY: can't skip past the final step -> graceful respond, no advance.
    assert route_after_input(_state(intent="skip_ahead", step=3)) == "respond"


def test_route_go_back_moves_when_not_first_step() -> None:
    assert route_after_input(_state(intent="go_back", step=2)) == "go_back"


def test_route_go_back_on_first_step_is_blocked() -> None:
    # BOUNDARY: can't go before the first step -> graceful respond, no go_back.
    assert route_after_input(_state(intent="go_back", step=1)) == "respond"


# --- route_after_answer: the v0 "both" fix ----------------------------------


def test_route_after_answer_question_loops_back() -> None:
    assert route_after_answer(_state(intent="question", step=1)) == "await_learner"


def test_route_after_answer_both_advances() -> None:
    assert route_after_answer(_state(intent="both", step=1)) == "advance"


def test_route_after_answer_both_on_final_step_ends() -> None:
    assert route_after_answer(_state(intent="both", step=3)) == "end"
