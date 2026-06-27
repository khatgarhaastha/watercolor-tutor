"""Flow-control logic: deciding where the graph goes after the learner replies.

`graph.py` *wires* the graph; this module decides the *path* through it. Keeping
routing separate from node "work" makes the decision logic easy to read and to
unit-test on its own — no LLM call and no graph run required.

The router is a PURE function: it reads the state and returns the NAME of the
next step. It never mutates state (the actual step change lives in the `advance`
node). That purity is what makes every branch trivial to test.
"""

from typing import Literal

from .logging_config import get_logger
from .prompts import TOTAL_STEPS
from .state import TutorState

logger = get_logger(__name__)

# Phrases that signal the learner is ready to move on. This is a deliberately
# simple keyword heuristic for v0 — predictable, free, and trivial to test. A
# smarter LLM-based intent classifier is a planned later upgrade (and the basis
# for the deferred "adaptive branching" milestone). Trade-off to know: an
# ambiguous reply like "i have all the supplies with me" won't match here, so
# it's treated as a comment/question and the tutor will simply re-prompt.
READY_SIGNALS = (
    "ready",
    "next",
    "continue",
    "go on",
    "move on",
    "got it",
    "let's go",
    "lets go",
    "done",
    "i'm good",
    "im good",
    "all set",
    "sounds good",
)


def is_ready_signal(text: str) -> bool:
    """True if the learner's text looks like 'I'm ready to continue'."""
    lowered = text.lower()
    return any(signal in lowered for signal in READY_SIGNALS)


def _role(message: object) -> str:
    """Normalize a message's role across both shapes we encounter.

    Before a graph run, messages are plain dicts ({"role": "user", ...}). During
    a run, the `add_messages` reducer converts them to LangChain message objects
    whose role lives in `.type` ('human'/'ai'). We map both to 'user'/'assistant'
    so callers don't have to care which shape they're looking at.
    """
    if isinstance(message, dict):
        return str(message.get("role", ""))
    type_to_role = {"human": "user", "ai": "assistant", "system": "system"}
    msg_type = str(getattr(message, "type", ""))
    return type_to_role.get(msg_type, msg_type)


def _text(message: object) -> str:
    """Read a message's text content across both shapes (dict or object)."""
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def last_user_message(state: TutorState) -> str | None:
    """Return the text of the learner's most recent message, or None if none."""
    for message in reversed(state["messages"]):
        if _role(message) == "user":
            return _text(message)
    return None


# --- Navigation bounds ------------------------------------------------------
# The lesson is steps 1..TOTAL_STEPS. These pure helpers are the single place
# that knows the edges, so the router never sends the learner past them and the
# move nodes never compute an out-of-range step.


def can_advance(step: int) -> bool:
    """True if there is a step after this one."""
    return step < TOTAL_STEPS


def can_go_back(step: int) -> bool:
    """True if there is a step before this one."""
    return step > 1


def clamp_step(step: int) -> int:
    """Force a step number back into the valid 1..TOTAL_STEPS range.

    Defense-in-depth: the router already refuses out-of-bounds moves, but the
    move nodes clamp too, so an invalid step is impossible by construction.
    """
    return max(1, min(step, TOTAL_STEPS))


def _advance_or_end(step: int) -> Literal["advance", "end"]:
    """Advance to the next step, or end the lesson if we're on the last one."""
    return "advance" if can_advance(step) else "end"


def route_after_reply(state: TutorState) -> Literal["classify", "vision_feedback"]:
    """First fork after the learner replies (pure: reads only image_path).

    An attached image is an unambiguous request for feedback, so it routes
    straight to vision_feedback — we don't need the classifier in that case. Any
    other reply goes to `classify` for the usual intent routing.
    """
    if state["image_path"]:
        logger.info("router: image attached -> vision_feedback")
        return "vision_feedback"
    return "classify"


def route_after_input(
    state: TutorState,
) -> Literal["answer", "advance", "go_back", "reexplain", "respond", "web_search", "end"]:
    """Decide where to go after the learner replies (the first conditional edge).

    Pure: reads only state["intent"] and state["step"]. This is also where the
    navigation guardrails live — a blocked skip/back routes to `respond` for a
    graceful boundary message instead of running off the ends of the lesson.

    Returns a string KEY; `graph.py` maps "end" to LangGraph's END sentinel.
    """
    intent = state["intent"]
    step = state["step"]
    logger.info("routing intent=%s step=%s", intent, step)

    if intent == "ready":
        # Finished the step and ready: advance, or finish on the last step.
        return _advance_or_end(step)
    if intent == "skip_ahead":
        # Jump forward — unless already on the last step (then reassure & stay).
        return "advance" if can_advance(step) else "respond"
    if intent == "go_back":
        # Step back — unless already on the first step (then reassure & stay).
        return "go_back" if can_go_back(step) else "respond"
    if intent == "confused":
        return "reexplain"  # re-teach the current step differently
    if intent == "needs_web_info":
        return "web_search"  # fetch live/external info via the MCP client
    if intent in ("question", "both"):
        return "answer"

    # off_topic, sharing_progress, or any unrecognized label: reply and stay put.
    return "respond"


def route_after_answer(state: TutorState) -> Literal["await_learner", "advance", "end"]:
    """Decide where to go after answering a question (the second conditional edge).

    This is what fixes the v0 bug. If the intent was "both" (a question AND a
    ready-signal), we advance now that the question is answered; otherwise we loop
    back to wait for the learner's next reply.
    """
    if state["intent"] == "both":
        logger.info("router: both -> %s after answering", _advance_or_end(state["step"]))
        return _advance_or_end(state["step"])

    logger.info("router: question answered -> await_learner")
    return "await_learner"
