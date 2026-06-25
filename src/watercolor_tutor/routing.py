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


def route_after_input(state: TutorState) -> Literal["answer", "advance", "end"]:
    """Decide what happens after the learner replies (this IS the conditional edge).

    - A ready-signal with more steps left -> "advance" (teach the next step)
    - A ready-signal on the final step     -> "end"     (lesson complete)
    - Anything else                        -> "answer"  (treat it as a question)

    Returns a string KEY; `graph.py` maps "end" to LangGraph's END sentinel when
    it wires this up. Keeping the sentinel out of here keeps routing dependency-free.
    """
    learner_said = last_user_message(state) or ""

    if is_ready_signal(learner_said):
        if state["step"] < TOTAL_STEPS:
            logger.info("router: ready -> advance (from step=%s)", state["step"])
            return "advance"
        logger.info("router: ready on final step -> end")
        return "end"

    logger.info("router: treating reply as a question -> answer")
    return "answer"
