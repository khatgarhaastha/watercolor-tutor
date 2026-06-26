"""The advance node — moves the lesson on to the next step.

A tiny, deterministic, non-LLM node. The router decides *when* to advance; this
node performs the actual state change. Splitting "decide" (router) from "do"
(node) is what keeps the router a pure, easily-tested function.
"""

from ..logging_config import get_logger
from ..routing import clamp_step
from ..state import TutorState

logger = get_logger(__name__)


def advance(state: TutorState) -> dict:
    """Move to the next step and clear the awaiting flag (we're about to teach).

    The router only routes here when there IS a next step, but we still clamp so
    an out-of-range step is impossible by construction (defense-in-depth).
    """
    next_step = clamp_step(state["step"] + 1)
    logger.info("advancing step %s -> %s", state["step"], next_step)
    return {"step": next_step, "awaiting_question": False}
