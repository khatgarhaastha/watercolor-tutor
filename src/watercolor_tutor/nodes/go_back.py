"""The go_back node — return to the previous step (the 'go_back'/'revisit' intent).

The mirror image of `advance`: it decrements the step, then `teach` re-teaches the
earlier step. Like advance, it clamps defensively — the router only routes here
when there IS a previous step, so the clamp is belt-and-suspenders.
"""

from ..logging_config import get_logger
from ..routing import clamp_step
from ..state import TutorState

logger = get_logger(__name__)


def go_back(state: TutorState) -> dict:
    """Step back one step (clamped to the first step) and re-teach it."""
    prev_step = clamp_step(state["step"] - 1)
    logger.info("going back step %s -> %s", state["step"], prev_step)
    return {"step": prev_step, "awaiting_question": False}
