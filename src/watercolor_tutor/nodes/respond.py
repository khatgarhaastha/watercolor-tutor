"""The respond node — reply WITHOUT changing the step.

Handles the intents whose right response is "say something and stay put":
off_topic (gently redirect), sharing_progress (acknowledge encouragingly), and
the two navigation BOUNDARY cases (skip-on-last, back-on-first) — the router only
sends skip_ahead/go_back here when the move is blocked at an edge, so the intent
itself tells us which boundary message to give.

Contrast with `answer`, which handles questions and can advance (for "both").
This node never advances — graph.py wires it straight back to await_learner.
"""

from .. import llm
from ..logging_config import get_logger
from ..prompts import RESPOND_NO_VISION_GUARD, RESPONSE_INSTRUCTIONS, STEP_TITLES, SYSTEM_PROMPT
from ..routing import last_user_message
from ..state import TutorState

logger = get_logger(__name__)


def respond(state: TutorState) -> dict:
    """Reply to the learner per their intent's framing, staying on the current step."""
    intent = state["intent"]
    step = state["step"]
    logger.info("responding intent=%s step=%s", intent, step)

    # Pick the framing for this intent; fall back to a gentle off-topic redirect
    # for any label that somehow reaches here without its own framing.
    instruction = RESPONSE_INSTRUCTIONS.get(intent, RESPONSE_INSTRUCTIONS["off_topic"])
    context = (
        f" The learner is on Step {step}: {STEP_TITLES[step]}. "
        f"{instruction} {RESPOND_NO_VISION_GUARD}"
    )

    reply = llm.generate(
        SYSTEM_PROMPT + context, [{"role": "user", "content": last_user_message(state) or ""}]
    )
    return {"messages": [{"role": "assistant", "content": reply}]}
