"""The welcome node — the first thing the learner sees.

This is a deliberately simple, deterministic node (no LLM call yet) so the graph
is runnable and testable from the very first slice. LLM-driven lesson nodes are
added later; they'll follow the same `fn(state) -> partial state` shape.
"""

from ..logging_config import get_logger
from ..prompts import WELCOME_MESSAGE
from ..state import TutorState

logger = get_logger(__name__)


def welcome(state: TutorState) -> dict:
    """Greet the learner and mark that we've started at step 1.

    Returns only the keys it changes — LangGraph merges this into the state.
    """
    logger.info("entering welcome node step=%s", state.get("step", 0))
    return {
        "messages": [{"role": "assistant", "content": WELCOME_MESSAGE}],
        "step": 1,
    }
