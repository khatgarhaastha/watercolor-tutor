"""The answer node — responds to a learner's question about the current step.

Like `teach`, this is an LLM-backed node that stays tiny. It pulls the learner's
latest message out of state and asks Claude to answer it, grounded in which step
we're on (passed via the system prompt).

v0 simplification: we send just the question as a single "user" turn (the API
requires the first message to use the "user" role). Threading the full
back-and-forth history for richer follow-ups is a noted later enhancement.
"""

from .. import llm
from ..logging_config import get_logger
from ..prompts import STEP_TITLES, SYSTEM_PROMPT
from ..routing import last_user_message
from ..state import TutorState

logger = get_logger(__name__)


def answer(state: TutorState) -> dict:
    """Answer the learner's most recent question; keep awaiting their next reply."""
    step = state["step"]
    question = last_user_message(state) or ""
    logger.info("answering question at step=%s", step)

    # Ground the answer in the current step via a system-prompt addendum, then
    # send the question itself as the (required) leading "user" turn.
    context = (
        f" The learner is on Step {step}: {STEP_TITLES[step]}. "
        "Answer their question briefly, then invite them to continue when ready."
    )
    reply = llm.generate(SYSTEM_PROMPT + context, [{"role": "user", "content": question}])

    # Note: we deliberately do NOT touch awaiting_question — after answering we
    # still wait for the learner's next message, so it remains True.
    return {"messages": [{"role": "assistant", "content": reply}]}
