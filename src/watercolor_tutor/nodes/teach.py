"""The teach node — delivers the lesson for the learner's current step.

This is our first LLM-backed node. Where `welcome` returned hardcoded text, this
node asks Claude to teach the current step, using the tutor's system prompt plus
the step-specific instruction from `prompts`. Notice the node itself stays tiny:
all the "intelligence" lives in the prompt and the model. That separation —
small nodes, prompts carry the work — is a core agentic-design idea.
"""

from .. import llm
from ..logging_config import get_logger
from ..prompts import STEP_PROMPTS, STEP_TITLES, SYSTEM_PROMPT
from ..state import TutorState

logger = get_logger(__name__)


def teach(state: TutorState) -> dict:
    """Teach the current step, then mark that we're awaiting the learner.

    Returns only the keys it changes (a new assistant message and the
    `awaiting_question` flag); LangGraph merges that into the running state.
    """
    step = state["step"]
    logger.info("teaching step=%s title=%r", step, STEP_TITLES.get(step))

    # Send a single user turn: the instruction for this step. (Conversation
    # context for follow-up questions is threaded in a later slice.) The first
    # message must use the "user" role — an Anthropic API requirement.
    messages = [{"role": "user", "content": STEP_PROMPTS[step]}]
    lesson = llm.generate(SYSTEM_PROMPT, messages)

    return {
        "messages": [{"role": "assistant", "content": lesson}],
        # The step is taught, so the next thing we do is wait for the learner.
        "awaiting_question": True,
    }
