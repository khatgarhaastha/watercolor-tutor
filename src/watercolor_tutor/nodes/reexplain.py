"""The reexplain node — re-teach the CURRENT step (the 'confused' intent).

Distinct from `answer`: the learner didn't ask a specific question, they didn't
follow the step. So we re-teach the same step a different way (simpler, with an
analogy) rather than answering a question. Stays on the current step and loops
back to wait for the learner.
"""

from .. import llm, retrieval
from ..logging_config import get_logger
from ..prompts import REEXPLAIN_INSTRUCTION, STEP_PROMPTS, STEP_TITLES, SYSTEM_PROMPT
from ..state import TutorState

logger = get_logger(__name__)


def reexplain(state: TutorState) -> dict:
    """Re-teach the current step in a fresh, simpler way."""
    step = state["step"]
    logger.info("re-explaining step=%s", step)

    # Ground the re-explanation in the same step's technique notes (so the simpler
    # retelling still cites the real specifics rather than improvising them).
    grounding = retrieval.grounding_for(step, STEP_PROMPTS[step])
    instruction = REEXPLAIN_INSTRUCTION.format(step=step, title=STEP_TITLES[step])
    instruction += f"\n\n{grounding}" if grounding else ""
    lesson = llm.generate(SYSTEM_PROMPT, [{"role": "user", "content": instruction}])

    return {
        "messages": [{"role": "assistant", "content": lesson}],
        "awaiting_question": True,
    }
