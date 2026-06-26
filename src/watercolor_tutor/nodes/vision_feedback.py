"""The vision_feedback node — look at the learner's painting and critique it.

Triggered when the learner shares an image (route_after_reply sees image_path in
state). It loads + encodes the image, asks Claude to look at it IN THE CONTEXT OF
THE CURRENT STEP, and returns step-anchored feedback. The image is consumed once:
we clear image_path and stay on the current step.
"""

from .. import images, llm
from ..logging_config import get_logger
from ..prompts import FEEDBACK_FOCUS, STEP_TITLES, SYSTEM_PROMPT, VISION_FEEDBACK_INSTRUCTION
from ..state import TutorState

logger = get_logger(__name__)


def vision_feedback(state: TutorState) -> dict:
    """Give feedback on the shared image, anchored to the current step."""
    step = state["step"]
    logger.info("vision feedback for step=%s image=%s", step, state["image_path"])

    # Load + base64-encode the image (CLI already validated the path).
    image_b64, media_type = images.load_image(state["image_path"])

    # Anchor the critique to THIS step's focus, not generic praise.
    prompt = VISION_FEEDBACK_INSTRUCTION.format(
        step=step, title=STEP_TITLES[step], focus=FEEDBACK_FOCUS[step]
    )
    feedback = llm.see(SYSTEM_PROMPT, image_b64, media_type, prompt)

    # Consume the image (clear it) and stay on the current step.
    return {"messages": [{"role": "assistant", "content": feedback}], "image_path": ""}
