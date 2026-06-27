"""The vision_feedback node — look at the learner's painting and critique it.

Triggered when the learner shares an image (route_after_reply sees image_path in
state). It loads + encodes the image, asks Claude to look at it IN THE CONTEXT OF
THE CURRENT STEP, and returns step-anchored feedback. The image is consumed once:
we clear image_path and stay on the current step.
"""

from .. import images, llm, retrieval
from ..logging_config import get_logger
from ..prompts import (
    FEEDBACK_FOCUS,
    STEP_TITLES,
    SYSTEM_PROMPT,
    VISION_FEEDBACK_INSTRUCTION,
    VISION_GROUNDING_PREAMBLE,
)
from ..state import TutorState

logger = get_logger(__name__)

# How many diagnostics chunks to retrieve for a critique — a touch more than
# teaching (3), so the model has the step's technique + the shared fault rubric +
# a principle to match what it sees against.
FEEDBACK_K = 4


def vision_feedback(state: TutorState) -> dict:
    """Give feedback on the shared image, anchored to the current step + corpus."""
    step = state["step"]
    logger.info("vision feedback for step=%s image=%s", step, state["image_path"])

    # Load + base64-encode the image (CLI already validated the path).
    image_b64, media_type = images.load_image(state["image_path"])

    # Anchor the critique to THIS step's focus, not generic praise.
    prompt = VISION_FEEDBACK_INSTRUCTION.format(
        step=step, title=STEP_TITLES[step], focus=FEEDBACK_FOCUS[step]
    )

    # RAG (Design A-plus): retrieve the step's technique + the shared fault rubric
    # via the SAME retrieve() interface teaching uses (FEEDBACK_FOCUS doubles as the
    # query — it already names the faults to look for), then let the model match
    # what it SEES in the image to the relevant diagnostics in ONE vision call. The
    # preamble's "only if you actually see it" rule keeps the critique honest.
    #
    # Deliberately NOT Design B (image-driven retrieval): a two-pass flow that first
    # describes the image, then retrieves diagnostics targeted to that description.
    # Deferred — it doubles model calls and its retrieval quality hinges on the
    # first-pass description. Step-scoped retrieval is enough at this scale.
    notes = retrieval.retrieve(step, FEEDBACK_FOCUS[step], k=FEEDBACK_K)
    if notes:
        prompt += "\n\n" + VISION_GROUNDING_PREAMBLE + "\n\n" + "\n\n---\n\n".join(notes)

    feedback = llm.see(SYSTEM_PROMPT, image_b64, media_type, prompt)

    # Consume the image (clear it) and stay on the current step.
    return {"messages": [{"role": "assistant", "content": feedback}], "image_path": ""}
