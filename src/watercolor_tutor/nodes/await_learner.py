"""The await_learner node — pauses the graph to collect the learner's reply.

This is the human-in-the-loop pivot. Calling `interrupt()` SUSPENDS the entire
graph mid-run and hands control back to whatever invoked it (our CLI). The graph
later RESUMES when we invoke it again with `Command(resume=<text>)`, at which
point the `interrupt()` call returns that text — as if it had blocked waiting.

This pause/resume across separate invoke() calls only works because the graph is
compiled with a CHECKPOINTER (see graph.py), which saves the state at the pause
so it can be restored and continued.
"""

from langgraph.types import interrupt

from ..logging_config import get_logger
from ..state import TutorState

logger = get_logger(__name__)


def await_learner(state: TutorState) -> dict:
    """Pause for the learner, then record their reply as a user message."""
    logger.info("awaiting learner reply at step=%s", state["step"])

    # Execution suspends HERE until the caller resumes with Command(resume=...).
    # On resume, `interrupt` returns the value passed in. The CLI sends either a
    # plain string (a typed reply) or a dict (a /feedback command carrying an
    # image path). We normalize both into a user message — and, for an image,
    # also write image_path so route_after_reply sends us to vision_feedback.
    reply = interrupt("Awaiting the learner's reply")

    if isinstance(reply, dict):
        text = reply.get("text") or "[shared a photo of my painting]"
        update: dict = {"messages": [{"role": "user", "content": text}]}
        if reply.get("image_path"):
            update["image_path"] = reply["image_path"]
        return update

    return {"messages": [{"role": "user", "content": reply}]}
