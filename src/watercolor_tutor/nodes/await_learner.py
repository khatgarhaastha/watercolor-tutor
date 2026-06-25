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
    # On resume, `interrupt` returns the value that was passed in — the learner's
    # text — which we then append to the conversation as a "user" message. The
    # router (the conditional edge) reads this to decide what happens next.
    reply = interrupt("Awaiting the learner's reply")
    return {"messages": [{"role": "user", "content": reply}]}
