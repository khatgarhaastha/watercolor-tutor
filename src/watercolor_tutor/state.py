"""The shared graph state.

In LangGraph, every node receives the State and returns a *partial* update. The
`Annotated[..., reducer]` syntax tells LangGraph HOW to merge a node's update
into the running state instead of overwriting it.
"""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class TutorState(TypedDict):
    """State passed between every node in the tutor graph.

    Attributes:
        messages: Conversation history. The `add_messages` reducer appends new
            messages (and de-duplicates by id) rather than replacing the list.
        step: Which step of the lesson the learner is on. A plain int, so the
            default reducer applies — a node's value simply overwrites it.
    """

    messages: Annotated[list, add_messages]
    step: int
