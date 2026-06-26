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
        awaiting_question: True once a step has been taught and we're waiting on
            the learner to either ask a question or signal they're ready to move
            on. The router (added in a later slice) reads this to decide flow.
        intent: The learner's classified intent for their latest message
            ("question" | "ready" | "both"). Written by the `classify` node and
            read by the pure routers — this is how the LLM's understanding of the
            reply is carried from the node that decides it to the edges that act
            on it. Empty string before any classification has run.
    """

    messages: Annotated[list, add_messages]
    step: int
    awaiting_question: bool
    intent: str
