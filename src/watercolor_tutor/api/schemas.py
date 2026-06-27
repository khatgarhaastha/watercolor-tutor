"""Request/response models for the web API.

These Pydantic models are the API's CONTRACT — the JSON shapes a frontend can rely
on. They're deliberately thin: the agent's real work lives in the graph, and the
API just translates between HTTP and the conversation helpers.
"""

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Body for POST /sessions. The name maps to a session (resume-or-create)."""

    # The learner's display name. Slugified into the thread_id, so the SAME name
    # returns the SAME session (resume); a new name starts fresh. Empty -> "Learner".
    name: str | None = Field(default=None, description="The learner's display name.")


class MessageRequest(BaseModel):
    """Body for POST /sessions/{thread_id}/messages — one learner turn of text."""

    text: str = Field(..., description="The learner's reply.")


class Message(BaseModel):
    """One turn in the conversation."""

    role: str = Field(..., description="'assistant' (the tutor) or 'user' (the learner).")
    content: str


class SessionResponse(BaseModel):
    """The standard reply: where the lesson stands plus the relevant messages.

    For a turn (POST messages / feedback) `messages` is only what the tutor just
    said; for POST /sessions and GET /sessions/{id} it's the full history (so a UI
    can render the whole conversation).
    """

    thread_id: str = Field(..., description="The session id; pass it back on every call.")
    name: str = Field("", description="The learner's display name.")
    step: int = Field(..., description="Current lesson step (1..total_steps; 0 before start).")
    total_steps: int = Field(..., description="How many steps the lesson has (for 'Step X of N').")
    status: str = Field(..., description="'awaiting' (ready for input) or 'complete'.")
    messages: list[Message]


class SessionSummary(BaseModel):
    """One row in the session picker (GET /sessions)."""

    thread_id: str
    name: str = Field("", description="Display name, falling back to the thread_id.")
    step: int
    status: str
