"""Request/response models for the web API.

These Pydantic models are the API's CONTRACT — the JSON shapes a frontend can rely
on. They're deliberately thin: the agent's real work lives in the graph, and the
API just translates between HTTP and the conversation helpers.
"""

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """Body for POST /sessions. Everything is optional — an empty POST is fine."""

    # An optional human-friendly label; the real thread_id is this plus a random
    # suffix, so every POST /sessions yields a brand-new session (never collides).
    session_id: str | None = Field(
        default=None, description="Optional label prefixed to the generated session id."
    )


class MessageRequest(BaseModel):
    """Body for POST /sessions/{thread_id}/messages — one learner turn of text."""

    text: str = Field(..., description="The learner's reply.")


class Message(BaseModel):
    """One turn in the conversation."""

    role: str = Field(..., description="'assistant' (the tutor) or 'user' (the learner).")
    content: str


class SessionResponse(BaseModel):
    """The standard reply: where the lesson stands plus the relevant messages.

    For a turn (start / message / feedback) `messages` is only what the tutor just
    said; for GET /sessions/{id} it's the full history (so a UI can rehydrate).
    """

    thread_id: str = Field(..., description="The session id; pass it back on every call.")
    step: int = Field(..., description="Current lesson step (1..N; 0 before it starts).")
    status: str = Field(..., description="'awaiting' (ready for input) or 'complete'.")
    messages: list[Message]
