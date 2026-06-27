"""Shared conversation mechanics — the resume-and-read-back cycle.

Both front ends drive the SAME compiled graph the SAME way: invoke it (to start a
lesson or to resume a pause), let it run to the next `interrupt()`, then read back
what the tutor newly said and where the lesson now stands. The terminal app does
this in a blocking loop; the web API does it once per HTTP request. Keeping the
mechanics here means the two entry points can never drift apart.

The graph's pause/resume only works because it's compiled with a CHECKPOINTER:
state is saved at each pause and restored on the next invoke(), keyed by the
`thread_id` in the run config. That's also what makes the API able to be stateless
per request — all continuity lives in the checkpointer, not in the process.
"""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command


def initial_state(name: str = "") -> dict:
    """A fresh, empty tutor state for a brand-new session.

    `name` is the learner's display name (web UI only); it persists in the
    checkpoint so the session picker can show it. The CLI leaves it empty.
    """
    return {
        "messages": [],
        "step": 0,
        "awaiting_question": False,
        "intent": "",
        "image_path": "",
        "name": name,
    }


def message_role(message: Any) -> str:
    """Role across both shapes the state can hold (dict seed vs LangChain object)."""
    return getattr(message, "type", None) or message.get("role")


def message_text(message: Any) -> str:
    """Text content across both shapes."""
    return getattr(message, "content", None) or message["content"]


def _is_assistant(message: Any) -> bool:
    """The tutor's turns appear as 'assistant' (dict seed) or 'ai' (LangChain)."""
    return message_role(message) in ("ai", "assistant")


def normalized_role(message: Any) -> str:
    """Role mapped to the public vocabulary the front ends speak ('assistant'/'user').

    Messages reloaded from the checkpointer come back as LangChain objects whose
    `.type` is 'ai'/'human' (not the 'assistant'/'user' we seed with), so normalize
    both vocabularies down to one the API contract can rely on.
    """
    return "assistant" if _is_assistant(message) else "user"


def assistant_texts(messages: list, since: int = 0) -> list[str]:
    """The tutor's messages from index `since` onward, as plain strings.

    `since` is how we return ONLY what was produced this turn: snapshot the message
    count before invoking, then slice from there afterward. The learner's own turns
    are skipped — they already have those.
    """
    return [message_text(m) for m in messages[since:] if _is_assistant(m)]


def last_assistant_text(messages: list) -> str | None:
    """The most recent tutor message — used to re-orient a returning learner."""
    for message in reversed(messages):
        if _is_assistant(message):
            return message_text(message)
    return None


def history(messages: list) -> list[dict]:
    """Full conversation as {role, content} dicts — for rehydrating a session."""
    return [{"role": normalized_role(m), "content": message_text(m)} for m in messages]


def session_status(graph: CompiledStateGraph, config: RunnableConfig) -> str:
    """Where a thread stands.

    - 'absent'   — no saved state for this thread_id yet (never started).
    - 'awaiting' — paused at await_learner's interrupt(), ready for a reply.
    - 'complete' — the lesson reached END; there's nothing left to resume.
    """
    snapshot = graph.get_state(config)
    if not snapshot.values:
        return "absent"
    return "awaiting" if snapshot.next else "complete"


def start_lesson(
    graph: CompiledStateGraph, config: RunnableConfig, name: str = ""
) -> tuple[int, str, list[str]]:
    """Begin a brand-new lesson on this thread (welcome + first teach), then pause.

    Returns (current_step, status, new_assistant_messages).
    """
    state = graph.invoke(initial_state(name), config=config)
    return state["step"], session_status(graph, config), assistant_texts(state["messages"], 0)


def resume_turn(
    graph: CompiledStateGraph, config: RunnableConfig, resume: object
) -> tuple[int, str, list[str]]:
    """Feed the learner's reply into the paused graph and run to the next pause.

    `resume` becomes the return value of await_learner's interrupt(): a plain string
    for a typed reply, or a {"text", "image_path"} dict to route an uploaded image
    straight to vision_feedback. Caller must ensure the thread is 'awaiting' first.

    Returns (current_step, status, new_assistant_messages).
    """
    before = len(graph.get_state(config).values["messages"])
    state = graph.invoke(Command(resume=resume), config=config)
    return state["step"], session_status(graph, config), assistant_texts(state["messages"], before)
