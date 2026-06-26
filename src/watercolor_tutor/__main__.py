"""Entry point: `python -m watercolor_tutor`.

Runs the tutor as an interactive terminal conversation. The graph drives the
lesson; this loop just (a) prints new tutor messages and (b) feeds the learner's
typed replies back in to resume the paused graph.

The pause/resume rhythm:
  1. invoke() runs until the graph hits interrupt() in await_learner, then returns.
  2. We detect the pause (graph.get_state(config).next is non-empty), read input.
  3. invoke(Command(resume=text)) continues from exactly where it paused.
"""

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from . import images
from .config import get_settings
from .graph import compile_graph
from .logging_config import configure_logging, get_logger

FEEDBACK_COMMAND = "/feedback"


def _print_new_assistant_messages(messages: list, already_printed: int) -> int:
    """Print tutor (assistant) messages we haven't shown yet; return the new total.

    Messages may be dicts (our seeds) or LangChain message objects (after the
    add_messages reducer runs), so we read role/content defensively. We skip the
    learner's own messages — they already typed those.
    """
    for message in messages[already_printed:]:
        role = getattr(message, "type", None) or message.get("role")
        if role in ("ai", "assistant"):
            content = getattr(message, "content", None) or message["content"]
            print(f"\n{content}")
    return len(messages)


def _parse_feedback_command(line: str) -> tuple[str, str] | None:
    """If `line` is a /feedback command, return (image_path, message); else None.

    Usage: `/feedback <path> [optional message]`. Pure string parsing — the path
    is validated separately so this stays trivially testable.
    """
    if not line.startswith(FEEDBACK_COMMAND):
        return None
    rest = line[len(FEEDBACK_COMMAND) :].strip()
    path, _, message = rest.partition(" ")
    return path, message.strip()


def main() -> None:
    """Run the interactive watercolor lesson."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)
    logger.info("starting watercolor-tutor model=%s", settings.model)

    graph = compile_graph()
    # The checkpointer needs a thread_id to know which conversation to save and
    # resume across invoke() calls. One id == one session.
    config: RunnableConfig = {"configurable": {"thread_id": "cli-session"}}

    # Kick off: welcome + teach step 1, then the graph pauses at await_learner.
    state = graph.invoke(
        {"messages": [], "step": 0, "awaiting_question": False, "intent": "", "image_path": ""},
        config=config,
    )
    printed = _print_new_assistant_messages(state["messages"], 0)
    print("\n(Tip: share your painting for feedback with  /feedback <path-to-image>)")

    # Loop as long as the graph is paused waiting on the learner. An empty
    # `.next` means the graph reached END — the lesson is complete.
    while graph.get_state(config).next:
        try:
            reply = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nNo worries — come back any time. 🎨")
            return
        if not reply:
            continue

        feedback = _parse_feedback_command(reply)
        if feedback is not None:
            path, message = feedback
            try:
                images.load_image(path)  # validate now; show a friendly error if bad
            except (ValueError, OSError) as exc:
                print(f"\nCouldn't use that image: {exc}")
                continue
            resume: object = {"text": message, "image_path": path}
        else:
            resume = reply

        # Resume the paused graph, feeding the reply into await_learner's interrupt().
        state = graph.invoke(Command(resume=resume), config=config)
        printed = _print_new_assistant_messages(state["messages"], printed)

    print("\n🎉 That's your first watercolor lesson — happy painting!")


if __name__ == "__main__":
    main()
