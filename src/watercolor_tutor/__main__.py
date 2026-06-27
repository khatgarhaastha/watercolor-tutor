"""Entry point: `python -m watercolor_tutor`.

Runs the tutor as an interactive terminal conversation with DURABLE state. The
graph is compiled with a SQLite checkpointer, so a session (identified by its
thread_id) survives app restarts: re-run with the same `--session` to resume
exactly where you left off (mid-lesson, same step); `--fresh` starts a new one.

The pause/resume rhythm within a run:
  1. invoke() runs until the graph hits interrupt() in await_learner, then returns.
  2. We detect the pause (graph.get_state(config).next is non-empty), read input.
  3. invoke(Command(resume=text)) continues from exactly where it paused.
"""

import argparse
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.state import CompiledStateGraph

from . import conversation, images
from .config import get_settings
from .graph import compile_graph
from .logging_config import configure_logging, get_logger
from .observability import setup_tracing

# Prefixes that signal an EXPLICIT feedback request (so a bad path errors helpfully
# rather than being silently treated as a normal text reply).
_FEEDBACK_PREFIXES = ("/feedback", "feedback ")


def _print(texts: list[str]) -> None:
    """Show the tutor's new messages (returned by conversation.* helpers)."""
    for text in texts:
        print(f"\n{text}")


def _parse_image_message(line: str) -> tuple[str, str] | None:
    """If `line` references an existing, supported image file, return (path, message).

    Recognizes a bare pasted path, `feedback <path>`, or `/feedback <path>` (with an
    optional trailing message). The plausible-path guard — a token must expand to a
    REAL file with a supported image extension — keeps ordinary text from ever being
    misrouted to vision feedback.
    """
    tokens = line.split()
    for i, token in enumerate(tokens):
        candidate = Path(token).expanduser()
        if candidate.suffix.lower() in images.SUPPORTED_MEDIA_TYPES and candidate.is_file():
            rest = tokens[:i] + tokens[i + 1 :]
            if rest and rest[0].lower().lstrip("/") == "feedback":
                rest = rest[1:]  # drop a leading 'feedback' / '/feedback' word
            return str(candidate), " ".join(rest)
    return None


def _converse(graph: CompiledStateGraph, config: RunnableConfig) -> None:
    """Drive the input loop while the graph is paused awaiting the learner."""
    while graph.get_state(config).next:
        try:
            reply = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nNo worries — your progress is saved. Come back any time. 🎨")
            return
        if not reply:
            continue

        # An image path (bare, or after 'feedback'/'/feedback') -> real vision
        # feedback. Any other message -> the usual intent routing.
        parsed = _parse_image_message(reply)
        if parsed is not None:
            path, message = parsed
            try:
                images.load_image(path)  # validate (size/type) before sending
            except (ValueError, OSError) as exc:
                print(f"\nCouldn't use that image: {exc}")
                continue
            resume: object = {"text": message, "image_path": path}
        elif reply.lower().lstrip().startswith(_FEEDBACK_PREFIXES):
            # Explicit feedback request, but no usable image was found at that path.
            print("\nCouldn't find a usable image there. Try /feedback <path> (jpg/png/webp/gif).")
            continue
        else:
            resume = reply

        # Resume the paused graph, feeding the reply into await_learner's interrupt().
        _, _, new_messages = conversation.resume_turn(graph, config, resume)
        _print(new_messages)

    print("\n🎉 That's a wrap on your first watercolor lesson — you did it! Happy painting!")


def main() -> None:
    """Run the interactive watercolor lesson with durable, resumable sessions."""
    parser = argparse.ArgumentParser(
        prog="watercolor_tutor", description="A step-by-step watercolor tutor."
    )
    parser.add_argument(
        "--session",
        default="default",
        help="Session id (thread). Re-run with the same id to resume that session.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a brand-new session, ignoring any saved state.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    # Enable LangSmith tracing if a key is configured (no-op otherwise). Must run
    # before the first LLM call so the Anthropic client gets wrapped.
    tracing = setup_tracing()

    # thread_id IS the session identity. It's supplied externally (here, the
    # --session flag; a UI could supply it from a name-selection screen instead).
    # --fresh appends a unique suffix so you always get a clean new session.
    thread_id = f"{args.session}-{uuid.uuid4().hex[:8]}" if args.fresh else args.session
    logger.info(
        "starting watercolor-tutor model=%s session=%s db=%s tracing=%s",
        settings.model,
        thread_id,
        settings.db_path,
        tracing,
    )

    # SqliteSaver writes each state snapshot to a file on disk, so the session
    # survives restarts. The `with` block owns the DB connection for this run;
    # setup() creates the tables on first use (idempotent).
    with SqliteSaver.from_conn_string(settings.db_path) as saver:
        saver.setup()
        graph = compile_graph(saver)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        snapshot = graph.get_state(config)
        if snapshot.values and snapshot.next:
            # Returning learner, mid-lesson: resume from the saved snapshot and
            # re-show the last tutor message so they have context.
            print("\nWelcome back — resuming your lesson where you left off. 🎨")
            last = conversation.last_assistant_text(snapshot.values["messages"])
            if last:
                print(f"\n{last}")
        else:
            # New (or already-finished) session: start the lesson fresh.
            _, _, new_messages = conversation.start_lesson(graph, config)
            _print(new_messages)
            print(
                "\n(Tips: share your painting with  /feedback <path>  ·  "
                "or ask to see a reference, e.g. 'can I see an example of a wash?')"
            )

        _converse(graph, config)


if __name__ == "__main__":
    main()
