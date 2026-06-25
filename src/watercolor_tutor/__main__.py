"""Entry point: `python -m watercolor_tutor`.

Wires together logging, the compiled graph, and a single invocation. Later
slices will turn this into an interactive loop.
"""

from .config import get_settings
from .graph import compile_graph
from .logging_config import configure_logging, get_logger


def main() -> None:
    """Run one pass through the tutor graph and print the conversation."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    logger.info("starting watercolor-tutor model=%s", settings.model)

    graph = compile_graph()
    # Start with empty state; `welcome` greets and sets step=1, then `teach`
    # produces the first lesson. (Step 1 is materials & setup.)
    result = graph.invoke({"messages": [], "step": 0, "awaiting_question": False})

    for message in result["messages"]:
        # Messages may be dicts (our seed) or LangChain message objects later.
        content = getattr(message, "content", None) or message.get("content", "")
        print(content)


if __name__ == "__main__":
    main()
