"""Graph wiring — builds and compiles the tutor's StateGraph.

This module is WIRING ONLY. Reading it should explain the whole agent at a
glance: which nodes exist and how control flows between them. The actual work
lives in `nodes/`.

Current shape (first slice):

    START -> welcome -> END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes.welcome import welcome
from .state import TutorState


def build_graph() -> StateGraph:
    """Construct the (uncompiled) StateGraph builder.

    Kept separate from compilation so tests can inspect the builder and so
    later slices can attach a checkpointer at compile time for persistence.
    """
    builder = StateGraph(TutorState)
    builder.add_node("welcome", welcome)
    builder.add_edge(START, "welcome")
    builder.add_edge("welcome", END)
    return builder


def compile_graph() -> CompiledStateGraph:
    """Build and compile the graph into a runnable.

    Returns a compiled graph exposing `invoke`/`stream`/`ainvoke`/`astream`.
    """
    return build_graph().compile()
