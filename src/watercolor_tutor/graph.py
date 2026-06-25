"""Graph wiring — builds and compiles the tutor's StateGraph.

This module is WIRING ONLY. Reading it should explain the whole agent at a
glance: which nodes exist and how control flows between them. The actual work
lives in `nodes/`.

Current shape:

    START -> welcome -> teach -> await_learner --[route_after_input]--> answer
                          ^            ^                              |  advance | end
                          |            |__________ answer ____________|          |
                          |_______________________ advance _____________________|
                                                                            END

`welcome` greets once; `teach` delivers the current step; `await_learner` pauses
for the learner; the conditional edge routes their reply to `answer` (a question,
loop back) , `advance` (ready -> next step), or END (ready on the final step).
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes.advance import advance
from .nodes.answer import answer
from .nodes.await_learner import await_learner
from .nodes.teach import teach
from .nodes.welcome import welcome
from .routing import route_after_input
from .state import TutorState


def build_graph() -> StateGraph:
    """Construct the (uncompiled) StateGraph builder.

    Kept separate from compilation so tests can inspect the builder and so
    later slices can attach a checkpointer at compile time for persistence.
    """
    builder = StateGraph(TutorState)
    builder.add_node("welcome", welcome)
    builder.add_node("teach", teach)
    builder.add_node("await_learner", await_learner)
    builder.add_node("answer", answer)
    builder.add_node("advance", advance)

    builder.add_edge(START, "welcome")
    builder.add_edge("welcome", "teach")
    builder.add_edge("teach", "await_learner")

    # The conditional edge. route_after_input(state) returns a string key; the
    # path_map below translates each key to a destination node — and maps "end"
    # to LangGraph's END sentinel. (Keeping END out of routing.py is why the
    # router stays a plain, dependency-free, easily-tested function.)
    builder.add_conditional_edges(
        "await_learner",
        route_after_input,
        {"answer": "answer", "advance": "advance", "end": END},
    )

    # The two loop-backs that make this a graph rather than a straight line:
    builder.add_edge("answer", "await_learner")  # answered a question -> wait again
    builder.add_edge("advance", "teach")  # moved to next step -> teach it
    return builder


def compile_graph() -> CompiledStateGraph:
    """Build and compile the graph into a runnable.

    We compile with a CHECKPOINTER, which is what lets the graph pause at the
    `interrupt()` in await_learner and resume on the next invoke(). InMemorySaver
    holds that state in memory for the life of the process — fine for one CLI
    session (a persistent checkpointer would survive restarts, a later concern).
    """
    return build_graph().compile(checkpointer=InMemorySaver())
