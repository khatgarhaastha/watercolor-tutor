"""Graph wiring — builds and compiles the tutor's StateGraph.

This module is WIRING ONLY. Reading it should explain the whole agent at a
glance: which nodes exist and how control flows between them. The actual work
lives in `nodes/`.

Current shape (v1):

    START -> welcome -> teach -> await_learner -> classify -[route_after_input]-> ...

    route_after_input (on intent):  ready -> advance | question,both -> answer
    answer -[route_after_answer]->  both -> advance  | question      -> await_learner
    advance -> teach (loop)         (ready/both on the last step -> END)

`await_learner` pauses for the learner; `classify` runs the LLM intent classifier
and writes `intent` to state; the pure routers then act on it. "both" (a question
AND a ready-signal) routes to `answer`, and route_after_answer advances afterward
— fixing the v0 case where a mixed reply dropped the question.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes.advance import advance
from .nodes.answer import answer
from .nodes.await_learner import await_learner
from .nodes.classify import classify
from .nodes.teach import teach
from .nodes.welcome import welcome
from .routing import route_after_answer, route_after_input
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
    builder.add_node("classify", classify)
    builder.add_node("answer", answer)
    builder.add_node("advance", advance)

    builder.add_edge(START, "welcome")
    builder.add_edge("welcome", "teach")
    builder.add_edge("teach", "await_learner")
    # After the learner replies, classify their intent before deciding flow.
    builder.add_edge("await_learner", "classify")

    # First conditional edge: route on the classified intent. The path_map
    # translates each string key to a node — and "end" to LangGraph's END
    # sentinel. (Keeping END out of routing.py is why the routers stay plain,
    # dependency-free, easily-tested functions.)
    builder.add_conditional_edges(
        "classify",
        route_after_input,
        {"answer": "answer", "advance": "advance", "end": END},
    )

    # Second conditional edge: after answering, "both" advances; a plain question
    # loops back to wait. This is the pair that fixes the v0 "both" bug.
    builder.add_conditional_edges(
        "answer",
        route_after_answer,
        {"await_learner": "await_learner", "advance": "advance", "end": END},
    )

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
