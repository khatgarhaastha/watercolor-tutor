"""Graph wiring — builds and compiles the tutor's StateGraph.

This module is WIRING ONLY. Reading it should explain the whole agent at a
glance: which nodes exist and how control flows between them. The actual work
lives in `nodes/`.

Current shape (v1, full intent set):

    START -> welcome -> teach -> await_learner -> classify -[route_after_input]-> ...

    route_after_input (on intent):
        ready       -> advance   (or END on the last step)
        skip_ahead  -> advance   (or respond, if already on the last step)
        go_back     -> go_back    (or respond, if already on the first step)
        confused    -> reexplain
        question,both -> answer
        off_topic, sharing_progress -> respond
    answer  -[route_after_answer]-> both -> advance/END | question -> await_learner
    advance -> teach     go_back -> teach     reexplain -> await_learner
    respond -> await_learner

`classify` runs the LLM intent classifier and writes `intent` to state; the pure
routers act on it. Navigation bounds live in route_after_input: a blocked skip or
go_back is sent to `respond` for a graceful boundary message rather than running
off the ends of the 3-step lesson.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes.advance import advance
from .nodes.answer import answer
from .nodes.await_learner import await_learner
from .nodes.classify import classify
from .nodes.go_back import go_back
from .nodes.reexplain import reexplain
from .nodes.respond import respond
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
    builder.add_node("go_back", go_back)
    builder.add_node("reexplain", reexplain)
    builder.add_node("respond", respond)

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
        {
            "answer": "answer",
            "advance": "advance",
            "go_back": "go_back",
            "reexplain": "reexplain",
            "respond": "respond",
            "end": END,
        },
    )

    # Second conditional edge: after answering, "both" advances; a plain question
    # loops back to wait. This is the pair that fixes the v0 "both" bug.
    builder.add_conditional_edges(
        "answer",
        route_after_answer,
        {"await_learner": "await_learner", "advance": "advance", "end": END},
    )

    # Step-moving nodes re-teach; reply-and-stay nodes loop back to the learner.
    builder.add_edge("advance", "teach")
    builder.add_edge("go_back", "teach")
    builder.add_edge("reexplain", "await_learner")
    builder.add_edge("respond", "await_learner")
    return builder


def compile_graph() -> CompiledStateGraph:
    """Build and compile the graph into a runnable.

    We compile with a CHECKPOINTER, which is what lets the graph pause at the
    `interrupt()` in await_learner and resume on the next invoke(). InMemorySaver
    holds that state in memory for the life of the process — fine for one CLI
    session (a persistent checkpointer would survive restarts, a later concern).
    """
    return build_graph().compile(checkpointer=InMemorySaver())
