"""Graph wiring — builds and compiles the tutor's StateGraph.

This module is WIRING ONLY. Reading it should explain the whole agent at a
glance: which nodes exist and how control flows between them. The actual work
lives in `nodes/`.

Current shape (v2, full intent set + vision feedback):

    START -> welcome -> teach -> await_learner -[route_after_reply]-> ...

    route_after_reply:  image attached -> vision_feedback ;  else -> classify
    classify -[route_after_input]-> (on intent):
        ready       -> advance   (or END on the last step)
        skip_ahead  -> advance   (or respond, if already on the last step)
        go_back     -> go_back    (or respond, if already on the first step)
        confused    -> reexplain
        question,both -> answer
        off_topic, sharing_progress -> respond
    answer  -[route_after_answer]-> both -> advance/END | question -> await_learner
    advance -> teach     go_back -> teach     reexplain/respond/vision_feedback -> await_learner

`vision_feedback` looks at a shared image in the context of the current step. The
image's presence (set by await_learner from a /feedback command) is the routing
signal — no classifier intent needed, so the v1 machinery is untouched.
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
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
from .nodes.vision_feedback import vision_feedback
from .nodes.welcome import welcome
from .routing import route_after_answer, route_after_input, route_after_reply
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
    builder.add_node("vision_feedback", vision_feedback)

    builder.add_edge(START, "welcome")
    builder.add_edge("welcome", "teach")
    builder.add_edge("teach", "await_learner")
    # After the learner replies, fork: a shared image goes to vision_feedback;
    # any other reply goes to classify for the usual intent routing.
    builder.add_conditional_edges(
        "await_learner",
        route_after_reply,
        {"classify": "classify", "vision_feedback": "vision_feedback"},
    )

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
    builder.add_edge("vision_feedback", "await_learner")
    return builder


def compile_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the graph into a runnable.

    The CHECKPOINTER is what lets the graph pause at the `interrupt()` in
    await_learner and resume on the next invoke(). It's dependency-injected so the
    persistence backend is swappable:
      - default `InMemorySaver` — ephemeral, per-process; keeps tests offline.
      - the CLI passes a `SqliteSaver` — durable, so a session survives restarts.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()
    return build_graph().compile(checkpointer=checkpointer)
