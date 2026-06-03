"""Tests for graph wiring and the welcome node.

These run fully offline — the welcome node is deterministic and does not call
the Anthropic API.
"""

from watercolor_tutor.graph import compile_graph
from watercolor_tutor.nodes.welcome import welcome
from watercolor_tutor.state import TutorState


def test_welcome_node_returns_partial_update(initial_state: TutorState) -> None:
    update = welcome(initial_state)
    assert update["step"] == 1
    assert update["messages"][0]["role"] == "assistant"
    assert "Welcome" in update["messages"][0]["content"]


def test_graph_compiles_and_invokes(initial_state: TutorState) -> None:
    graph = compile_graph()
    result = graph.invoke(initial_state)
    # After the welcome node runs, we should be on step 1 with one message.
    assert result["step"] == 1
    assert len(result["messages"]) == 1
