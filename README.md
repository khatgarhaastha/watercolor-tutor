# Watercolor Tutor 🎨

An agentic AI that teaches beginners to paint watercolors, step by step — built
with [LangGraph](https://github.com/langchain-ai/langgraph) for orchestration and
the [Anthropic API](https://docs.anthropic.com/) as the reasoning engine.

This is a learning-focused portfolio project. The code favors clarity and small,
reviewable slices over cleverness.

## Architecture

The agent is a LangGraph `StateGraph`. The moving parts:

| File | Responsibility |
|------|----------------|
| `state.py` | The shared `TutorState` passed between nodes |
| `nodes/` | The work — each node is `fn(state) -> partial state` |
| `graph.py` | Wiring only: builds and compiles the graph |
| `config.py` | The single place that reads environment variables |
| `llm.py` | Thin Anthropic client wrapper |
| `logging_config.py` | Structured logging setup |

## Getting started

```bash
# 1. Create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package with dev tooling (editable)
pip install -e ".[dev]"

# 3. Configure secrets
cp .env.example .env   # then edit .env and add your real ANTHROPIC_API_KEY

# 4. Run the agent
python -m watercolor_tutor

# 5. Run the checks
ruff format && ruff check && mypy && pytest
```

## Status

Early scaffold. The graph currently has a single welcome node; lesson nodes and
LLM-driven instruction are added in later slices.
