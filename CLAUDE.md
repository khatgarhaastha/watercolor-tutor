# CLAUDE.md — Watercolor Tutor

Agentic AI that teaches beginners to paint watercolors, step by step.
Portfolio project. Optimize for readability and reviewability over cleverness.

## Tech stack
- Python 3.11+ (src layout, installed as editable package)
- LangGraph 1.2.x for agent orchestration (StateGraph + compiled graph)
- Anthropic API as the LLM brain (claude models)
- pytest for tests, ruff for lint/format, mypy for type checking, structured logging

## Architecture conventions
- `state.py` defines the LangGraph State (TypedDict + reducers). State is the
  single source of truth passed between nodes.
- `nodes/` holds the work: each node is `fn(state) -> dict` returning ONLY the
  keys it changes. One node per file, named after its responsibility.
- `graph.py` is wiring ONLY — build StateGraph, add nodes/edges, compile. No
  business logic here. Reading it should explain the whole agent.
- `config.py` is the ONLY place that reads environment variables.
- Prompts live in `prompts/`, never inline in node logic.

## Code style
- Type-annotate all public functions. Prefer small, pure, testable functions.
- Comment the WHY, not the what. Explain agentic concepts for a learner.
- Use the structured logger from `logging_config.py`; never use bare `print`.
- Imports: `from langgraph.graph import StateGraph, START, END` (START/END are
  sentinels, not strings). Nodes return partial state dicts.

## Secrets
- The Anthropic API key lives in `.env` (gitignored). NEVER commit `.env`.
- `.env.example` documents required vars with placeholder values.
- Access secrets only through `config.py`.

## Testing
- `pytest` from the repo root. Aim for nodes to be unit-testable without live
  API calls — inject a fake/stub LLM via fixtures in `conftest.py`.
- Don't call the real Anthropic API in tests.

## Workflow
- Build in small, reviewable slices. Don't scaffold ahead of need.
- Run `ruff format && ruff check && mypy && pytest` before considering a slice done.
- Commit only when asked; branch off main for new work.
