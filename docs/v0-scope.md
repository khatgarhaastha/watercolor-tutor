# Watercolor Tutor — v0 Scope

## Description

A terminal-based tutor that guides a beginner through their **first watercolor
painting in 3 steps** (materials → basic brush control → first simple wash).
It is **linear but interactive**: at each step it teaches, then lets the learner
ask questions and answers them before moving on. **Text-only.**

This is the smallest version that is genuinely a *tutor* (not just a script),
while introducing the one agentic concept we want next: a Q&A side-loop driven
by conditional edges.

## In scope (v0)

- **3-step linear lesson** — materials → brush control → first wash, delivered in
  a fixed order. Matches the arc the welcome message already promises.
- **Q&A side-loop at each step** — after teaching a step, the learner can ask
  questions; the tutor answers (via Claude) and only advances when they're ready.
  Implemented with LangGraph conditional edges.
- **Text-only interaction** — a terminal conversation. No images in or out.

## Out of scope (deferred to later milestones)

| Feature | Why deferred |
|---------|--------------|
| **Adaptive / branching difficulty** (re-explain vs skip ahead based on the learner's level) | Keep v0's flow linear and predictable so we nail teaching quality before adding routing complexity. |
| **Broader curriculum** (color mixing, wet-on-wet, layering, multi-session) | v0 proves one complete lesson end-to-end; breadth is cheap to add once the teaching loop is solid. |
| **Vision / photo feedback** (learner uploads a photo of their work) | Adds image I/O and vision handling + testing; the core text teaching loop must work first. |
| **"Describe any inspiration"** (tailoring the lesson to a learner's reference/idea) | Personalization is meaningful only after the baseline lesson is good; avoids scope creep in v0. |
| **RAG** (retrieval over a knowledge base of techniques) | The 3-step beginner content fits in prompts; no corpus to retrieve over yet. Revisit when curriculum grows. |
| **MCP** (external tools/data via Model Context Protocol) | v0 needs no external tools or live data; introduce only when a real integration earns it. |

## Note

These deferrals are deliberate, not oversights. v0 is intentionally the minimal
complete teaching loop; each item above is a planned later milestone, not a gap.
