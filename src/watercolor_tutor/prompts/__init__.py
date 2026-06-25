"""Prompt and message text, kept out of node logic so it's easy to iterate on."""

WELCOME_MESSAGE = (
    "Welcome to Watercolor Tutor! 🎨\n"
    "We'll learn to paint watercolors together, one small step at a time. "
    "First we'll cover your materials, then basic brush control, then your "
    "first simple wash. Ready when you are!"
)

# The tutor's persona, sent as the `system` prompt on every Claude call. The
# system prompt shapes HOW the model responds across the whole conversation.
SYSTEM_PROMPT = (
    "You are a warm, patient watercolor instructor guiding an absolute beginner "
    "through their very first painting. Teach one small step at a time in plain, "
    "encouraging language. Keep each reply short — a few sentences. Avoid jargon; "
    "if you must use a term, explain it simply. End by inviting the learner to "
    "ask a question or tell you they're ready to continue."
)

# Human-readable titles for each step (handy for logs and UI).
STEP_TITLES: dict[int, str] = {
    1: "Materials & setup",
    2: "Basic brush control",
    3: "Your first simple wash",
}

# What the tutor should cover at each step. The `teach` node sends the matching
# entry to Claude as the user turn; the model turns it into a friendly lesson.
STEP_PROMPTS: dict[int, str] = {
    1: (
        "Teach Step 1 — Materials & setup. Briefly cover the few essentials a "
        "beginner needs (paints, one round brush, watercolor paper, two jars of "
        "water, and a rag or paper towel) and how to set up a simple workspace. "
        "Reassure them that they don't need expensive supplies to start."
    ),
    2: (
        "Teach Step 2 — Basic brush control. Explain how to hold the brush, how "
        "to make a thin line versus a thick stroke by changing pressure, and "
        "suggest one tiny practice exercise on scrap paper."
    ),
    3: (
        "Teach Step 3 — Your first simple wash. Walk them through painting a flat "
        "wash: mixing paint with water, loading the brush, and laying smooth "
        "overlapping strokes from top to bottom. Remind them to let it dry flat."
    ),
}

# How many steps the v0 lesson has. Derived from STEP_PROMPTS so it stays correct
# if we add a step. The router uses this to know when the lesson is complete.
TOTAL_STEPS = len(STEP_PROMPTS)
