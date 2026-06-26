"""Prompt and message text, kept out of node logic so it's easy to iterate on."""

WELCOME_MESSAGE = (
    "Welcome to Watercolor Tutor! 🎨\n"
    "We'll learn to paint watercolors together, one small step at a time. "
    "First we'll cover your materials, then basic brush control, then your "
    "first simple wash. Ready when you are!"
)

# Human-readable titles for each step. This is the single source of truth for
# the lesson's shape — the system prompt and the router both derive from it.
STEP_TITLES: dict[int, str] = {
    1: "Materials & setup",
    2: "Basic brush control",
    3: "Your first simple wash",
}

# How many steps the v0 lesson has. Derived from STEP_TITLES so it stays correct
# if we add a step. The router uses this to know when the lesson is complete.
TOTAL_STEPS = len(STEP_TITLES)

# A compact outline of the whole lesson, embedded in the system prompt so the
# model always knows the fixed structure — and, crucially, that there is no step
# beyond the last one (otherwise it tends to invent a "Step 4").
_LESSON_OUTLINE = "; ".join(f"Step {n}: {title}" for n, title in STEP_TITLES.items())

# The tutor's persona, sent as the `system` prompt on every Claude call. The
# system prompt shapes HOW the model responds across the whole conversation.
SYSTEM_PROMPT = (
    "You are a warm, patient watercolor instructor guiding an absolute beginner "
    "through their very first painting. Teach one small step at a time in plain, "
    "encouraging language. Keep each reply short — a few sentences. Avoid jargon; "
    "if you must use a term, explain it simply.\n"
    f"This is a fixed {TOTAL_STEPS}-step first lesson — {_LESSON_OUTLINE}. "
    f"There is no step beyond Step {TOTAL_STEPS}; never invent or reference a later "
    "step. On any step before the last, end by inviting the learner to ask a "
    f"question or say they're ready to continue. On Step {TOTAL_STEPS} (the last "
    "one), after teaching, congratulate them on finishing their first lesson and "
    "invite any final questions — do not point them toward a next step."
)

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
        "Teach Step 3 — Your first simple wash. This is the FINAL step of the "
        "lesson. Walk them through painting a flat wash: mixing paint with water, "
        "loading the brush, and laying smooth overlapping strokes from top to "
        "bottom. Remind them to let it dry flat. Then congratulate them on "
        "completing their first watercolor lesson — do not mention any further step."
    ),
}

# System prompt for the intent classifier. Deliberately short: it's well below
# the model's minimum cacheable prefix (~2048 tokens on Sonnet 4.6), so prompt
# caching would not engage here anyway — short + cheap is the right call.
INTENT_SYSTEM_PROMPT = (
    "You classify a beginner watercolor learner's reply during a fixed 3-step "
    "lesson. Choose exactly one intent:\n"
    "- 'question': asks something specific, not signalling they want to move on "
    "(e.g. 'which brush is best?').\n"
    "- 'ready': signals they want to continue to the next step, no question "
    "(e.g. 'ready', 'next', 'got it', 'all set').\n"
    "- 'both': in one message asks a question AND signals they want to move on "
    "(e.g. 'sounds good, but what brush should I use?').\n"
    "- 'confused': says they don't understand or asks you to explain the current "
    "step again or more simply (e.g. 'I'm lost', 'can you explain that differently?').\n"
    "- 'skip_ahead': wants to jump forward to a later step (e.g. 'can we skip to "
    "the wash?').\n"
    "- 'go_back': wants to return to an earlier step (e.g. 'wait, can we revisit "
    "brushes?').\n"
    "- 'off_topic': not about the painting lesson at all (e.g. 'what's your "
    "favorite color?', small talk).\n"
    "- 'sharing_progress': describes what they painted or how it's going (e.g. "
    "'I just painted a blue sky!').\n"
    "First give one brief sentence of reasoning, then the intent label."
)

# Re-teach instruction for the `reexplain` node (the 'confused' intent). This is
# a re-teach of the CURRENT step, not an answer to a question.
REEXPLAIN_INSTRUCTION = (
    "The learner is confused about Step {step}: {title}. Re-explain THIS step a "
    "different way — simpler, with a concrete example or analogy — without "
    "repeating your earlier wording. Keep it short and encouraging."
)

# Framing for the `respond` node, keyed by intent. These replies never change the
# step: the learner stays where they are and we loop back to wait for them. The
# 'skip_ahead'/'go_back' entries are the graceful BOUNDARY messages — `respond`
# is only reached for those intents when the move is blocked at an edge.
RESPONSE_INSTRUCTIONS: dict[str, str] = {
    "off_topic": (
        "Their message is off-topic for the painting lesson. Warmly and briefly "
        "acknowledge it, then gently steer them back to the current step."
    ),
    "sharing_progress": (
        "They're sharing what they painted or how it's going. Respond with "
        "specific, warm encouragement, then invite them to continue when ready."
    ),
    "skip_ahead": (
        "They want to skip ahead, but they're already on the FINAL step — there "
        "is no next step. Reassure them warmly and encourage them to finish this one."
    ),
    "go_back": (
        "They want to go back, but they're already on the FIRST step — there's "
        "nothing before it. Reassure them warmly and carry on with this step."
    ),
}

# What to look for when giving vision feedback on each step's painting. This is
# what anchors the critique to the CURRENT step instead of generic praise.
FEEDBACK_FOCUS: dict[int, str] = {
    1: "materials & setup — appropriate supplies, paper taped flat, a sensible workspace",
    2: "brush control — line consistency, pressure variation (thin vs thick), steadiness",
    3: "the flat wash — smoothness, even saturation, streaks, hard edges, and blooms",
}

# Prompt for the vision feedback call. The vision_feedback node fills in the
# learner's CURRENT step, its title, and the focus above.
VISION_FEEDBACK_INSTRUCTION = (
    "The learner shares a photo of their watercolor work for Step {step}: {title}. "
    "Give specific, encouraging, step-anchored feedback. Focus on: {focus}. Name "
    "one thing that's working well and one concrete thing to improve. Avoid "
    "generic praise — ground every comment in what you actually see in the image."
)
