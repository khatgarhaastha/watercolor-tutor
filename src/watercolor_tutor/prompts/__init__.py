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
    "step. Do NOT declare the lesson complete or celebrate finishing "
    "mid-conversation — the lesson simply ends after the final step. On any step, "
    "end by inviting the learner to ask a question or to tell you when they're "
    "ready to continue.\n"
    "At most once or twice in the WHOLE lesson (never every turn), you may casually "
    "mention the learner can ask to see a reference image of what's being taught."
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
        "bottom. Remind them to let it dry flat. Encourage them to give it a try, "
        "and invite them to ask questions or tell you when they're done. Do NOT "
        "announce that the lesson is finished yet, and do not mention any further step."
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
    "- 'needs_web_info': wants CURRENT or EXTERNAL info the lesson can't cover — "
    "specific products/paint sets/brushes/paper to BUY, prices, where to buy, or "
    "recent online tutorials (e.g. 'what's a good cheap beginner set to buy?', "
    "'how much is cold-press paper?').\n"
    "- 'needs_reference_image': wants to SEE an example or reference of the "
    "technique (e.g. 'show me what a good wash looks like', 'can I see an example?').\n"
    "Three-way distinction: a technique how-to question is 'question' (answered "
    "from the lesson); wanting to BUY something or current prices is 'needs_web_info'; "
    "wanting to SEE an example is 'needs_reference_image'.\n"
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
        "They're describing their progress in words. Respond warmly and "
        "encouragingly, but you have NOT seen any image — do NOT describe or "
        "critique how their painting looks. If they'd like real feedback on it, "
        "invite them to share it with /feedback <path>. Then invite them to continue."
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

# Blanket guard for the `respond` node: it never loads an image, so it must never
# invent or critique visual details of the learner's painting (defensive against
# confabulated feedback — real vision lives in the vision_feedback node).
RESPOND_NO_VISION_GUARD = (
    "You have not seen any image; do not describe or critique how their painting looks."
)

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

# Preamble for the retrieved diagnostics in a vision critique (Slice 2.5). The
# "only if you actually SEE it" rule is the heart of Design A-plus: we retrieve
# the whole step's fault rubric, but the model only surfaces the diagnostics that
# match what's visible — so the critique stays honest and targeted.
VISION_GROUNDING_PREAMBLE = (
    "Studio diagnostics (fault → cause → fix) from the handbook. If — and ONLY if — "
    "you actually SEE one of these problems in the image, name it and give its "
    "specific cause and fix from these notes. Do not list faults that aren't "
    "visible, and don't contradict these notes:"
)

# Instructions for the web_search node (v2 Slice 3b-1). The node assembles the
# learner's question + live search results with f-strings (not .format), so result
# text containing braces is harmless.
WEB_SEARCH_PREAMBLE = (
    "The learner asked for live/external info, and here are current web search "
    "results. Answer their question for an absolute beginner using these results: "
    "be practical and concise, recommend cheap/beginner-friendly options where "
    "relevant, and cite the source links you draw from. If the results don't "
    "actually answer it, say so honestly."
)
WEB_SEARCH_UNAVAILABLE = (
    "Live web search is currently unavailable. Tell the learner you couldn't fetch "
    "live results right now, then give the best general guidance you can from your "
    "own knowledge, and suggest they verify current specifics (prices, products) "
    "themselves."
)

# Step-anchored, BEGINNER-AIMED reference-image queries (v2 Slice 3b-2). Built per
# step (not from the learner's literal words) so results skew toward teaching
# references rather than gallery masterpieces or product photos.
REFERENCE_QUERIES: dict[int, str] = {
    1: "beginner watercolor supplies and palette setup example",
    2: "beginner watercolor brush control thin and thick strokes example",
    3: "beginner watercolor flat wash technique example",
}

# Instruction for the reference node: the LLM does METADATA FILTERING over the raw
# results — picking beginner-appropriate references and skipping the rest — and
# frames them honestly (search results, not curated). Copyright: links only.
REFERENCE_SELECTION_PREAMBLE = (
    "These are web search results for a reference the learner asked to see. Pick "
    "the 1-2 MOST APPROPRIATE for an absolute beginner learning this step: prefer "
    "clear technique demonstrations or tutorials; SKIP finished masterpieces, art "
    "that's for sale, products, and anything unrelated. Present each chosen one as "
    "a one-line description plus its source link (do not reproduce the artwork), and "
    "be honest that these are search results you found, not curated/verified. If "
    "none look suitable, say so and suggest they search the query themselves."
)
REFERENCE_UNAVAILABLE = (
    "Reference image search is currently unavailable. Tell the learner you couldn't "
    "fetch a reference right now, and suggest they search the web for the query "
    "themselves (e.g. on an image-search site)."
)
