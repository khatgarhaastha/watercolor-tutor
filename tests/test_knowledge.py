"""Tests for the RAG retriever, running the REAL retriever over the real corpus.

Static embeddings are deterministic, so these assertions are stable. They load
the embedding model once (downloaded on first run, then cached) — that's why they
live apart from the stubbed flow/node tests.
"""

from watercolor_tutor.retrieval import SHARED_STEP, _chunks, retrieve


def test_chunks_are_tagged_by_step() -> None:
    steps = {chunk.step for chunk in _chunks()}
    assert SHARED_STEP in steps  # the 00-diagnostics doc
    assert {1, 2, 3} <= steps  # one per lesson step


def test_step3_ratio_query_returns_the_mix_ratio() -> None:
    notes = " ".join(retrieve(3, "how much pigment to water for the wash mix"))
    assert "1:3" in notes  # a wash specific that lives only in the corpus


def test_step3_angle_query_returns_the_board_angle() -> None:
    notes = " ".join(retrieve(3, "what board angle or tilt for a flat wash"))
    assert "15" in notes


def test_step1_returns_materials_specifics() -> None:
    notes = " ".join(retrieve(1, "what paper and brush should a beginner buy"))
    assert ("size 8" in notes) or ("140 lb" in notes) or ("300 gsm" in notes)


def test_step_scoping_keeps_wash_notes_out_of_step1() -> None:
    # Even with a wash-y query, step 1's candidates exclude the wash doc — so the
    # wash-only ratio can't surface. This is the step-scoping guarantee.
    notes = " ".join(retrieve(1, "flat wash pigment to water ratio", k=5))
    assert "1:3" not in notes


def test_shared_diagnostics_are_eligible_on_any_step() -> None:
    # The 00-diagnostics chunks are shared, so a backrun query works on step 2.
    notes = " ".join(retrieve(2, "feathery blotch backruns and blooms", k=5)).lower()
    assert "backrun" in notes or "bloom" in notes
