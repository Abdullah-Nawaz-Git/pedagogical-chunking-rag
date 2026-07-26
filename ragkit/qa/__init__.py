"""
ragkit.qa
═════════

QA-dataset preparation for the B1 / B2 / Proposed retrieval benchmark.

This sub-package builds ONE frozen Arabic question-answer dataset from the
Proposed pedagogical chunks and then maps gold provenance onto all three
retrieval corpora. It deliberately does NOT perform retrieval, vector search,
answer generation for evaluation, or RAGAS scoring — it only *prepares* the
dataset and its gold mappings.

Pipeline stages (each a thin, resumable CLI subcommand):

    1. select        — choose Proposed chunks and emit generation tasks
    2. generate      — ask an LLM provider for candidate QA pairs per task
    3. validate      — deterministic filtering (never a correctness claim)
    4. finalize      — pick exactly 30 per type / 180 total, assign ids
    5. map-gold      — B1 (page overlap), B2 + Proposed (source-block) mappings

See ``ragkit.qa.cli`` for the command-line entry point and
``experiments/build_qa_dataset.py`` for the thin runner.
"""

from __future__ import annotations
