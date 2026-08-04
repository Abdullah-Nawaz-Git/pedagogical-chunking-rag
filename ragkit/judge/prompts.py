"""
ragkit.judge.prompts
════════════════════

The four isolated judge prompts + the shared judge guidance.

Each dimension has its OWN user prompt and is shown ONLY the inputs its
``JudgeDimensionSpec`` permits (``ragkit.config.JUDGE_DIMENSION_SPECS``):

    Context Recall     question + reference answer + ranked contexts
    Context Precision  question + reference answer + ranked contexts
    Faithfulness       question + generated answer + ranked contexts
    Answer Relevancy   question + generated answer            (no contexts)

No prompt is ever handed gold unit ids, gold pages, mapping labels/status, Hit@k
values, or any other gold-derived relevance signal — the judge must reach its own
verdict from the visible evidence alone.

Every prompt requests STRICT JSON only: a normalised ``score`` in [0.0, 1.0], a
short ``rationale``, and an optional ``confidence``. The shared guidance encodes
the project's grading rules (mathematical equivalence, no lexical-overlap
requirement, context-only faithfulness, diagram-claim handling, and preservation
of Arabic / RTL / notation).
"""

from __future__ import annotations

from typing import List, Sequence

from .. import config as cfg

PROMPT_VERSION = "v2"


# ════════════════════════════════════════════
# SHARED JUDGE GUIDANCE (identical for every dimension)
# ════════════════════════════════════════════

SHARED_JUDGE_SYSTEM_PROMPT = """You are a careful, impartial evaluator for an Arabic (Grade 10) mathematics textbook retrieval benchmark.

You produce ONE score in [0.0, 1.0] for a SINGLE evaluation dimension, judging only what you are shown.

General rules (apply to every dimension):
- Accept mathematically equivalent formulas and valid alternative derivations; two forms that are algebraically equal are equally correct.
- Do NOT require exact lexical overlap with the reference answer. Judge meaning, not wording.
- Preserve and respect Arabic and right-to-left text, mathematical formulas, symbols, and variable notation exactly as written; do not penalise correct RTL rendering or LaTeX.
- Do not use external mathematical knowledge to fill gaps that the shown evidence does not contain.
- Penalise unsupported theorem conditions, formula claims, numeric values, or diagram relationships.
- A claim about a diagram counts as supported only when the shown text includes the relevant diagram description, labels, or stated relationship.
- Be decisive but fair: reserve 1.0 for fully satisfying the dimension and 0.0 for entirely failing it; use the range in between for partial cases.

Output rules:
- Return STRICT JSON only, with no text before or after it.
- Use exactly these keys: "score" (number in [0.0, 1.0]), "rationale" (short string), "confidence" (number in [0.0, 1.0] or null).
- Keep the rationale to one or two sentences. Write the rationale in English; quote Arabic/notation verbatim when needed."""


OUTPUT_SCHEMA_HINT = """Return ONLY this JSON object, with no text before or after it:

{
  "score": 0.0,
  "rationale": "",
  "confidence": null
}"""


# Per-dimension instruction. Kept separate from the shared guidance so each
# dimension's specific rubric is explicit and auditable.
_DIMENSION_INSTRUCTIONS = {
    cfg.JUDGE_CONTEXT_RECALL: (
        "Dimension: Context Recall.\n"
        "Question: Do the retrieved contexts CONTAIN the information needed to answer "
        "the question, according to the reference answer?\n"
        "Score 1.0 when every fact the reference answer relies on is present somewhere "
        "in the contexts; score lower as more of that information is missing; score 0.0 "
        "when the contexts contain none of it.\n"
        "Judge sufficiency of evidence, not ranking or wording."
    ),
    cfg.JUDGE_CONTEXT_PRECISION: (
        "Dimension: Context Precision.\n"
        "Question: Are the USEFUL contexts (those that help answer the question per the "
        "reference answer) ranked ABOVE the irrelevant or less-useful ones?\n"
        "The contexts are given in retrieval rank order (rank 1 first). Score 1.0 when "
        "all useful contexts appear before all non-useful ones; score lower as useful "
        "contexts are pushed down below noise; score 0.0 when useful contexts are absent "
        "or ranked last.\n"
        "Judge ordering quality, not whether an answer could be written."
    ),
    cfg.JUDGE_FAITHFULNESS: (
        "Dimension: Faithfulness.\n"
        "Question: Is the generated answer SUPPORTED by the retrieved contexts?\n"
        "Judge faithfulness ONLY from the retrieved contexts, never from outside "
        "mathematical knowledge. Verify every asserted value, formula, condition, "
        "step, and diagram relationship against the shown contexts, claim by claim; "
        "a single unsupported claim lowers the score.\n"
        "If the answer merely states that the context is insufficient (e.g. 'لا يوجد "
        "في السياق المُقدَّم'), treat this as abstention: score at most 0.5 even when "
        "that statement is TRUE, and score 0.0 when the information is in fact present "
        "in the contexts but the answer failed to retrieve or use it."
    ),
    cfg.JUDGE_ANSWER_RELEVANCY: (
        "Dimension: Answer Relevancy.\n"
        "Question: Does the generated answer DIRECTLY answer the question that was asked?\n"
        "Judge only relevance and directness — whether the answer addresses what was "
        "asked without evasion, padding, or drift. Do NOT judge factual correctness or "
        "grounding here (those are other dimensions), and you are deliberately not shown "
        "the retrieved contexts.\n"
        "Score 1.0 for a focused, on-target answer; lower for partial or padded answers; "
        "0.0 for an answer that does not address the question."
    ),
}


# ════════════════════════════════════════════
# INPUT BLOCK RENDERERS
# ════════════════════════════════════════════


def render_contexts(context_texts: Sequence[str]) -> str:
    """Render ranked contexts as a numbered, rank-ordered block.

    Rank markers make ordering explicit for Context Precision without ever
    revealing gold ids: the judge sees ``[rank N]`` and the text, nothing else.
    """
    if not context_texts:
        return "(no retrieved context was provided)"
    lines: List[str] = []
    for rank, text in enumerate(context_texts, start=1):
        body = (text or "").strip() or "(empty context)"
        lines.append(f"[rank {rank}]\n{body}")
    return "\n\n".join(lines)


# ════════════════════════════════════════════
# PROMPT BUILDER
# ════════════════════════════════════════════


def build_user_prompt(
    dimension: str,
    *,
    question_ar: str,
    reference_answer_ar: str = "",
    generated_answer_ar: str = "",
    context_texts: Sequence[str] = (),
) -> str:
    """Assemble the user prompt for ``dimension`` from ONLY its permitted inputs.

    The dimension's ``JudgeDimensionSpec`` decides which sections are rendered, so
    isolation is enforced here (not just by the caller): an input that the spec
    does not flag is never written into the prompt even if a value is passed.
    """
    spec = cfg.JUDGE_DIMENSION_SPECS[dimension]
    instruction = _DIMENSION_INSTRUCTIONS[dimension]

    sections: List[str] = [instruction, ""]

    if spec.needs_question:
        sections += ["--- Question (Arabic) ---", (question_ar or "").strip(), ""]

    if spec.needs_reference_answer:
        sections += [
            "--- Reference answer (Arabic) ---",
            (reference_answer_ar or "").strip() or "(no reference answer provided)",
            "",
        ]

    if spec.needs_generated_answer:
        sections += [
            "--- Generated answer (Arabic) ---",
            (generated_answer_ar or "").strip() or "(no generated answer provided)",
            "",
        ]

    if spec.needs_contexts:
        sections += [
            "--- Retrieved contexts (in retrieval rank order) ---",
            render_contexts(context_texts),
            "",
        ]

    sections.append(OUTPUT_SCHEMA_HINT)
    return "\n".join(sections)
