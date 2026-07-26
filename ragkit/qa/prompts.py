"""
ragkit.qa.prompts
═════════════════

Stage 2 — prompt construction for QA generation.

Every prompt shares a strict system instruction (Arabic-only, source-grounded,
no page/chunk/"above" references, strict-JSON output) and adds a per-type
instruction plus the serialised source evidence taken from the generation task's
``source_payload``. The model is NEVER allowed to set gold provenance ids — those
are copied from the task in code (see ``ragkit.qa.generation``).

The prompt version is recorded on every candidate/final record so a dataset can
be traced back to the exact instruction text that produced it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

PROMPT_VERSION = "v1"

# The shared system instruction sent with every request (verbatim per spec).
SHARED_SYSTEM_PROMPT = """You are creating an Arabic QA item for a Grade 10 mathematics textbook retrieval benchmark.

Use only the supplied source evidence.
Do not use external mathematical knowledge.
Do not invent values, formulas, mathematical relationships, labels, steps, or definitions.
Do not mention page numbers, chunks, "the text", "above", or "the diagram above".
Write a clear Modern Standard Arabic question and a concise Arabic reference answer.
Preserve mathematical notation accurately when necessary.
Return strict JSON only."""

# The exact JSON object the model must return.
OUTPUT_SCHEMA_HINT = """Return ONLY this JSON object, with no text before or after it:

{
  "question_ar": "",
  "question_en": null,
  "answer_reference_ar": "",
  "difficulty": "easy",
  "answer_mode": "extractive",
  "required_diagram": false,
  "required_formula": false
}"""

# Per-type instruction appended to the shared prompt.
_TYPE_INSTRUCTIONS: Dict[str, str] = {
    "definition_recall": (
        "Task type: definition_recall.\n"
        "Ask for a definition that is explicitly supported by the source evidence.\n"
        "The answer must be extractive (answer_mode=\"extractive\")."
    ),
    "theorem_statement": (
        "Task type: theorem_statement.\n"
        "Ask for the complete statement of the theorem, including its condition "
        "and/or formula when present in the source.\n"
        "The answer must be extractive (answer_mode=\"extractive\")."
    ),
    "formula_retrieval": (
        "Task type: formula_retrieval.\n"
        "Ask for a formula that is explicitly stated in the source evidence.\n"
        "Set required_formula=true. The answer must be extractive.\n"
        "Preserve the LaTeX notation of the formula exactly."
    ),
    "diagram_dependent": (
        "Task type: diagram_dependent.\n"
        "Write a question whose answer REQUIRES the supplied diagram description "
        "and/or its labels — it must not be answerable from prose alone.\n"
        "Set required_diagram=true."
    ),
    "worked_example_reasoning": (
        "Task type: worked_example_reasoning.\n"
        "Ask about the method or the steps SHOWN in the worked example — do not "
        "invent a new mathematical exercise.\n"
        "Set answer_mode=\"reasoning\"."
    ),
}


def _format_payload(payload: Dict[str, Any]) -> str:
    """Render one source payload as compact, model-readable evidence."""
    lines: List[str] = []
    lesson = payload.get("lesson_number") or ""
    title = payload.get("lesson_title_ar") or ""
    if lesson or title:
        lines.append(f"Lesson: {lesson} {title}".strip())
    if payload.get("content_type"):
        lines.append(f"Content type: {payload['content_type']}")
    if payload.get("heading_ar"):
        lines.append(f"Heading: {payload['heading_ar']}")
    text = (payload.get("main_text_ar") or "").strip()
    if text:
        lines.append("Text:\n" + text)

    expressions = payload.get("math_expressions") or []
    if expressions:
        lines.append("Formulas (LaTeX): " + json.dumps(expressions, ensure_ascii=False))

    diagrams = payload.get("diagrams") or []
    if diagrams:
        described = []
        for diagram in diagrams:
            desc = (diagram.get("description") or "").strip()
            labels = diagram.get("labels") or []
            if desc or labels:
                described.append({"description": desc, "labels": labels})
        if described:
            lines.append("Diagram(s): " + json.dumps(described, ensure_ascii=False))

    named = payload.get("named_elements") or {}
    named_bits = {k: v for k, v in named.items() if v}
    if named_bits:
        lines.append("Named elements: " + json.dumps(named_bits, ensure_ascii=False))

    return "\n".join(lines)


def build_source_evidence(task: Dict[str, Any]) -> str:
    """Serialise all source payload(s) on a task into an evidence block."""
    payloads = task.get("source_payloads")
    if not payloads:
        single = task.get("source_payload")
        payloads = [single] if single else []
    blocks = []
    for i, payload in enumerate(payloads, start=1):
        header = f"--- Source {i} ---" if len(payloads) > 1 else "--- Source ---"
        blocks.append(header + "\n" + _format_payload(payload))
    return "\n\n".join(blocks)


def build_user_prompt(task: Dict[str, Any]) -> str:
    """Assemble the full user prompt (type instruction + evidence + schema)."""
    question_type = task["question_type"]
    instruction = _TYPE_INSTRUCTIONS.get(question_type, "")
    evidence = build_source_evidence(task)
    return (
        f"{instruction}\n\n"
        f"{evidence}\n\n"
        f"{OUTPUT_SCHEMA_HINT}"
    )
