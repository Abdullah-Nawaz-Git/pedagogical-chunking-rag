"""
ragkit.qa.validation
═════════════════════

Stage 4 — simple, deterministic candidate filtering.

This stage does NOT establish factual correctness. It only applies mechanical
checks (presence, type, length, banned phrases, type-specific provenance sanity,
and normalized-Arabic duplicate detection) and marks survivors as
``validated``. Every rejection is recorded with a reason so nothing is silently
dropped.

The eleven rejection rules mirror the specification exactly; the first duplicate
of any normalized question is kept deterministically and later duplicates are
rejected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .. import config as cfg
from . import normalize, schemas

logger = logging.getLogger("ragkit.qa.validate")

_VALID_TYPES = set(schemas.QUESTION_TYPES)
_EXAMPLE_TYPES = {"example", "worked_example", "explore"}


def _diagram_has_description(candidate: Dict[str, Any]) -> bool:
    for payload in candidate.get("source_payloads") or []:
        for diagram in payload.get("diagrams") or []:
            if (diagram.get("description") or "").strip():
                return True
    return False


def _source_content_types(candidate: Dict[str, Any]) -> List[str]:
    types = list(candidate.get("source_content_types") or [])
    if types:
        return types
    # Fall back to payload content types if the flattened list is absent.
    return [
        p.get("content_type")
        for p in (candidate.get("source_payloads") or [])
        if p.get("content_type")
    ]


def validate_candidate(
    candidate: Dict[str, Any],
    config: cfg.QAConfig,
    seen_questions: Dict[str, str],
) -> Tuple[bool, Optional[str]]:
    """Return ``(passed, rejection_reason)`` for a single candidate.

    ``seen_questions`` maps normalized-question → first candidate id and is
    mutated as questions are accepted, giving deterministic keep-first dedup.
    """
    vcfg = config.validation

    # Only ever validate successfully-generated candidates.
    if candidate.get("status") != schemas.STATUS_GENERATED:
        return False, f"non-generated status: {candidate.get('status')}"

    question = (candidate.get("question_ar") or "").strip()
    answer = (candidate.get("answer_reference_ar") or "").strip()
    qtype = candidate.get("question_type")

    # 1. Missing question or answer.
    if not question or not answer:
        return False, "missing question or answer"

    # 2. Invalid question type.
    if qtype not in _VALID_TYPES:
        return False, f"invalid question_type: {qtype}"

    # 3. Length limits.
    if not (vcfg.min_question_characters <= len(question) <= vcfg.max_question_characters):
        return False, f"question length {len(question)} out of bounds"
    if not (vcfg.min_answer_characters <= len(answer) <= vcfg.max_answer_characters):
        return False, f"answer length {len(answer)} out of bounds"

    # 4. Banned phrases (after Arabic normalization).
    banned = normalize.contains_banned_phrase(question, list(vcfg.banned_question_phrases))
    if banned:
        return False, f"banned phrase: {banned}"

    # 5. Formula question must declare required_formula=true.
    if qtype == "formula_retrieval" and not candidate.get("required_formula"):
        return False, "formula question without required_formula=true"

    # 6. Diagram question must declare required_diagram=true.
    if qtype == "diagram_dependent" and not candidate.get("required_diagram"):
        return False, "diagram question without required_diagram=true"

    # 7. Theorem question must come from a theorem chunk.
    if qtype == "theorem_statement" and "theorem" not in _source_content_types(candidate):
        return False, "theorem question not sourced from a theorem chunk"

    # 8. Diagram question needs a non-empty diagram description in the payload.
    if qtype == "diagram_dependent" and not _diagram_has_description(candidate):
        return False, "diagram question lacks a diagram description in source payload"

    # 9. Worked example must be sourced from example/worked_example/explore.
    if qtype == "worked_example_reasoning":
        if not (_EXAMPLE_TYPES & set(_source_content_types(candidate))):
            return False, "worked example not sourced from example/worked_example/explore"

    # 10. Duplicate normalized question (keep first).
    key = normalize.normalize_for_compare(question)
    if key in seen_questions:
        return False, f"duplicate of {seen_questions[key]}"

    seen_questions[key] = candidate.get("candidate_id", "<unknown>")
    return True, None


_REJECTION_COLUMNS = (
    "candidate_id",
    "task_id",
    "question_type",
    "status",
    "rejection_reason",
    "question_ar",
)


def run_validation(
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 4, writing qa_validated.jsonl and qa_rejections.csv."""
    layout.ensure()
    candidates = schemas.read_jsonl(layout.qa_candidates)
    if not candidates:
        raise FileNotFoundError(
            f"No candidates at {layout.qa_candidates}. Run 'generate' first."
        )

    seen_questions: Dict[str, str] = {}
    validated: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []

    # Deterministic order: candidate_id sort so keep-first dedup is stable.
    for candidate in sorted(candidates, key=lambda c: c.get("candidate_id", "")):
        passed, reason = validate_candidate(candidate, config, seen_questions)
        if passed:
            record = dict(candidate)
            record["status"] = schemas.STATUS_VALIDATED
            validated.append(record)
        else:
            rejections.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "task_id": candidate.get("task_id"),
                    "question_type": candidate.get("question_type"),
                    "status": candidate.get("status"),
                    "rejection_reason": reason,
                    "question_ar": (candidate.get("question_ar") or "")[:200],
                }
            )

    by_type: Dict[str, int] = {}
    for record in validated:
        by_type[record["question_type"]] = by_type.get(record["question_type"], 0) + 1

    summary = {
        "candidates": len(candidates),
        "validated": len(validated),
        "rejected": len(rejections),
        "validated_by_type": by_type,
    }

    if dry_run:
        logger.info("[dry-run] validated=%d rejected=%d", len(validated), len(rejections))
        return summary

    schemas.write_jsonl(layout.qa_validated, validated)
    schemas.write_csv(layout.qa_rejections, rejections, _REJECTION_COLUMNS)
    logger.info(
        "Stage 4 complete: validated=%d rejected=%d → %s",
        len(validated), len(rejections), layout.qa_validated,
    )
    for qtype in schemas.QUESTION_TYPES:
        logger.info("  %-26s validated=%d", qtype, by_type.get(qtype, 0))
    return summary
