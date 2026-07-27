"""
ragkit.qa.source_selection
═══════════════════════════

Stage 1 — choose Proposed chunks and emit deterministic generation tasks.

QA generation draws ONLY from the Proposed pedagogical chunks; B1/B2 are never
a QA source. For each of the five question types this module applies the
type-specific eligibility rules from the specification, then greedily selects a
surplus of tasks under two fairness constraints:

    * at most ``max_questions_per_source_chunk`` tasks from one Proposed chunk,
    * softly keep any single lesson below ``max_lesson_fraction`` of a type's
      tasks (relaxed only when it is infeasible to hit the target otherwise).

Selection is fully deterministic given ``random_seed``: the eligible pool is
shuffled with a seeded RNG and then walked in that fixed order.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.qa.select")

# Roughly how many candidates we want overall (240–270 per the spec). With
# ``candidates_per_task`` candidates per task this drives the per-type surplus
# of tasks so the finaliser has slack to hit every quota.
_CANDIDATE_SURPLUS_FACTOR = 1.5


# ════════════════════════════════════════════
# ELIGIBILITY PREDICATES (one per question type)
# ════════════════════════════════════════════


def _has_text(chunk: Dict[str, Any]) -> bool:
    return bool((chunk.get("main_text_ar") or "").strip())


def _named(chunk: Dict[str, Any], key: str) -> List[Any]:
    return list((chunk.get("named_elements") or {}).get(key) or [])


def _eligible_definition(chunk: Dict[str, Any], _cfg: cfg.QAConfig) -> bool:
    if not _has_text(chunk):
        return False
    if chunk.get("content_type") in ("definition", "lesson_intro", "vocabulary"):
        return True
    return bool(_named(chunk, "definitions") or _named(chunk, "vocabulary"))


def _eligible_theorem(chunk: Dict[str, Any], _cfg: cfg.QAConfig) -> bool:
    return _has_text(chunk) and chunk.get("content_type") == "theorem"


def _eligible_formula(chunk: Dict[str, Any], _cfg: cfg.QAConfig) -> bool:
    if not _has_text(chunk):
        return False
    expressions = list(chunk.get("math_expressions") or [])
    return (bool(chunk.get("has_math")) or bool(expressions)) and bool(expressions)


def _eligible_diagram(chunk: Dict[str, Any], _cfg: cfg.QAConfig) -> bool:
    if not chunk.get("has_diagram"):
        return False
    diagrams = list(chunk.get("diagrams") or [])
    if not diagrams:
        return False
    return any((d.get("description") or "").strip() for d in diagrams)


def _eligible_worked_example(chunk: Dict[str, Any], config: cfg.QAConfig) -> bool:
    if chunk.get("content_type") not in config.selection.allowed_example_types:
        return False
    text = (chunk.get("main_text_ar") or "").strip()
    if len(text) < config.selection.min_example_characters:
        return False
    # Skip obviously incomplete question-only prompts: require some evidence of a
    # worked method (an equation, or a solution/answer cue) alongside the prompt.
    if list(chunk.get("math_expressions") or []):
        return True
    method_cues = ("الحل", "الحلّ", "الخطوة", "بما أن", "إذن", "نعوض", "المثال")
    return any(cue in text for cue in method_cues)


# Map question type → (eligibility predicate, selection_reason builder).
_ELIGIBILITY: Dict[str, Callable[[Dict[str, Any], cfg.QAConfig], bool]] = {
    "definition_recall": _eligible_definition,
    "theorem_statement": _eligible_theorem,
    "formula_retrieval": _eligible_formula,
    "diagram_dependent": _eligible_diagram,
    "worked_example_reasoning": _eligible_worked_example,
}


def _selection_reason(question_type: str, chunk: Dict[str, Any]) -> str:
    ct = chunk.get("content_type")
    if question_type == "definition_recall":
        if _named(chunk, "definitions"):
            return "named_elements.definitions"
        if _named(chunk, "vocabulary"):
            return "named_elements.vocabulary"
        return f"content_type={ct}"
    if question_type == "theorem_statement":
        return "content_type=theorem"
    if question_type == "formula_retrieval":
        return "has_math/math_expressions"
    if question_type == "diagram_dependent":
        return "has_diagram+description"
    if question_type == "worked_example_reasoning":
        return f"content_type={ct}"
    return "unspecified"


# ════════════════════════════════════════════
# TASK BUILDER
# ════════════════════════════════════════════


def _lesson_of(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("lesson_number") or "")


def _make_task(
    task_index: int,
    question_type: str,
    chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble a single generation task from one or more source chunks."""
    source_chunk_ids: List[str] = []
    source_block_ids: List[str] = []
    source_page_numbers: List[int] = []
    lesson_numbers: List[str] = []
    source_content_types: List[str] = []
    payloads: List[Dict[str, Any]] = []

    for chunk in chunks:
        source_chunk_ids.append(chunk.get("chunk_id"))
        for bid in (chunk.get("source_block_ids") or []):
            if bid not in source_block_ids:
                source_block_ids.append(bid)
        for page in (chunk.get("source_page_numbers") or []):
            if page not in source_page_numbers:
                source_page_numbers.append(page)
        lesson = _lesson_of(chunk)
        if lesson and lesson not in lesson_numbers:
            lesson_numbers.append(lesson)
        ct = chunk.get("content_type")
        if ct and ct not in source_content_types:
            source_content_types.append(ct)
        payloads.append(schemas.build_source_payload(chunk))

    return {
        "task_id": f"task-{task_index:06d}",
        "question_type": question_type,
        "source_chunk_ids": source_chunk_ids,
        "source_block_ids": source_block_ids,
        "source_page_numbers": sorted(p for p in source_page_numbers if isinstance(p, int)),
        "lesson_numbers": lesson_numbers,
        "source_content_types": source_content_types,
        "selection_reason": _selection_reason(question_type, chunks[0]),
        "source_payload": payloads[0] if len(payloads) == 1 else None,
        "source_payloads": payloads,
    }


# ════════════════════════════════════════════
# GREEDY, SEEDED SELECTION FOR A SINGLE TYPE
# ════════════════════════════════════════════


def _select_single_source_type(
    question_type: str,
    chunks: List[Dict[str, Any]],
    config: cfg.QAConfig,
    rng: random.Random,
    chunk_usage: Counter,
    target_tasks: int,
) -> Tuple[List[List[Dict[str, Any]]], Dict[str, Any]]:
    """Return selected chunk-groups (each a 1-item list) + a per-type report."""
    predicate = _ELIGIBILITY[question_type]
    eligible = [c for c in chunks if predicate(c, config)]
    target_tasks = min(target_tasks, len(eligible))

    # Deterministic order: seed once per type so ordering is reproducible and
    # independent of the order types are processed in.
    order = list(range(len(eligible)))
    rng.shuffle(order)
    shuffled = [eligible[i] for i in order]

    max_per_chunk = config.selection.max_questions_per_source_chunk
    lesson_cap = max(1, math.floor(config.selection.max_lesson_fraction * target_tasks))

    selected: List[List[Dict[str, Any]]] = []
    lesson_counts: Counter = Counter()

    # First pass honours the soft per-lesson cap; second pass relaxes it only if
    # we still fall short of the target and eligible chunks remain.
    def _walk(respect_lesson_cap: bool) -> None:
        for chunk in shuffled:
            if len(selected) >= target_tasks:
                return
            cid = chunk.get("chunk_id")
            if chunk_usage[cid] >= max_per_chunk:
                continue
            lesson = _lesson_of(chunk)
            if respect_lesson_cap and lesson and lesson_counts[lesson] >= lesson_cap:
                continue
            selected.append([chunk])
            chunk_usage[cid] += 1
            if lesson:
                lesson_counts[lesson] += 1

    _walk(respect_lesson_cap=True)
    if len(selected) < target_tasks:
        # Reset usage attributed only within this relaxed retry is unnecessary;
        # we simply continue picking previously skipped (lesson-capped) chunks.
        remaining = [c for c in shuffled if chunk_usage[c.get("chunk_id")] < max_per_chunk]
        for chunk in remaining:
            if len(selected) >= target_tasks:
                break
            if any(chunk is grp[0] for grp in selected):
                continue
            cid = chunk.get("chunk_id")
            if chunk_usage[cid] >= max_per_chunk:
                continue
            selected.append([chunk])
            chunk_usage[cid] += 1

    report = {
        "question_type": question_type,
        "eligible": len(eligible),
        "target_tasks": target_tasks,
        "selected_tasks": len(selected),
        "deficit": max(0, target_tasks - len(selected)),
        "lessons": dict(Counter(_lesson_of(g[0]) for g in selected)),
        "content_types": dict(Counter(g[0].get("content_type") for g in selected)),
    }
    return selected, report


# ════════════════════════════════════════════
# TOP-LEVEL STAGE ENTRY POINT
# ════════════════════════════════════════════


def run_source_selection(
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 1 and write the source-selection plan + reports.

    Returns a summary dict (also printed by the CLI). When ``dry_run`` is True no
    files are written; the plan is only computed and summarised.
    """
    layout.ensure()
    chunks = schemas.load_chunks(config.proposed_chunks_path)
    logger.info("Loaded %d Proposed chunks from %s", len(chunks), config.proposed_chunks_path)

    rng = random.Random(config.random_seed)
    chunk_usage: Counter = Counter()
    all_tasks: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    warnings: List[str] = []

    quotas = config.quotas.as_dict()
    task_index = 1

    # Single-source types in canonical QUESTION_TYPES order.
    for question_type in schemas.QUESTION_TYPES:
        quota = quotas[question_type]
        target_tasks = math.ceil(quota * _CANDIDATE_SURPLUS_FACTOR / config.candidates_per_task)
        # A fresh RNG per type keeps ordering independent of prior consumption.
        type_rng = random.Random(config.random_seed + hash(question_type) % 10_000)
        groups, report = _select_single_source_type(
            question_type, chunks, config, type_rng, chunk_usage, target_tasks,
        )
        for group in groups:
            all_tasks.append(_make_task(task_index, question_type, group))
            task_index += 1
        reports.append(report)
        if report["deficit"] > 0:
            warnings.append(
                f"{question_type}: only {report['selected_tasks']} tasks selected "
                f"(target {target_tasks}); {report['eligible']} chunks eligible."
            )

    approx_candidates = len(all_tasks) * config.candidates_per_task
    summary = {
        "total_tasks": len(all_tasks),
        "approx_candidates": approx_candidates,
        "candidates_per_task": config.candidates_per_task,
        "by_type": {r["question_type"]: r["selected_tasks"] for r in reports},
        "reports": reports,
        "warnings": warnings,
    }

    if dry_run:
        logger.info("[dry-run] would write %d tasks (~%d candidates)", len(all_tasks), approx_candidates)
        return summary

    schemas.write_jsonl(layout.source_selection_plan, all_tasks)
    schemas.write_json(layout.config_used, schemas.config_to_dict(config))

    logger.info(
        "Stage 1 complete: %d tasks (~%d candidates) → %s",
        len(all_tasks), approx_candidates, layout.source_selection_plan,
    )
    for report in reports:
        logger.info(
            "  %-26s tasks=%s eligible=%s deficit=%s",
            report["question_type"], report["selected_tasks"],
            report["eligible"], report["deficit"],
        )
    for warning in warnings:
        logger.warning("  %s", warning)

    return summary
