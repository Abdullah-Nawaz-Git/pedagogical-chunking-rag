"""
ragkit.qa.finalize
═══════════════════

Stage 5 — assemble the frozen final QA dataset.

Selects exactly ``quota`` validated candidates for every question type (140 in
total by default), assigns stable ids ``qa-0001..qa-0140``, preserves all
provenance, and writes the dataset plus JSON/Markdown summaries. Selection is
deterministic given the seed and enforces one hard constraint — at most two
final questions from any single Proposed source chunk — while treating lesson
balance and difficulty mix as soft preferences.

If any type has fewer than its quota of validated candidates the stage fails
loudly and writes a deficit report instead of a partial "complete" dataset.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.qa.finalize")


class DeficitError(RuntimeError):
    """Raised when a question type has fewer validated candidates than its quota."""


# Fields carried verbatim from a validated candidate onto the final record.
_FINAL_COLUMNS = (
    "qa_id",
    "dataset_version",
    "question_ar",
    "question_en",
    "answer_reference_ar",
    "gold_source_chunk_ids",
    "gold_source_block_ids",
    "gold_page_numbers",
    "source_content_types",
    "lesson_numbers",
    "question_type",
    "difficulty",
    "answer_mode",
    "required_diagram",
    "required_formula",
    "generation_provider",
    "generation_model",
    "generation_prompt_version",
    "generation_timestamp_utc",
    "source_corpus_version",
    "status",
)


def _select_for_type(
    question_type: str,
    candidates: List[Dict[str, Any]],
    quota: int,
    chunk_usage: Counter,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Deterministically choose ``quota`` candidates under the per-chunk cap."""
    pool = sorted(candidates, key=lambda c: c.get("candidate_id", ""))
    rng.shuffle(pool)

    lesson_cap = max(1, math.ceil(0.30 * quota))
    chosen: List[Dict[str, Any]] = []
    lesson_counts: Counter = Counter()
    difficulty_counts: Counter = Counter()

    def _chunk_ids(cand: Dict[str, Any]) -> List[str]:
        return list(cand.get("gold_source_chunk_ids") or [])

    def _fits_chunk_cap(cand: Dict[str, Any]) -> bool:
        return all(chunk_usage[cid] < 2 for cid in _chunk_ids(cand))

    def _primary_lesson(cand: Dict[str, Any]) -> str:
        lessons = cand.get("lesson_numbers") or []
        return str(lessons[0]) if lessons else ""

    def _commit(cand: Dict[str, Any]) -> None:
        chosen.append(cand)
        for cid in _chunk_ids(cand):
            chunk_usage[cid] += 1
        lesson_counts[_primary_lesson(cand)] += 1
        difficulty_counts[str(cand.get("difficulty"))] += 1

    # Pass 1: honour per-chunk cap + soft lesson cap + light difficulty spread.
    for cand in pool:
        if len(chosen) >= quota:
            break
        if not _fits_chunk_cap(cand):
            continue
        if lesson_counts[_primary_lesson(cand)] >= lesson_cap:
            continue
        _commit(cand)

    # Pass 2: relax the soft lesson cap (still respect the hard per-chunk cap).
    if len(chosen) < quota:
        chosen_ids = {c.get("candidate_id") for c in chosen}
        for cand in pool:
            if len(chosen) >= quota:
                break
            if cand.get("candidate_id") in chosen_ids:
                continue
            if not _fits_chunk_cap(cand):
                continue
            _commit(cand)
            chosen_ids.add(cand.get("candidate_id"))

    return chosen[:quota]


def _build_final_record(qa_index: int, candidate: Dict[str, Any], config: cfg.QAConfig) -> Dict[str, Any]:
    return {
        "qa_id": f"qa-{qa_index:04d}",
        "dataset_version": config.dataset_version,
        "question_ar": candidate.get("question_ar", ""),
        "question_en": candidate.get("question_en"),
        "answer_reference_ar": candidate.get("answer_reference_ar", ""),
        "gold_source_chunk_ids": list(candidate.get("gold_source_chunk_ids") or []),
        "gold_source_block_ids": list(candidate.get("gold_source_block_ids") or []),
        "gold_page_numbers": list(candidate.get("gold_page_numbers") or []),
        "source_content_types": list(candidate.get("source_content_types") or []),
        "lesson_numbers": list(candidate.get("lesson_numbers") or []),
        "question_type": candidate.get("question_type", ""),
        "difficulty": candidate.get("difficulty", "easy"),
        "answer_mode": candidate.get("answer_mode", "extractive"),
        "required_diagram": bool(candidate.get("required_diagram", False)),
        "required_formula": bool(candidate.get("required_formula", False)),
        "generation_provider": candidate.get("generation_provider", ""),
        "generation_model": candidate.get("generation_model", ""),
        "generation_prompt_version": candidate.get("generation_prompt_version", "v1"),
        "generation_timestamp_utc": candidate.get("generation_timestamp_utc", ""),
        "source_corpus_version": candidate.get("source_corpus_version", config.source_corpus_version),
        "status": schemas.STATUS_VALIDATED,
    }


def _count_by(records: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter = Counter()
    for record in records:
        value = record.get(key)
        if isinstance(value, list):
            for item in value:
                counter[str(item)] += 1
        else:
            counter[str(value)] += 1
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def _build_summary(records: List[Dict[str, Any]], config: cfg.QAConfig) -> Dict[str, Any]:
    return {
        "dataset_version": config.dataset_version,
        "total": len(records),
        "by_question_type": _count_by(records, "question_type"),
        "by_lesson_number": _count_by(records, "lesson_numbers"),
        "by_source_content_type": _count_by(records, "source_content_types"),
        "by_difficulty": _count_by(records, "difficulty"),
        "by_answer_mode": _count_by(records, "answer_mode"),
        "by_required_diagram": _count_by(records, "required_diagram"),
        "by_required_formula": _count_by(records, "required_formula"),
    }


def _summary_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        f"# QA Dataset Summary ({summary['dataset_version']})",
        "",
        f"Total QA items: **{summary['total']}**",
        "",
    ]

    def _table(title: str, data: Dict[str, int]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|---|---:|")
        for key, count in data.items():
            lines.append(f"| {key} | {count} |")
        lines.append("")

    _table("By question type", summary["by_question_type"])
    _table("By difficulty", summary["by_difficulty"])
    _table("By answer mode", summary["by_answer_mode"])
    _table("By required diagram", summary["by_required_diagram"])
    _table("By required formula", summary["by_required_formula"])
    _table("By source content type", summary["by_source_content_type"])
    _table("By lesson number", summary["by_lesson_number"])
    return "\n".join(lines)


def run_finalize(
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 5, writing the frozen dataset + summaries (or a deficit report)."""
    layout.ensure()
    validated = schemas.read_jsonl(layout.qa_validated)
    if not validated:
        raise FileNotFoundError(
            f"No validated candidates at {layout.qa_validated}. Run 'validate' first."
        )

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in validated:
        by_type[record.get("question_type")].append(record)

    quotas = config.quotas.as_dict()

    # Fail clearly on any deficit BEFORE writing a dataset.
    deficits = {
        qtype: quota - len(by_type.get(qtype, []))
        for qtype, quota in quotas.items()
        if len(by_type.get(qtype, [])) < quota
    }
    if deficits:
        report_lines = [
            "# QA Dataset Deficit Report",
            "",
            "The final dataset was NOT created because one or more question types "
            "have fewer validated candidates than their quota.",
            "",
            "| Question type | Validated | Quota | Deficit |",
            "|---|---:|---:|---:|",
        ]
        for qtype, quota in quotas.items():
            have = len(by_type.get(qtype, []))
            short = max(0, quota - have)
            report_lines.append(f"| {qtype} | {have} | {quota} | {short} |")
        report = "\n".join(report_lines) + "\n"
        if not dry_run:
            layout.qa_deficit_report.write_text(report, encoding="utf-8")
        logger.error("Finalize failed — deficits: %s", deficits)
        raise DeficitError(
            f"Insufficient validated candidates for: {deficits}. "
            f"See {layout.qa_deficit_report}."
        )

    # Deterministic selection.
    chunk_usage: Counter = Counter()
    selected_by_type: Dict[str, List[Dict[str, Any]]] = {}
    for qtype in schemas.QUESTION_TYPES:
        quota = quotas[qtype]
        rng = random.Random(config.random_seed + hash(qtype) % 10_000)
        chosen = _select_for_type(qtype, by_type[qtype], quota, chunk_usage, rng)
        if len(chosen) < quota:
            # Per-chunk cap made the quota infeasible — treat as a hard deficit.
            report = (
                "# QA Dataset Deficit Report\n\n"
                f"Question type '{qtype}' could only supply {len(chosen)}/{quota} "
                "final items under the max-2-per-source-chunk constraint.\n"
            )
            if not dry_run:
                layout.qa_deficit_report.write_text(report, encoding="utf-8")
            raise DeficitError(
                f"'{qtype}' yielded {len(chosen)}/{quota} under the per-chunk cap."
            )
        selected_by_type[qtype] = chosen

    # Assemble final records: canonical type order, then stable candidate order.
    final_records: List[Dict[str, Any]] = []
    qa_index = 1
    for qtype in schemas.QUESTION_TYPES:
        for candidate in sorted(selected_by_type[qtype], key=lambda c: c.get("candidate_id", "")):
            final_records.append(_build_final_record(qa_index, candidate, config))
            qa_index += 1

    summary = _build_summary(final_records, config)

    if dry_run:
        logger.info("[dry-run] would finalize %d records", len(final_records))
        return {"total": len(final_records), "summary": summary}

    schemas.write_jsonl(layout.qa_dataset_jsonl, final_records)
    schemas.write_csv(layout.qa_dataset_csv, final_records, _FINAL_COLUMNS)
    schemas.write_json(layout.qa_dataset_summary_json, summary)
    layout.qa_dataset_summary_md.write_text(_summary_markdown(summary), encoding="utf-8")

    logger.info(
        "Stage 5 complete: wrote %d final QA items → %s",
        len(final_records), layout.qa_dataset_jsonl,
    )
    for qtype in schemas.QUESTION_TYPES:
        logger.info("  %-26s %d", qtype, summary["by_question_type"].get(qtype, 0))
    return {"total": len(final_records), "summary": summary}
