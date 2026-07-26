"""
ragkit.qa.generation
═════════════════════

Stage 3 — candidate QA generation from the source-selection plan.

For every task in ``source_selection_plan.jsonl`` the configured provider is
asked for ``candidates_per_task`` candidates. Each raw response is parsed and
turned into a candidate record whose **gold provenance is copied from the task,
never from the model output**. All results are appended to
``qa_candidates.jsonl`` — successes and failures alike — so nothing is silently
discarded, and the run is resumable: tasks already present in the output are
skipped unless ``--force`` is given.

Candidate statuses:
    * ``generated``          — response parsed into a candidate
    * ``parse_failed``       — a response came back but was not valid JSON
    * ``generation_failed``  — the provider reported an error / empty response
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .. import config as cfg
from . import llm_provider, prompts, schemas
from .llm_provider import ProviderResult

logger = logging.getLogger("ragkit.qa.generate")

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Keys the model is trusted to provide (copied through when parseable).
_MODEL_KEYS = schemas.LLM_OUTPUT_FIELDS


def _parse_response(raw: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse a raw model response into a dict, tolerating code fences."""
    text = (raw or "").strip()
    if not text:
        return None, "empty response"
    text = _CODE_FENCE_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"json decode error: {exc}"
    if not isinstance(obj, dict):
        return None, "response was not a JSON object"
    return obj, None


def _build_candidate(
    task: Dict[str, Any],
    candidate_index: int,
    result: ProviderResult,
    config: cfg.QAConfig,
) -> Dict[str, Any]:
    """Build one candidate record, copying provenance from the task."""
    candidate: Dict[str, Any] = {
        "candidate_id": f"{task['task_id']}-c{candidate_index}",
        "task_id": task["task_id"],
        "question_type": task["question_type"],
        # -- gold provenance: ALWAYS from the task, never from the model --
        "gold_source_chunk_ids": list(task.get("source_chunk_ids") or []),
        "gold_source_block_ids": list(task.get("source_block_ids") or []),
        "gold_page_numbers": list(task.get("source_page_numbers") or []),
        "source_content_types": list(task.get("source_content_types") or []),
        "lesson_numbers": list(task.get("lesson_numbers") or []),
        # -- generation provenance --
        "generation_provider": result.provider,
        "generation_model": result.model,
        "generation_prompt_version": result.prompt_version,
        "generation_timestamp_utc": result.timestamp_utc,
        "generation_temperature": result.temperature,
        "source_corpus_version": config.source_corpus_version,
        # -- source evidence retained for validation --
        "source_payloads": task.get("source_payloads")
        or ([task["source_payload"]] if task.get("source_payload") else []),
        # -- raw + diagnostics --
        "raw_response": result.raw_response,
        "parse_error": None,
        "status": schemas.STATUS_GENERATED,
    }

    if result.error:
        candidate["status"] = schemas.STATUS_GENERATION_FAILED
        candidate["parse_error"] = result.error
        return candidate

    parsed, parse_error = _parse_response(result.raw_response)
    if parsed is None:
        candidate["status"] = schemas.STATUS_PARSE_FAILED
        candidate["parse_error"] = parse_error
        return candidate

    # Copy only the model-owned fields, with defensive defaults.
    candidate["question_ar"] = parsed.get("question_ar")
    candidate["question_en"] = parsed.get("question_en")
    candidate["answer_reference_ar"] = parsed.get("answer_reference_ar")
    candidate["difficulty"] = parsed.get("difficulty", "easy")
    candidate["answer_mode"] = parsed.get("answer_mode", "extractive")
    candidate["required_diagram"] = bool(parsed.get("required_diagram", False))
    candidate["required_formula"] = bool(parsed.get("required_formula", False))
    return candidate


def _completed_task_ids(path: Path) -> Set[str]:
    """Task ids already present in an existing candidates file (for resume)."""
    done: Set[str] = set()
    for row in schemas.iter_jsonl(path):
        tid = row.get("task_id")
        if tid:
            done.add(tid)
    return done


def run_generation(
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 3, appending candidate records to qa_candidates.jsonl."""
    layout.ensure()
    tasks = schemas.read_jsonl(layout.source_selection_plan)
    if not tasks:
        raise FileNotFoundError(
            f"No tasks found at {layout.source_selection_plan}. Run 'select' first."
        )

    out_path = layout.qa_candidates
    already_done: Set[str] = set()
    if out_path.exists() and not force:
        already_done = _completed_task_ids(out_path)
        logger.info("Resume: %d tasks already have candidates; they will be skipped", len(already_done))
    elif force and out_path.exists():
        out_path.unlink()
        logger.info("--force: cleared existing %s", out_path)

    pending = [t for t in tasks if t["task_id"] not in already_done]
    provider = llm_provider.build_provider(config)
    logger.info(
        "Generating %d candidates/task for %d pending tasks with provider=%s model=%s",
        config.candidates_per_task, len(pending), provider.name, provider.model,
    )

    counts = {
        schemas.STATUS_GENERATED: 0,
        schemas.STATUS_PARSE_FAILED: 0,
        schemas.STATUS_GENERATION_FAILED: 0,
    }
    written = 0

    if dry_run:
        logger.info("[dry-run] would generate for %d tasks", len(pending))
        return {"pending_tasks": len(pending), "provider": provider.name, "model": provider.model}

    # Append per task so a mid-run failure never loses earlier successes.
    for task in pending:
        try:
            results = provider.generate(task, config.candidates_per_task, config.generation_temperature)
        except Exception as exc:  # noqa: BLE001 - degrade one task to failures, keep going
            logger.warning("provider crashed on %s: %s", task["task_id"], exc)
            results = [
                ProviderResult(
                    raw_response="", provider=provider.name, model=provider.model,
                    temperature=config.generation_temperature,
                    prompt_version=prompts.PROMPT_VERSION, timestamp_utc="",
                    error=f"provider exception: {exc}",
                )
                for _ in range(config.candidates_per_task)
            ]

        candidates = [
            _build_candidate(task, i, result, config) for i, result in enumerate(results)
        ]
        schemas.append_jsonl(out_path, candidates)
        for cand in candidates:
            counts[cand["status"]] = counts.get(cand["status"], 0) + 1
            written += 1

    logger.info(
        "Stage 3 complete: wrote %d candidates → %s (generated=%d parse_failed=%d generation_failed=%d)",
        written, out_path, counts[schemas.STATUS_GENERATED],
        counts[schemas.STATUS_PARSE_FAILED], counts[schemas.STATUS_GENERATION_FAILED],
    )
    return {"written": written, "counts": counts, "provider": provider.name, "model": provider.model}
