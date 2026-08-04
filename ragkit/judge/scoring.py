"""
ragkit.judge.scoring
════════════════════

Turns one :class:`~ragkit.judge.corpus.JudgeItem` into atomic score records — one
per eligible dimension — by building the isolated prompt, calling the judge with
bounded retries, and parsing the reply.

Isolation is enforced here from ``JUDGE_DIMENSION_SPECS``: a dimension's prompt is
assembled from ONLY its permitted inputs, so Answer Relevancy never sees the
contexts and no dimension ever sees gold ids, ranks, or mapping status.

Which dimensions run for an item depends on the mode and on availability:

    retrieval_only   only the two context dimensions (need no generated answer)
    generation       all four; a generation dimension with no answer is recorded
                     as ``skipped_no_generated_answer`` (never silently dropped)
    auto             context dimensions always; generation dimensions only when a
                     generated answer exists for that item

Failures are transparent: an exhausted API call is ``api_error``, an unparseable
reply is ``parse_error``, and an out-of-range score is ``invalid_score`` — each
keeps the raw response and never contributes a coerced number to the aggregates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .. import config as cfg
from . import client as client_mod
from . import parser as parser_mod
from . import prompts, schemas
from .corpus import JudgeItem

logger = logging.getLogger("ragkit.judge.scoring")
llm_logger = logging.getLogger("ragkit.judge.llm_calls")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _log_llm_call_event(payload: Dict[str, Any]) -> None:
    """Write one JSONL event to the dedicated judge LLM call logger."""
    try:
        llm_logger.info(json.dumps(payload, ensure_ascii=False))
    except Exception:  # pragma: no cover - logging must never break scoring
        logger.debug("failed to write judge LLM call log event", exc_info=True)


def eligible_dimensions(
    config: cfg.JudgeExperimentConfig,
    item: JudgeItem,
) -> List[str]:
    """Dimensions to score for ``item`` under the run's mode + availability."""
    requested = [d for d in config.dimensions if d in cfg.JUDGE_DIMENSIONS]
    # Preserve canonical order regardless of how ``dimensions`` was ordered.
    requested = [d for d in cfg.JUDGE_DIMENSIONS if d in requested]

    chosen: List[str] = []
    for dim in requested:
        spec = cfg.JUDGE_DIMENSION_SPECS[dim]
        if spec.requires_generation:
            if config.mode == cfg.JUDGE_MODE_RETRIEVAL_ONLY:
                continue
            if config.mode == cfg.JUDGE_MODE_AUTO and not item.answer_available:
                continue
        chosen.append(dim)
    return chosen


def _base_record(
    config: cfg.JudgeExperimentConfig,
    item: JudgeItem,
    dim: str,
    provider: client_mod.JudgeProvider,
) -> Dict[str, Any]:
    """Shared record skeleton, with only the inputs this dimension may see filled."""
    spec = cfg.JUDGE_DIMENSION_SPECS[dim]
    return {
        "qa_id": item.qa_id,
        "system": item.system,
        "system_label": config.system.label,
        "experimental_role": config.system.experimental_role,
        "metric": dim,
        "metric_display": spec.display,
        "question_type": item.question_type,
        "difficulty": item.difficulty,
        "answer_mode": item.answer_mode,
        "required_diagram": item.required_diagram,
        "required_formula": item.required_formula,
        "question_ar": item.question_ar,
        # Only the inputs the dimension was actually shown are recorded, so the
        # row is a faithful audit of what the judge saw.
        "reference_answer_ar": item.reference_answer_ar if spec.needs_reference_answer else "",
        "generated_answer_ar": item.generated_answer_ar if spec.needs_generated_answer else "",
        "answer_available": item.answer_available,
        "judged_context_chunk_ids": list(item.context.chunk_ids) if spec.needs_contexts else [],
        "context_source": item.context.source if spec.needs_contexts else "",
        "context_token_count": item.context.token_count if spec.needs_contexts else 0,
        "context_token_budget": item.context.budget_tokens,
        "context_truncated": item.context.truncated if spec.needs_contexts else False,
        "context_char_count": item.context.char_count if spec.needs_contexts else 0,
        "context_estimated_model_tokens": item.context.estimated_model_tokens if spec.needs_contexts else 0,
        "retrieved_chunk_ids": list(item.retrieved_chunk_ids),
        "retrieved_ranks": list(item.retrieved_ranks),
        "retrieved_scores": list(item.retrieved_scores),
        "score": None,
        "rationale": "",
        "confidence": None,
        "judge_status": "",
        "parse_ok": False,
        "retries_used": 0,
        "judge_provider": provider.name,
        "judge_model": provider.model,
        "prompt_version": config.prompt_version,
        "judge_temperature": config.model.temperature,
        "timestamp_utc": _utc_now(),
        "gold_granularity": item.gold_granularity,
        "mapping_type": item.mapping_type,
        "mapping_method": item.mapping_method,
        "mapping_status": item.mapping_status,
        "mapping_caveat": item.mapping_caveat,
        "warnings": list(item.warnings),
        "raw_response": "",
    }


def _skip_record(
    config: cfg.JudgeExperimentConfig,
    item: JudgeItem,
    dim: str,
    provider: client_mod.JudgeProvider,
    status: str,
    reason: str,
) -> Dict[str, Any]:
    record = _base_record(config, item, dim, provider)
    record["judge_status"] = status
    record["warnings"] = list(record["warnings"]) + [reason]
    return record


def _call_with_retries(
    provider: client_mod.JudgeProvider,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    *,
    log_context: Optional[Dict[str, Any]] = None,
) -> Tuple[client_mod.JudgeRawResult, parser_mod.ParsedScore, int]:
    """Call the judge, retrying transient API/parse/range failures.

    Returns the last raw result, its parse, and the number of retries used. A
    clean ``scored`` result short-circuits; otherwise every attempt is spent
    before the last (failed) outcome is returned for transparent recording.
    """
    attempts = max(1, max_retries + 1)
    last_raw: Optional[client_mod.JudgeRawResult] = None
    last_parsed: Optional[parser_mod.ParsedScore] = None
    base_event = dict(log_context or {})
    prompt_fingerprint = {
        "system_prompt_sha256": _sha256_text(system_prompt),
        "system_prompt_chars": len(system_prompt),
        "system_prompt": system_prompt,
        "user_prompt_sha256": _sha256_text(user_prompt),
        "user_prompt_chars": len(user_prompt),
        "user_prompt": user_prompt,
    }
    for attempt in range(attempts):
        started_at = _utc_now()
        t0 = time.perf_counter()
        raw = provider.judge(system_prompt, user_prompt)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if raw.error:
            last_raw = raw
            last_parsed = parser_mod.ParsedScore(
                status=schemas.STATUS_API_ERROR,
                score=None,
                rationale="",
                confidence=None,
                parse_ok=False,
                error=raw.error,
            )
            _log_llm_call_event(
                {
                    **base_event,
                    **prompt_fingerprint,
                    "event": "judge_llm_call_attempt",
                    "timestamp_utc": started_at,
                    "provider": raw.provider,
                    "model": raw.model,
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "latency_ms": latency_ms,
                    "api_error": raw.error,
                    "raw_response": raw.raw_response,
                    "raw_response_chars": len(raw.raw_response),
                    "parse_status": last_parsed.status,
                    "parse_ok": last_parsed.parse_ok,
                    "parse_error": last_parsed.error,
                }
            )
            continue
        parsed = parser_mod.parse_judge_response(raw.raw_response)
        last_raw, last_parsed = raw, parsed
        _log_llm_call_event(
            {
                **base_event,
                **prompt_fingerprint,
                "event": "judge_llm_call_attempt",
                "timestamp_utc": started_at,
                "provider": raw.provider,
                "model": raw.model,
                "attempt": attempt + 1,
                "max_attempts": attempts,
                "latency_ms": latency_ms,
                "api_error": raw.error,
                "raw_response": raw.raw_response,
                "raw_response_chars": len(raw.raw_response),
                "parse_status": parsed.status,
                "parse_ok": parsed.parse_ok,
                "parse_error": parsed.error,
                "score": parsed.score,
                "confidence": parsed.confidence,
                "rationale": parsed.rationale,
            }
        )
        if parsed.status == schemas.STATUS_SCORED:
            return raw, parsed, attempt
    assert last_raw is not None and last_parsed is not None
    return last_raw, last_parsed, attempts - 1


def score_dimension(
    config: cfg.JudgeExperimentConfig,
    provider: client_mod.JudgeProvider,
    item: JudgeItem,
    dim: str,
) -> Dict[str, Any]:
    """Score one (item, dimension) into a single atomic record."""
    spec = cfg.JUDGE_DIMENSION_SPECS[dim]

    if spec.needs_generated_answer and not item.answer_available:
        return _skip_record(
            config, item, dim, provider,
            schemas.STATUS_SKIPPED_NO_ANSWER,
            "no generated answer available (run retrieval with --generate-answers)",
        )
    if spec.needs_contexts and not item.context.context_texts:
        return _skip_record(
            config, item, dim, provider,
            schemas.STATUS_SKIPPED_NO_CONTEXT,
            "no retrieved context available to judge",
        )

    user_prompt = prompts.build_user_prompt(
        dim,
        question_ar=item.question_ar,
        reference_answer_ar=item.reference_answer_ar,
        generated_answer_ar=item.generated_answer_ar,
        context_texts=item.context.context_texts,
    )
    raw, parsed, retries_used = _call_with_retries(
        provider,
        prompts.SHARED_JUDGE_SYSTEM_PROMPT,
        user_prompt,
        config.max_retries,
        log_context={
            "qa_id": item.qa_id,
            "system": item.system,
            "metric": dim,
            "prompt_version": config.prompt_version,
            "judge_temperature": config.model.temperature,
            "context_source": item.context.source if spec.needs_contexts else "",
            "context_token_count": (
                item.context.token_count if spec.needs_contexts else 0
            ),
            "context_truncated": (
                item.context.truncated if spec.needs_contexts else False
            ),
        },
    )

    record = _base_record(config, item, dim, provider)
    record["score"] = parsed.score
    record["rationale"] = parsed.rationale
    record["confidence"] = parsed.confidence
    record["judge_status"] = parsed.status
    record["parse_ok"] = parsed.parse_ok
    record["retries_used"] = retries_used
    record["raw_response"] = raw.raw_response
    if parsed.error:
        record["warnings"] = list(record["warnings"]) + [f"judge: {parsed.error}"]
    return record


def score_item(
    config: cfg.JudgeExperimentConfig,
    provider: client_mod.JudgeProvider,
    item: JudgeItem,
    already_scored: Set[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    """Score every eligible, not-yet-scored dimension for one item."""
    new_records: List[Dict[str, Any]] = []
    for dim in eligible_dimensions(config, item):
        key = (item.qa_id, item.system, dim)
        if config.resume and key in already_scored:
            continue
        new_records.append(score_dimension(config, provider, item, dim))
    return new_records
