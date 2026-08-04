"""
ragkit.judge.metrics
═════════════════════

Aggregation for the LLM-as-judge scores. This is a SECONDARY analysis layer: it
never touches gold ids, Hit@k, or MRR (those live in ``ragkit.retrieval.metrics``
and remain the primary evidence). Here we only summarise the model-produced,
per-dimension scores in [0, 1].

Per-item scores arrive as judge records (one record == one (qa_id, system, metric)
score). This module:

    * averages each dimension over any slice (overall / by question type),
    * reports coverage (how many items were actually scored vs errored / skipped),
    * computes the paired treatment-vs-control difference per dimension, with an
      optional bootstrap confidence interval on the paired per-item differences.

All means treat a missing / errored score as *absent* (excluded from the mean but
counted in ``errors``) rather than silently 0, so a provider outage depresses
coverage instead of faking a low score. This mirrors how the retrieval layer keeps
mapping gaps visible.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import config as cfg
from . import schemas


# ════════════════════════════════════════════
# GROUPING HELPERS
# ════════════════════════════════════════════


def _scored_values(records: Sequence[Dict[str, Any]], metric: str) -> List[float]:
    """Valid [0,1] scores for one dimension (errored / null scores excluded)."""
    values: List[float] = []
    for record in records:
        if record.get("metric") != metric:
            continue
        if record.get("judge_status") not in schemas.SCORED_STATUSES:
            continue
        score = record.get("score")
        if score is None:
            continue
        values.append(float(score))
    return values


def _count_for_metric(records: Sequence[Dict[str, Any]], metric: str) -> Dict[str, int]:
    """Coverage counts for one dimension: scored / errored / total attempted."""
    total = 0
    errored = 0
    skipped = 0
    for record in records:
        if record.get("metric") != metric:
            continue
        total += 1
        status = str(record.get("judge_status") or "")
        if status in schemas.FAILURE_STATUSES:
            errored += 1
        elif status not in schemas.SCORED_STATUSES:
            skipped += 1
    return {
        "attempted": total,
        "errored": errored,
        "skipped": skipped,
        "scored": total - errored - skipped,
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return (sum(values) / len(values)) if values else None


# ════════════════════════════════════════════
# AGGREGATION OVER A SLICE
# ════════════════════════════════════════════


def aggregate(
    records: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Dict[str, Any]:
    """Mean + coverage per dimension over a record slice.

    ``records`` may mix dimensions; each dimension is filtered out by ``metric``.
    Distinct qa ids are counted so a reader knows the slice size independent of how
    many dimensions ran.
    """
    qa_ids = {r.get("qa_id") for r in records if r.get("qa_id")}
    by_dimension: Dict[str, Any] = {}
    for metric in dimensions:
        values = _scored_values(records, metric)
        counts = _count_for_metric(records, metric)
        by_dimension[metric] = {
            "mean": _mean(values),
            "n_scored": counts["scored"],
            "n_errored": counts["errored"],
            "n_skipped": counts["skipped"],
            "n_attempted": counts["attempted"],
        }
    return {
        "n_items": len(qa_ids),
        "by_dimension": by_dimension,
    }


def aggregate_by_question_type(
    records: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per question type in the canonical QA type order."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("question_type") or ""), []).append(record)

    ordered: Dict[str, Dict[str, Any]] = {}
    for question_type in cfg.QA_QUESTION_TYPES:
        if question_type in grouped:
            ordered[question_type] = aggregate(grouped[question_type], dimensions)
    for question_type in sorted(set(grouped) - set(cfg.QA_QUESTION_TYPES)):
        ordered[question_type] = aggregate(grouped[question_type], dimensions)
    return ordered


def aggregate_by_field(
    records: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
    field: str,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate scores by a per-record audit field without assuming its values.

    Mapping status, answer availability, and judge status are all retained in the
    ledger.  Keeping the grouping generic makes omissions and provider failures
    visible in the aggregate instead of turning them into silent missing rows.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        value = record.get(field)
        label = str(value).lower() if isinstance(value, bool) else str(value or "unknown")
        grouped.setdefault(label, []).append(record)
    result: Dict[str, Dict[str, Any]] = {}
    for label, group in sorted(grouped.items()):
        result[label] = aggregate(group, dimensions)
        result[label]["n_records"] = len(group)
    return result


def context_statistics(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise measured context use once per (qa_id, system), not per metric."""
    by_item: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        if not record.get("judged_context_chunk_ids"):
            continue
        key = (str(record.get("qa_id") or ""), str(record.get("system") or ""))
        by_item.setdefault(key, record)
    contexts = list(by_item.values())
    token_counts = [float(r.get("context_token_count") or 0) for r in contexts]
    retry_counts = [float(r.get("retries_used") or 0) for r in records]
    return {
        "n_items_with_context": len(contexts),
        "mean_context_tokens": _mean(token_counts),
        "max_context_tokens": max(token_counts) if token_counts else None,
        "n_truncated": sum(bool(r.get("context_truncated")) for r in contexts),
        "truncation_rate": (
            sum(bool(r.get("context_truncated")) for r in contexts) / len(contexts)
            if contexts else None
        ),
        "mean_retries_used": _mean(retry_counts),
        "retrying_records": sum(count > 0 for count in retry_counts),
    }


# ════════════════════════════════════════════
# PAIRED TREATMENT-vs-CONTROL COMPARISON
# ════════════════════════════════════════════


def _score_index(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    """Map (qa_id, metric) -> score for valid scores only, for one system."""
    index: Dict[Tuple[str, str], float] = {}
    for record in records:
        if record.get("judge_status") not in schemas.SCORED_STATUSES:
            continue
        score = record.get("score")
        qa_id = record.get("qa_id")
        metric = record.get("metric")
        if score is None or not qa_id or not metric:
            continue
        index[(str(qa_id), str(metric))] = float(score)
    return index


def paired_differences(
    treatment_records: Sequence[Dict[str, Any]],
    control_records: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Dict[str, List[float]]:
    """Per-dimension list of (treatment - control) differences over items scored
    by BOTH systems for that dimension.

    Pairing on qa_id keeps the comparison honest: an item only contributes to a
    dimension's difference when both systems produced a valid score for it.
    """
    treatment_index = _score_index(treatment_records)
    control_index = _score_index(control_records)

    diffs: Dict[str, List[float]] = {metric: [] for metric in dimensions}
    for (qa_id, metric), t_score in treatment_index.items():
        if metric not in diffs:
            continue
        c_score = control_index.get((qa_id, metric))
        if c_score is None:
            continue
        diffs[metric].append(t_score - c_score)
    return diffs


def _bootstrap_ci(
    diffs: Sequence[float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> Optional[Dict[str, float]]:
    """Percentile bootstrap CI on the MEAN of paired differences.

    Returns ``None`` when disabled (resamples <= 0) or when there is nothing to
    resample, so the report can distinguish "not computed" from "0.0".
    """
    if resamples <= 0 or not diffs:
        return None
    rng = random.Random(seed)
    n = len(diffs)
    means: List[float] = []
    for _ in range(resamples):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lower_idx = int((1.0 - confidence) / 2.0 * resamples)
    upper_idx = int((1.0 + confidence) / 2.0 * resamples) - 1
    lower_idx = max(0, min(lower_idx, resamples - 1))
    upper_idx = max(0, min(upper_idx, resamples - 1))
    return {
        "lower": means[lower_idx],
        "upper": means[upper_idx],
        "confidence": confidence,
        "resamples": resamples,
    }


def compare_systems(
    treatment_records: Sequence[Dict[str, Any]],
    control_records: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
    aggregation: cfg.JudgeAggregationConfig,
    *,
    treatment_name: str,
    control_name: str,
) -> Dict[str, Any]:
    """Full paired comparison for one (treatment, control) pair.

    Per dimension: mean difference, number of paired items, and (optionally) a
    bootstrap CI on the mean difference. The CI is descriptive corroboration, not a
    significance test — the judge is secondary evidence.
    """
    diffs = paired_differences(treatment_records, control_records, dimensions)
    per_dimension: Dict[str, Any] = {}
    for metric in dimensions:
        metric_diffs = diffs.get(metric, [])
        per_dimension[metric] = {
            "mean_difference": _mean(metric_diffs),
            "n_paired": len(metric_diffs),
            "bootstrap_ci": _bootstrap_ci(
                metric_diffs,
                resamples=aggregation.bootstrap_resamples,
                confidence=aggregation.bootstrap_confidence,
                seed=aggregation.bootstrap_seed,
            ),
        }
    return {
        "treatment": treatment_name,
        "control": control_name,
        "direction": f"{treatment_name} minus {control_name}",
        "by_dimension": per_dimension,
    }


# ════════════════════════════════════════════
# SINGLE-SYSTEM SUMMARY OBJECT
# ════════════════════════════════════════════


def build_summary(
    records: Sequence[Dict[str, Any]],
    config: cfg.JudgeExperimentConfig,
    *,
    dimensions: Sequence[str],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact per-system summary written to ``judge_summary_<system>.json``.

    Carries the same protocol / provenance discipline as the retrieval summary so a
    reader can confirm the judge ran identically across systems, and repeats the
    mapping caveat verbatim for B1.
    """
    system = config.system
    summary: Dict[str, Any] = {
        "system": system.name,
        "system_label": system.label,
        "experimental_role": system.experimental_role,
        "evaluation_kind": "llm_as_judge_inspired_by_ragas",
        "note": (
            "Secondary, model-dependent scores. Primary evidence remains the "
            "retrieval-native metrics in retrieval_eval/."
        ),
        # Protocol provenance: proves the judge itself was held constant.
        "judge_protocol": {
            "provider": config.model.provider,
            "model": _judge_model_name(config),
            "temperature": config.model.temperature,
            "max_output_tokens": config.model.max_output_tokens,
            "prompt_version": config.prompt_version,
            "mode": config.mode,
            "context_budget_tokens": config.context.context_budget_tokens,
            "dimensions": list(dimensions),
        },
        "gold_mapping_caveat_for_context": {
            "mapping_type": system.mapping_type,
            "disclaimer": system.mapping_caveat,
            "shown_to_judge": False,
        },
        "overall": aggregate(records, dimensions),
        "by_question_type": aggregate_by_question_type(records, dimensions),
        "by_mapping_status": aggregate_by_field(records, dimensions, "mapping_status"),
        "by_generation_availability": aggregate_by_field(records, dimensions, "answer_available"),
        "by_judge_status": aggregate_by_field(records, dimensions, "judge_status"),
        "context_statistics": context_statistics(records),
    }
    if extra:
        summary.update(extra)
    return summary


def _judge_model_name(config: cfg.JudgeExperimentConfig) -> str:
    import os

    if config.model.provider == cfg.JUDGE_PROVIDER_MOCK:
        return "mock-judge-deterministic-1"
    return os.environ.get(config.model.model_env_var, config.model.model_default)
