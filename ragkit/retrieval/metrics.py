"""
ragkit.retrieval.metrics
════════════════════════

The ONE metrics layer shared by Proposed, B1, and B2.

Per-question metrics
--------------------
``Hit@1`` / ``Hit@5``   a gold-relevant chunk appears in the top 1 / top 5.
``Reciprocal rank``     1 / (rank of the first gold-relevant chunk), else 0.
``Gold recall``         fraction of the item's gold TARGETS covered by the
                        retrieved set. A target is "covered" when at least one of
                        the chunks the gold mapping marked relevant for it was
                        retrieved.

The recall definition is where the systems genuinely differ, and the difference
is carried in the data rather than in the control flow:

    * Proposed / B2 targets are **source blocks** → *Gold Unit Recall*.
    * B1 targets are **pages** → *Gold Page Recall (proxy)*, because OCR windows
      keep no source-block provenance.

Both are computed by the same function; the label and granularity travel with
every record (``gold_recall_kind``, ``gold_granularity``) so aggregates can never
silently average a proxy together with true unit recall.

Aggregates
----------
``aggregate`` produces means over any record set (all questions, or one question
type). Hit rates and recall are plain means; MRR is the mean reciprocal rank.
Questions with no gold mapping are counted in ``n`` and score 0, so a mapping gap
depresses the score instead of quietly disappearing.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from .. import config as cfg
from .corpus import GoldTargets
from .engine import RetrievedChunk


# ════════════════════════════════════════════
# PER-QUESTION METRICS
# ════════════════════════════════════════════


def relevant_ranks(
    retrieved: Sequence[RetrievedChunk],
    gold: Optional[GoldTargets],
) -> List[int]:
    """1-based ranks of retrieved chunks that are gold-relevant, ascending."""
    if gold is None:
        return []
    relevant = set(gold.all_relevant_chunk_ids)
    return [hit.rank for hit in retrieved if hit.chunk_id in relevant]


def first_gold_rank(
    retrieved: Sequence[RetrievedChunk],
    gold: Optional[GoldTargets],
) -> Optional[int]:
    """Rank of the first gold-relevant chunk, or ``None`` when none was found."""
    ranks = relevant_ranks(retrieved, gold)
    return min(ranks) if ranks else None


def hit_at_k(rank: Optional[int], k: int) -> bool:
    """Whether the first gold-relevant chunk landed within the top ``k``."""
    return rank is not None and rank <= k


def reciprocal_rank(rank: Optional[int]) -> float:
    """1/rank of the first gold-relevant chunk; 0.0 when nothing relevant hit."""
    return 1.0 / rank if rank else 0.0


def gold_recall(
    retrieved: Sequence[RetrievedChunk],
    gold: Optional[GoldTargets],
) -> Dict[str, Any]:
    """Fraction of gold targets covered by the retrieved chunk set.

    Returns the ratio plus the raw counts and the per-target ids, so a reader can
    audit exactly which units (or, for B1, which pages) were missed. Targets that
    the gold mapping itself could not resolve to any chunk are excluded from the
    denominator and reported in ``unresolved_targets``: they are a mapping gap,
    not a retrieval failure.
    """
    if gold is None or not gold.target_ids:
        return {
            "recall": 0.0,
            "targets_total": 0,
            "targets_covered": 0,
            "covered_target_ids": [],
            "missed_target_ids": [],
            "unresolved_targets": [],
        }

    retrieved_ids = {hit.chunk_id for hit in retrieved}
    covered: List[str] = []
    missed: List[str] = []
    unresolved: List[str] = []

    for target in gold.target_ids:
        relevant_for_target = gold.chunks_by_target.get(target) or set()
        if not relevant_for_target:
            unresolved.append(target)
            continue
        if retrieved_ids & relevant_for_target:
            covered.append(target)
        else:
            missed.append(target)

    denominator = len(covered) + len(missed)
    recall = (len(covered) / denominator) if denominator else 0.0
    return {
        "recall": recall,
        "targets_total": denominator,
        "targets_covered": len(covered),
        "covered_target_ids": covered,
        "missed_target_ids": missed,
        "unresolved_targets": unresolved,
    }


def score_question(
    retrieved: Sequence[RetrievedChunk],
    gold: Optional[GoldTargets],
    config: cfg.RetrievalExperimentConfig,
) -> Dict[str, Any]:
    """Compute every per-question retrieval metric for one QA item."""
    rank = first_gold_rank(retrieved, gold)
    recall = gold_recall(retrieved, gold)

    scores: Dict[str, Any] = {
        "retrieved_ranks_relevant": relevant_ranks(retrieved, gold),
        "first_gold_rank": rank,
        "reciprocal_rank": reciprocal_rank(rank),
        "gold_recall": recall["recall"],
        "gold_targets_total": recall["targets_total"],
        "gold_targets_covered": recall["targets_covered"],
        "gold_missed_target_ids": recall["missed_target_ids"],
        "gold_unresolved_target_ids": recall["unresolved_targets"],
    }
    for k in config.retrieval.hit_at_ks:
        scores[f"hit@{k}"] = hit_at_k(rank, k)
    return scores


# ════════════════════════════════════════════
# AGGREGATION
# ════════════════════════════════════════════


def _mean(values: Sequence[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def aggregate(
    records: Sequence[Dict[str, Any]],
    config: cfg.RetrievalExperimentConfig,
) -> Dict[str, Any]:
    """Aggregate per-question records into means for one slice of the data."""
    system = config.system
    n = len(records)
    summary: Dict[str, Any] = {
        "n": n,
        "mrr": _mean([float(r.get("reciprocal_rank") or 0.0) for r in records]),
        # The recall key is named for what the system can honestly support, and
        # the kind/granularity are repeated so a summary is self-describing.
        "gold_recall_kind": system.recall_metric_name,
        "gold_granularity": system.gold_granularity,
        system.recall_metric_name: _mean(
            [float(r.get("gold_recall") or 0.0) for r in records]
        ),
        "context_token_count_mean": _mean(
            [float(r.get("context_token_count") or 0.0) for r in records]
        ),
        "context_token_count_total": sum(
            int(r.get("context_token_count") or 0) for r in records
        ),
    }
    for k in config.retrieval.hit_at_ks:
        key = f"hit@{k}"
        summary[key] = _mean([1.0 if r.get(key) else 0.0 for r in records])

    # Coverage diagnostics: how much of the slice even had a usable gold mapping.
    summary["questions_with_gold"] = sum(
        1 for r in records if int(r.get("gold_targets_total") or 0) > 0
    )
    summary["questions_without_gold"] = n - summary["questions_with_gold"]
    if system.proxy_disclaimer:
        summary["proxy_disclaimer"] = system.proxy_disclaimer
    return summary


def aggregate_by_question_type(
    records: Sequence[Dict[str, Any]],
    config: cfg.RetrievalExperimentConfig,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate per question type, in the canonical type order."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("question_type") or ""), []).append(record)

    ordered: Dict[str, Dict[str, Any]] = {}
    for question_type in cfg.QA_QUESTION_TYPES:
        if question_type in grouped:
            ordered[question_type] = aggregate(grouped[question_type], config)
    # Any unexpected type still gets reported rather than dropped.
    for question_type in sorted(set(grouped) - set(cfg.QA_QUESTION_TYPES)):
        ordered[question_type] = aggregate(grouped[question_type], config)
    return ordered


def build_summary(
    records: Sequence[Dict[str, Any]],
    config: cfg.RetrievalExperimentConfig,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the compact summary object written to ``retrieval_summary_*.json``."""
    system = config.system
    summary: Dict[str, Any] = {
        "system": system.name,
        "system_label": system.label or system.name,
        # Protocol provenance: proves the comparison was run under equal settings.
        "protocol": {
            "top_k": config.retrieval.top_k,
            "metadata_filter_applied": config.retrieval.use_metadata_filter,
            "query_embedding_model": config.embedding.model,
            "query_embedding_dim": config.embedding.dim,
            "query_task_type": config.retrieval.query_task_type,
            "retriever": config.retriever,
            "index_name": _index_name(config),
            "context_budget_tokens": config.answer.context_budget_tokens,
            "answers_generated": config.generate_answers,
            "answer_model": _answer_model(config),
            "answer_prompt_version": config.answer.prompt_version,
        },
        "gold": {
            "granularity": system.gold_granularity,
            "recall_metric": system.recall_metric_name,
            "mapping_file": system.gold_mapping_filename,
            "disclaimer": system.proxy_disclaimer,
        },
        "overall": aggregate(records, config),
        "by_question_type": aggregate_by_question_type(records, config),
    }
    if extra:
        summary.update(extra)
    return summary


def _index_name(config: cfg.RetrievalExperimentConfig) -> str:
    import os

    return os.environ.get(
        config.system.index_env_var, config.system.default_index_name
    )


def _answer_model(config: cfg.RetrievalExperimentConfig) -> str:
    import os

    if not config.generate_answers:
        return ""
    if config.answer.provider == "mock":
        return "mock-deterministic-1"
    return os.environ.get(config.answer.model_env_var, config.answer.model_default)
