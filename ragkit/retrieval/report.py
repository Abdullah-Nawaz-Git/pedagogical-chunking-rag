"""
ragkit.retrieval.report
═══════════════════════

Writes the retrieval-evaluation artifacts: per-question detail, a compact summary
JSON, and a Markdown report — for a single system and across all three.

Reporting is the last place a reader could confuse B1's page-overlap proxy for
true Gold Unit Recall, so the tables here never share a "Gold Recall" column
between granularities: unit-level systems and proxy systems get separate columns,
and B1's disclaimer is printed under every table it appears in.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.retrieval.report")

# How a float metric is rendered in Markdown tables.
_PRECISION = 4


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{_PRECISION}f}"
    if value is None or value == "":
        return "—"
    return str(value)


def _hit_keys(config: cfg.RetrievalExperimentConfig) -> List[str]:
    return [f"hit@{k}" for k in config.retrieval.hit_at_ks]


# ════════════════════════════════════════════
# PER-SYSTEM OUTPUT
# ════════════════════════════════════════════


def write_system_outputs(
    config: cfg.RetrievalExperimentConfig,
    layout: schemas.RetrievalOutputLayout,
    records: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Dict[str, str]:
    """Write records (JSONL + CSV), the summary JSON, and the Markdown report."""
    layout.ensure()
    system = config.system.name

    paths = {
        "records_jsonl": schemas.write_jsonl(layout.records_jsonl(system), records),
        "records_csv": schemas.write_csv(
            layout.records_csv(system), records, schemas.RECORD_COLUMNS
        ),
        "summary_json": schemas.write_json(layout.summary_json(system), summary),
        "config_used": schemas.write_json(
            layout.config_used(system), schemas.config_to_dict(config)
        ),
    }
    markdown = system_markdown(config, summary)
    layout.summary_md(system).write_text(markdown, encoding="utf-8")
    paths["summary_md"] = layout.summary_md(system)

    logger.info(
        "Wrote retrieval artifacts for %s: %s",
        system, ", ".join(p.name for p in paths.values()),
    )
    return {key: str(path) for key, path in paths.items()}


def system_markdown(
    config: cfg.RetrievalExperimentConfig,
    summary: Dict[str, Any],
) -> str:
    """Render the per-system Markdown report."""
    system = config.system
    overall = summary["overall"]
    protocol = summary["protocol"]
    recall_key = system.recall_metric_name
    hit_keys = _hit_keys(config)

    lines: List[str] = [
        f"# Retrieval Evaluation — {system.label or system.name}",
        "",
        f"System: **{system.name}** · questions: **{overall['n']}**",
        "",
        "## Protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Top-k retrieved | {protocol['top_k']} |",
        f"| Metadata filter applied | {_fmt(protocol['metadata_filter_applied'])} |",
        f"| Query embedding model | `{protocol['query_embedding_model']}` |",
        f"| Query embedding dim | {protocol['query_embedding_dim']} |",
        f"| Query task type | `{protocol['query_task_type']}` |",
        f"| Retriever | `{protocol['retriever']}` |",
        f"| Vector index | `{protocol['index_name']}` |",
        f"| Context budget (tokens) | {protocol['context_budget_tokens']} |",
        f"| Answers generated | {_fmt(protocol['answers_generated'])} |",
        f"| Answer model | {_fmt(protocol['answer_model'])} |",
        "",
        "## Gold relevance",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Granularity | `{system.gold_granularity}` |",
        f"| Recall metric | **{recall_key}** |",
        f"| Mapping file | `{system.gold_mapping_filename}` |",
        "",
    ]

    if system.proxy_disclaimer:
        lines += [f"> **Caveat.** {system.proxy_disclaimer}", ""]
    else:
        lines += [
            "> This system has source-block provenance, so the recall above is "
            "true instructional-unit recall.",
            "",
        ]

    # Overall table
    header = "| Metric | Value |\n|---|---:|"
    lines += ["## Overall", "", header]
    for key in hit_keys:
        lines.append(f"| {key.upper()} | {_fmt(overall.get(key))} |")
    lines += [
        f"| MRR | {_fmt(overall.get('mrr'))} |",
        f"| {recall_key} | {_fmt(overall.get(recall_key))} |",
        f"| Mean context tokens | {_fmt(overall.get('context_token_count_mean'))} |",
        f"| Total context tokens | {_fmt(overall.get('context_token_count_total'))} |",
        f"| Questions with gold mapping | {_fmt(overall.get('questions_with_gold'))} |",
        f"| Questions without gold mapping | {_fmt(overall.get('questions_without_gold'))} |",
        "",
    ]

    # By question type
    by_type = summary.get("by_question_type") or {}
    if by_type:
        columns = ["Question type", "n", *[k.upper() for k in hit_keys], "MRR", recall_key, "Ctx tokens (mean)"]
        lines += [
            "## By question type",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
        ]
        for question_type, stats in by_type.items():
            row = [
                question_type,
                str(stats.get("n", 0)),
                *[_fmt(stats.get(k)) for k in hit_keys],
                _fmt(stats.get("mrr")),
                _fmt(stats.get(recall_key)),
                _fmt(stats.get("context_token_count_mean")),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════
# CROSS-SYSTEM COMPARISON
# ════════════════════════════════════════════


def build_comparison(summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-system summaries into one comparison object."""
    by_system = {str(s.get("system")): s for s in summaries}
    ordered = [by_system[name] for name in cfg.RETRIEVAL_SYSTEM_ORDER if name in by_system]
    ordered += [s for name, s in by_system.items() if name not in cfg.RETRIEVAL_SYSTEM_ORDER]
    return {
        "systems": [s.get("system") for s in ordered],
        "protocol_equal_across_systems": _protocol_is_equal(ordered),
        "summaries": ordered,
        "notes": {
            "b1_proxy": cfg.B1_PROXY_DISCLAIMER,
            "recall_comparability": (
                "gold_unit_recall (Proposed, B2) and gold_page_recall_proxy (B1) "
                "measure different things and are reported in separate columns. "
                "Hit@1, Hit@5, and MRR are computed identically for all systems "
                "from each system's own gold mapping."
            ),
        },
    }


def _protocol_is_equal(summaries: Sequence[Dict[str, Any]]) -> bool:
    """True when every system ran under the same comparison-critical settings."""
    critical = (
        "top_k",
        "metadata_filter_applied",
        "query_embedding_model",
        "query_embedding_dim",
        "query_task_type",
        "context_budget_tokens",
        "answer_model",
        "answer_prompt_version",
    )
    seen = {
        tuple((summary.get("protocol") or {}).get(key) for key in critical)
        for summary in summaries
    }
    return len(seen) <= 1


def write_comparison(
    layout: schemas.RetrievalOutputLayout,
    summaries: Sequence[Dict[str, Any]],
    config: cfg.RetrievalExperimentConfig,
) -> Dict[str, str]:
    """Write the cross-system comparison JSON + Markdown."""
    layout.ensure()
    comparison = build_comparison(summaries)
    schemas.write_json(layout.comparison_json, comparison)
    layout.comparison_md.write_text(
        comparison_markdown(comparison, config), encoding="utf-8"
    )
    return {
        "comparison_json": str(layout.comparison_json),
        "comparison_md": str(layout.comparison_md),
    }


def comparison_markdown(
    comparison: Dict[str, Any],
    config: cfg.RetrievalExperimentConfig,
) -> str:
    """Render the cross-system Markdown report."""
    summaries: List[Dict[str, Any]] = list(comparison.get("summaries") or [])
    hit_keys = _hit_keys(config)

    lines: List[str] = [
        "# Retrieval Evaluation — System Comparison",
        "",
        "All systems were evaluated on the same frozen QA dataset with the same "
        "query embedding model, the same top-k, no metadata filtering, the same "
        "vector index configuration, and the same answer-generation model, prompt, "
        "and context budget.",
        "",
        f"Protocol identical across systems: **{_fmt(comparison.get('protocol_equal_across_systems'))}**",
        "",
        "## Ranked retrieval",
        "",
    ]

    columns = ["System", "n", *[k.upper() for k in hit_keys], "MRR", "Ctx tokens (mean)"]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for summary in summaries:
        overall = summary.get("overall") or {}
        row = [
            f"{summary.get('system')}",
            str(overall.get("n", 0)),
            *[_fmt(overall.get(k)) for k in hit_keys],
            _fmt(overall.get("mrr")),
            _fmt(overall.get("context_token_count_mean")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Recall is split by granularity so a proxy is never averaged with unit gold.
    lines += [
        "## Gold recall (reported separately by granularity)",
        "",
        "| System | Granularity | Metric | Value |",
        "|---|---|---|---:|",
    ]
    for summary in summaries:
        overall = summary.get("overall") or {}
        gold = summary.get("gold") or {}
        metric = str(gold.get("recall_metric") or "")
        lines.append(
            f"| {summary.get('system')} | `{gold.get('granularity')}` | "
            f"**{metric}** | {_fmt(overall.get(metric))} |"
        )
    lines += [
        "",
        f"> **B1 caveat.** {comparison['notes']['b1_proxy']}",
        "",
        f"> **Comparability.** {comparison['notes']['recall_comparability']}",
        "",
    ]

    # Per-type breakdown, one block per system.
    lines += ["## By question type", ""]
    for summary in summaries:
        gold = summary.get("gold") or {}
        recall_key = str(gold.get("recall_metric") or "")
        by_type = summary.get("by_question_type") or {}
        if not by_type:
            continue
        lines += [
            f"### {summary.get('system_label') or summary.get('system')}",
            "",
        ]
        columns = ["Question type", "n", *[k.upper() for k in hit_keys], "MRR", recall_key]
        lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
        for question_type, stats in by_type.items():
            row = [
                question_type,
                str(stats.get("n", 0)),
                *[_fmt(stats.get(k)) for k in hit_keys],
                _fmt(stats.get("mrr")),
                _fmt(stats.get(recall_key)),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


def log_summary(config: cfg.RetrievalExperimentConfig, summary: Dict[str, Any]) -> None:
    """Log the headline numbers for one system."""
    overall = summary["overall"]
    recall_key = config.system.recall_metric_name
    logger.info("── %s ──", config.system.name)
    for key in _hit_keys(config):
        logger.info("  %-22s %.4f", key, overall.get(key, 0.0))
    logger.info("  %-22s %.4f", "mrr", overall.get("mrr", 0.0))
    logger.info("  %-22s %.4f", recall_key, overall.get(recall_key, 0.0))
    logger.info(
        "  %-22s %.1f", "ctx tokens (mean)", overall.get("context_token_count_mean", 0.0)
    )
    if config.system.proxy_disclaimer:
        logger.warning("  NOTE: %s", config.system.proxy_disclaimer)
