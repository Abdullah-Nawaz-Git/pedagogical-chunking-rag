"""
ragkit.judge.report
════════════════════

Writes the LLM-as-judge artifacts: per-question judge detail, a compact summary
JSON, and a Markdown report — for a single system and across all three.

Two disclaimers are printed on EVERY report this module emits and never dropped:

    1. These are *LLM-as-judge scores inspired by RAGAS dimensions*, secondary and
       model-dependent. The retrieval-native metrics in ``retrieval_eval/`` remain
       the primary evidence.
    2. B1's gold mapping is a page-overlap proxy. Even though gold is NEVER shown to
       the judge, the caveat travels with B1 everywhere so a reader interpreting the
       row alongside the retrieval results is never misled.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.judge.report")

_PRECISION = 4


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{_PRECISION}f}"
    if value is None or value == "":
        return "—"
    return str(value)


def _dim_display(metric: str) -> str:
    spec = cfg.JUDGE_DIMENSION_SPECS.get(metric)
    return spec.display if spec else metric


def system_markdown(
    config: cfg.JudgeExperimentConfig,
    summary: Dict[str, Any],
    dimensions: Sequence[str],
) -> str:
    """Render the per-system judge Markdown report."""
    system = config.system
    overall = summary["overall"]
    protocol = summary["judge_protocol"]
    by_dimension = overall.get("by_dimension") or {}

    lines: List[str] = [
        f"# LLM-as-Judge Evaluation — {system.label}",
        "",
        f"System: **{system.name}** · role: _{system.experimental_role}_ · "
        f"items scored: **{overall.get('n_items', 0)}**",
        "",
        "> **Secondary evidence.** These are LLM-as-judge scores *inspired by* "
        "RAGAS dimensions. They corroborate but do not replace the retrieval-native "
        "metrics (Hit@k, MRR, Gold-Unit/Page Recall) in `retrieval_eval/`, which "
        "remain the primary evidence.",
        "",
        "## Judge protocol",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Provider | `{protocol['provider']}` |",
        f"| Judge model | `{protocol['model']}` |",
        f"| Temperature | {_fmt(protocol['temperature'])} |",
        f"| Max output tokens | {protocol['max_output_tokens']} |",
        f"| Prompt version | `{protocol['prompt_version']}` |",
        f"| Mode | `{protocol['mode']}` |",
        f"| Context budget (tokens) | {protocol['context_budget_tokens']} |",
        f"| Dimensions | {', '.join(_dim_display(d) for d in dimensions)} |",
        "",
        "> The judge never sees gold ids, ranks, Hit@k, or mapping status; those "
        "remain for deterministic evaluation only.",
        "",
    ]

    # Overall scores table.
    lines += [
        "## Overall judge scores",
        "",
        "| Dimension | Mean score | Scored | Skipped | Failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in dimensions:
        stats = by_dimension.get(metric) or {}
        lines.append(
            f"| {_dim_display(metric)} | {_fmt(stats.get('mean'))} | "
            f"{_fmt(stats.get('n_scored'))} | {_fmt(stats.get('n_skipped'))} | "
            f"{_fmt(stats.get('n_errored'))} |"
        )
    lines.append("")

    # B1 proxy caveat travels with the system even though gold is never judged.
    if system.mapping_caveat:
        lines += [
            f"> **B1 mapping caveat (context only).** {system.mapping_caveat} This "
            "caveat is unrelated to the judge inputs (the judge saw no gold) but is "
            "repeated so this row is read consistently with the retrieval results.",
            "",
        ]

    # By question type.
    by_type = summary.get("by_question_type") or {}
    if by_type:
        columns = ["Question type", "n", *[_dim_display(d) for d in dimensions]]
        lines += [
            "## By question type",
            "",
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
        ]
        for question_type, stats in by_type.items():
            dims = stats.get("by_dimension") or {}
            row = [
                question_type,
                str(stats.get("n_items", 0)),
                *[_fmt((dims.get(d) or {}).get("mean")) for d in dimensions],
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    return "\n".join(lines)


# ════════════════════════════════════════════
# CROSS-SYSTEM COMPARISON
# ════════════════════════════════════════════


def build_comparison(
    summaries: Sequence[Dict[str, Any]],
    comparisons: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Dict[str, Any]:
    """Combine per-system summaries + paired comparisons into one object."""
    by_system = {str(s.get("system")): s for s in summaries}
    ordered = [by_system[name] for name in cfg.JUDGE_SYSTEM_ORDER if name in by_system]
    ordered += [s for name, s in by_system.items() if name not in cfg.JUDGE_SYSTEM_ORDER]
    return {
        "systems": [s.get("system") for s in ordered],
        "dimensions": list(dimensions),
        "judge_protocol_equal_across_systems": _protocol_is_equal(ordered),
        "summaries": ordered,
        "paired_comparisons": list(comparisons),
        "notes": {
            "evidence_status": (
                "LLM-as-judge scores inspired by RAGAS dimensions. Secondary and "
                "model-dependent; the retrieval-native metrics in retrieval_eval/ "
                "are the primary evidence."
            ),
            "primary_comparison": (
                "The key comparison isolates chunk boundaries: proposed "
                "(pedagogical chunks) vs b2 (fixed 512-token windows), holding "
                "structured VLM extraction and representation constant."
            ),
            "b1_proxy": cfg.B1_PROXY_DISCLAIMER,
            "judge_gold_isolation": (
                "No gold-derived signal (gold ids, Hit@k, MRR, mapping status) was "
                "ever shown to the judge for any system."
            ),
        },
    }


def _protocol_is_equal(summaries: Sequence[Dict[str, Any]]) -> bool:
    """True when the judge ran identically across systems."""
    critical = (
        "provider",
        "model",
        "temperature",
        "max_output_tokens",
        "prompt_version",
        "mode",
        "context_budget_tokens",
    )
    seen = {
        tuple((summary.get("judge_protocol") or {}).get(key) for key in critical)
        for summary in summaries
    }
    return len(seen) <= 1


def write_comparison(
    layout: schemas.JudgeOutputLayout,
    summaries: Sequence[Dict[str, Any]],
    comparisons: Sequence[Dict[str, Any]],
    dimensions: Sequence[str],
) -> Dict[str, str]:
    """Write the combined Markdown report.

    The runner owns the matching ``judge_summary.json`` write so all five public
    artifacts use one canonical output layout rather than a stale per-system API.
    """
    layout.ensure()
    comparison = build_comparison(summaries, comparisons, dimensions)
    layout.summary_md.write_text(
        comparison_markdown(comparison, dimensions), encoding="utf-8"
    )
    return {
        "summary_md": str(layout.summary_md),
    }


def comparison_markdown(
    comparison: Dict[str, Any],
    dimensions: Sequence[str],
) -> str:
    """Render the cross-system judge Markdown report."""
    summaries: List[Dict[str, Any]] = list(comparison.get("summaries") or [])

    lines: List[str] = [
        "# LLM-as-Judge Evaluation — System Comparison",
        "",
        "> **Secondary evidence.** " + comparison["notes"]["evidence_status"],
        "",
        "All systems were scored by the same judge model, at the same temperature, "
        "with the same prompts, the same context budget, and the same dimension "
        "definitions. " + comparison["notes"]["judge_gold_isolation"],
        "",
        f"Judge protocol identical across systems: "
        f"**{_fmt(comparison.get('judge_protocol_equal_across_systems'))}**",
        "",
        "## Overall scores by dimension",
        "",
    ]

    columns = ["System", "Role", "n", *[_dim_display(d) for d in dimensions]]
    lines += ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for summary in summaries:
        overall = summary.get("overall") or {}
        by_dimension = overall.get("by_dimension") or {}
        row = [
            f"{summary.get('system')}",
            f"{summary.get('experimental_role') or '—'}",
            str(overall.get("n_items", 0)),
            *[_fmt((by_dimension.get(d) or {}).get("mean")) for d in dimensions],
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines += [
        "## Context and run coverage",
        "",
        "| System | Context items | Mean context tokens | Truncated | Retried records | Status counts |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        context = summary.get("context_statistics") or {}
        status_groups = summary.get("by_judge_status") or {}
        status_counts = ", ".join(
            f"{status}: {stats.get('n_records', 0)}"
            for status, stats in status_groups.items()
        ) or "—"
        lines.append(
            f"| {summary.get('system')} | {_fmt(context.get('n_items_with_context'))} | "
            f"{_fmt(context.get('mean_context_tokens'))} | "
            f"{_fmt(context.get('n_truncated'))} | {_fmt(context.get('retrying_records'))} | "
            f"{status_counts} |"
        )
    lines.append("")

    # Paired comparisons (treatment - control), one block each.
    paired = comparison.get("paired_comparisons") or []
    if paired:
        lines += [
            "## Paired comparison (treatment minus control)",
            "",
            "> " + comparison["notes"]["primary_comparison"],
            "",
        ]
        for pair in paired:
            lines += [
                f"### {pair.get('treatment')} − {pair.get('control')}",
                "",
                "| Dimension | Mean Δ | n paired | Bootstrap 95% CI |",
                "|---|---:|---:|---|",
            ]
            by_dimension = pair.get("by_dimension") or {}
            for metric in dimensions:
                stats = by_dimension.get(metric) or {}
                ci = stats.get("bootstrap_ci")
                ci_text = (
                    f"[{_fmt(ci['lower'])}, {_fmt(ci['upper'])}]" if ci else "—"
                )
                lines.append(
                    f"| {_dim_display(metric)} | {_fmt(stats.get('mean_difference'))} "
                    f"| {_fmt(stats.get('n_paired'))} | {ci_text} |"
                )
            lines.append("")

    # Global caveats.
    lines += [
        "## Caveats",
        "",
        f"- **B1 mapping proxy.** {comparison['notes']['b1_proxy']}",
        f"- **Judge isolation.** {comparison['notes']['judge_gold_isolation']}",
        "- **Interpretation.** Judge scores are corroborating evidence. Where they "
        "agree with the retrieval-native metrics they strengthen the finding; where "
        "they disagree, the deterministic retrieval metrics take precedence.",
        "",
    ]

    return "\n".join(lines)


def log_summary(
    config: cfg.JudgeExperimentConfig,
    summary: Dict[str, Any],
    dimensions: Sequence[str],
) -> None:
    """Log the headline judge scores for one system."""
    overall = summary["overall"]
    by_dimension = overall.get("by_dimension") or {}
    logger.info("── %s (judge) ──", config.system.name)
    for metric in dimensions:
        stats = by_dimension.get(metric) or {}
        mean = stats.get("mean")
        logger.info(
            "  %-20s %s  (scored %s, errored %s)",
            metric,
            f"{mean:.4f}" if isinstance(mean, float) else "—",
            stats.get("n_scored", 0),
            stats.get("n_errored", 0),
        )
    if config.system.mapping_caveat:
        logger.warning("  NOTE (context only): %s", config.system.mapping_caveat)
