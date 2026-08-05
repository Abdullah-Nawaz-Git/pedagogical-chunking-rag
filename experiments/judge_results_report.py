"""Create tables and figures from the fixed LLM-as-judge evaluation artifact.

This is a post-processing step: it does not call a judge model or regenerate
any scores.  It reads the single summary artifact already written by the
judge-evaluation pipeline under ``judge_eval/`` and creates a compact analysis
bundle that is suitable for inspection and inclusion in a report.

Run from the repository root:

    python -m experiments.judge_results_report

The default writes to ``judge_eval/analysis/``.  The input filename is fixed:

* ``judge_eval/judge_summary.json``

Unlike the retrieval report, this report includes all three systems —
Proposed, B2, and B1 — because the judge dimensions (context recall/
precision, faithfulness, answer relevancy) are scored the same way
regardless of how each system's gold context was mapped. The mapping
caveat that applies to B1's retrieval-side recall still applies to
anything here that is context-dependent (``context_recall`` and
``context_precision``); see the B1 proxy note carried through from
``notes.b1_proxy`` in the source file and reproduced in the report index.

These are secondary, model-dependent scores. The retrieval-native metrics
under ``retrieval_eval/`` remain the primary evidence.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


# Match the order used elsewhere in the project without making this
# reporting script depend on the retrieval/judge packages at import time.
SYSTEM_ORDER: tuple[str, ...] = ("proposed", "b2", "b1")
SYSTEM_SHORT_LABELS: Dict[str, str] = {
    "proposed": "Proposed",
    "b2": "B2",
    "b1": "B1",
}
SYSTEM_COLOURS: Dict[str, str] = {
    "proposed": "#0072B2",  # colour-blind-friendly blue
    "b2": "#009E73",        # green
    "b1": "#D55E00",        # vermilion
}
DIMENSION_LABELS: Dict[str, str] = {
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the analysis-only reporting step."""
    parser = argparse.ArgumentParser(
        description=(
            "Write tables and figures from judge_eval's fixed Proposed, B2, "
            "and B1 LLM-as-judge summary."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("judge_eval"),
        help="Directory containing judge_summary.json (default: judge_eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated tables and figures (default: INPUT_DIR/analysis).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Resolution for PNG figures (default: 220).",
    )
    return parser


# ════════════════════════════════════════════
# INPUT LOADING AND VALIDATION
# ════════════════════════════════════════════


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _as_float(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric {field}; received {value!r}") from exc


def _question_type_order(summaries: Mapping[str, Dict[str, Any]]) -> List[str]:
    """Use the first system's question-type order, retaining any extras."""
    first = summaries[SYSTEM_ORDER[0]]
    ordered = list((first.get("by_question_type") or {}).keys())
    for system in SYSTEM_ORDER[1:]:
        for question_type in (summaries[system].get("by_question_type") or {}).keys():
            if question_type not in ordered:
                ordered.append(question_type)
    if not ordered:
        raise ValueError("No question types found in judge summary")
    return ordered


def load_artifacts(input_dir: Path) -> Dict[str, Any]:
    """Load the fixed judge summary and fail early if systems can't be compared safely."""
    document = _read_json(input_dir / "judge_summary.json")

    file_systems = set(str(name) for name in document.get("systems") or [])
    if file_systems != set(SYSTEM_ORDER):
        raise ValueError(
            "judge_summary.json must contain exactly Proposed, B2, and B1; "
            f"found {sorted(file_systems)!r}"
        )

    dimensions = [str(dim) for dim in document.get("dimensions") or []]
    if not dimensions:
        raise ValueError("judge_summary.json is missing a non-empty 'dimensions' list")

    summaries: Dict[str, Dict[str, Any]] = {}
    for entry in document.get("summaries") or []:
        system = str(entry.get("system") or "")
        if system in SYSTEM_ORDER:
            summaries[system] = entry
    missing = [system for system in SYSTEM_ORDER if system not in summaries]
    if missing:
        raise ValueError(f"judge_summary.json is missing summaries for: {missing!r}")

    budgets = {
        int(((summaries[system].get("judge_protocol") or {}).get("context_budget_tokens")) or 0)
        for system in SYSTEM_ORDER
    }
    if len(budgets) != 1 or next(iter(budgets)) < 1:
        raise ValueError(
            f"The three judge runs must share one positive context budget; found {sorted(budgets)!r}"
        )

    return {
        "dimensions": dimensions,
        "summaries": summaries,
        "question_types": _question_type_order(summaries),
        "paired_comparisons": document.get("paired_comparisons") or [],
        "protocol_equal": bool(document.get("judge_protocol_equal_across_systems")),
        "context_budget_tokens": next(iter(budgets)),
        "notes": document.get("notes") or {},
    }


# ════════════════════════════════════════════
# TABLE DATA
# ════════════════════════════════════════════


def _system_label(summary: Mapping[str, Any]) -> str:
    system = str(summary.get("system") or "")
    return str(summary.get("system_label") or SYSTEM_SHORT_LABELS.get(system, system))


def _dim_stats(block: Mapping[str, Any] | None, dimension: str) -> Dict[str, Any]:
    stats = (block or {}).get(dimension) if block else None
    stats = stats or {}
    attempted = int(stats.get("n_attempted") or 0)
    errored = int(stats.get("n_errored") or 0)
    return {
        "mean": _as_float(stats.get("mean"), field=f"{dimension} mean"),
        "n_scored": int(stats.get("n_scored") or 0),
        "n_errored": errored,
        "n_skipped": int(stats.get("n_skipped") or 0),
        "n_attempted": attempted,
        "error_rate": (errored / attempted) if attempted else 0.0,
    }


def _overall_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one raw, CSV-friendly aggregate row for each system."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        overall = summary.get("overall") or {}
        by_dimension = overall.get("by_dimension") or {}
        context = summary.get("context_statistics") or {}
        row: Dict[str, Any] = {
            "system": system,
            "system_label": _system_label(summary),
            "n_items": int(overall.get("n_items") or 0),
            "mean_context_tokens": _as_float(
                context.get("mean_context_tokens"), field=f"{system} mean context tokens"
            ),
            "context_budget_tokens": data["context_budget_tokens"],
            "truncation_rate": _as_float(
                context.get("truncation_rate"), field=f"{system} truncation rate"
            ),
            "mean_retries_used": _as_float(
                context.get("mean_retries_used"), field=f"{system} mean retries used"
            ),
        }
        for dimension in data["dimensions"]:
            stats = _dim_stats(by_dimension, dimension)
            row[f"{dimension}_mean"] = stats["mean"]
            row[f"{dimension}_n_scored"] = stats["n_scored"]
            row[f"{dimension}_n_errored"] = stats["n_errored"]
            row[f"{dimension}_error_rate"] = stats["error_rate"]
        rows.append(row)
    return rows


def _by_question_type_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one row per system and question type."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        by_type = summary.get("by_question_type") or {}
        for question_type in data["question_types"]:
            stats_block = by_type.get(question_type)
            if not isinstance(stats_block, dict):
                raise ValueError(f"{system} judge summary is missing question type {question_type!r}")
            by_dimension = stats_block.get("by_dimension") or {}
            row: Dict[str, Any] = {
                "system": system,
                "system_label": _system_label(summary),
                "question_type": question_type,
                "n_items": int(stats_block.get("n_items") or 0),
            }
            for dimension in data["dimensions"]:
                stats = _dim_stats(by_dimension, dimension)
                row[f"{dimension}_mean"] = stats["mean"]
                row[f"{dimension}_error_rate"] = stats["error_rate"]
            rows.append(row)
    return rows


def _error_rate_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one row per system and dimension describing judge coverage."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        by_dimension = (summary.get("overall") or {}).get("by_dimension") or {}
        for dimension in data["dimensions"]:
            stats = _dim_stats(by_dimension, dimension)
            rows.append(
                {
                    "system": system,
                    "system_label": _system_label(summary),
                    "dimension": dimension,
                    "n_scored": stats["n_scored"],
                    "n_errored": stats["n_errored"],
                    "n_skipped": stats["n_skipped"],
                    "n_attempted": stats["n_attempted"],
                    "error_rate": stats["error_rate"],
                }
            )
    return rows


def _context_statistics_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        context = summary.get("context_statistics") or {}
        rows.append(
            {
                "system": system,
                "system_label": _system_label(summary),
                "n_items_with_context": int(context.get("n_items_with_context") or 0),
                "mean_context_tokens": _as_float(
                    context.get("mean_context_tokens"), field=f"{system} mean context tokens"
                ),
                "max_context_tokens": _as_float(
                    context.get("max_context_tokens"), field=f"{system} max context tokens"
                ),
                "n_truncated": int(context.get("n_truncated") or 0),
                "truncation_rate": _as_float(
                    context.get("truncation_rate"), field=f"{system} truncation rate"
                ),
                "mean_retries_used": _as_float(
                    context.get("mean_retries_used"), field=f"{system} mean retries used"
                ),
                "retrying_records": int(context.get("retrying_records") or 0),
            }
        )
    return rows


def _paired_comparison_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the paired bootstrap comparisons into one row per dimension."""
    rows: List[Dict[str, Any]] = []
    for comparison in data["paired_comparisons"]:
        treatment = str(comparison.get("treatment") or "")
        control = str(comparison.get("control") or "")
        by_dimension = comparison.get("by_dimension") or {}
        for dimension in data["dimensions"]:
            stats = by_dimension.get(dimension)
            if not isinstance(stats, dict):
                continue
            ci = stats.get("bootstrap_ci") or {}
            rows.append(
                {
                    "treatment": treatment,
                    "control": control,
                    "dimension": dimension,
                    "mean_difference": _as_float(
                        stats.get("mean_difference"), field=f"{treatment} vs {control} {dimension} delta"
                    ),
                    "ci_lower": _as_float(ci.get("lower"), field=f"{treatment} vs {control} {dimension} ci lower"),
                    "ci_upper": _as_float(ci.get("upper"), field=f"{treatment} vs {control} {dimension} ci upper"),
                    "confidence": _as_float(ci.get("confidence"), field=f"{treatment} vs {control} {dimension} confidence"),
                    "n_paired": int(stats.get("n_paired") or 0),
                }
            )
    return rows


# ════════════════════════════════════════════
# FILE WRITERS AND MARKDOWN
# ════════════════════════════════════════════


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write an Excel-friendly UTF-8 CSV, using every key present in the rows."""
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns: List[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


Formatter = Callable[[Any], str]


def _text(value: Any) -> str:
    return "—" if value is None or value == "" else str(value)


def _decimal(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _integer(value: Any) -> str:
    return "—" if value is None else f"{int(round(float(value))):,}"


def _signed_percent(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):+.1f}%"


def _markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str, Formatter]],
) -> str:
    """Render a compact GitHub-flavoured Markdown table."""
    lines = [
        "| " + " | ".join(label for _, label, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        cells = []
        for key, _, formatter in columns:
            value = formatter(row.get(key))
            cells.append(value.replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _display_dimension(dimension: str) -> str:
    return DIMENSION_LABELS.get(dimension, dimension.replace("_", " ").title())


def write_tables(data: Mapping[str, Any], output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Write CSV tables and the standalone Markdown versions of key tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = _overall_rows(data)
    by_type = _by_question_type_rows(data)
    error_rates = _error_rate_rows(data)
    context_stats = _context_statistics_rows(data)
    paired = _paired_comparison_rows(data)

    tables = {
        "overall_judge_scores": overall,
        "by_question_type": by_type,
        "error_rates_by_dimension": error_rates,
        "context_statistics": context_stats,
        "paired_comparisons": paired,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)

    overall_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("n_items", "n", _integer),
    ]
    overall_columns += [
        (f"{dimension}_mean", _display_dimension(dimension), _percent)
        for dimension in data["dimensions"]
    ]
    overall_columns += [
        ("mean_context_tokens", "Mean context tokens", _integer),
        ("truncation_rate", "Truncation rate", _percent),
    ]
    (output_dir / "overall_judge_scores.md").write_text(
        "# Overall LLM-as-judge scores\n\n" + _markdown_table(overall, overall_columns) + "\n",
        encoding="utf-8",
    )

    type_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("question_type", "Question type", _text),
        ("n_items", "n", _integer),
    ]
    type_columns += [
        (f"{dimension}_mean", _display_dimension(dimension), _percent)
        for dimension in data["dimensions"]
    ]
    (output_dir / "by_question_type.md").write_text(
        "# Judge scores by question type\n\n" + _markdown_table(by_type, type_columns) + "\n",
        encoding="utf-8",
    )

    error_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("dimension", "Dimension", lambda value: _display_dimension(str(value))),
        ("n_attempted", "Attempted", _integer),
        ("n_scored", "Scored", _integer),
        ("n_errored", "Errored", _integer),
        ("error_rate", "Error rate", _percent),
    ]
    (output_dir / "error_rates_by_dimension.md").write_text(
        "# Judge coverage and error rates by dimension\n\n"
        + _markdown_table(error_rates, error_columns)
        + "\n\n"
        + "> An errored item means the judge call failed or its output could not be parsed for that "
        "dimension; its score is excluded from the corresponding mean rather than counted as zero.\n",
        encoding="utf-8",
    )

    paired_columns: List[tuple[str, str, Formatter]] = [
        ("treatment", "Treatment", _text),
        ("control", "Control", _text),
        ("dimension", "Dimension", lambda value: _display_dimension(str(value))),
        ("mean_difference", "Mean difference", _signed_percent),
        ("ci_lower", "95% CI lower", _signed_percent),
        ("ci_upper", "95% CI upper", _signed_percent),
        ("n_paired", "n paired", _integer),
    ]
    (output_dir / "paired_comparisons.md").write_text(
        "# Paired judge-score comparisons\n\n" + _markdown_table(paired, paired_columns) + "\n",
        encoding="utf-8",
    )

    return tables


# ════════════════════════════════════════════
# FIGURES
# ════════════════════════════════════════════


def _pyplot():
    """Load Matplotlib in non-interactive mode without requiring a writable home."""
    cache_dir = Path(tempfile.gettempdir()) / "pedagogical_chunking_rag_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    # Matplotlib delegates font discovery to fontconfig on many systems.  Point
    # its cache at the same writable temporary location so report generation is
    # quiet in a sandboxed or CI environment with a read-only home directory.
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    return plt


def _save_figure(figure: Any, figures_dir: Path, stem: str, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(figures_dir / f"{stem}.svg", bbox_inches="tight")


def _annotate_bars(axis: Any, bars: Iterable[Any], *, decimals: int = 2) -> None:
    for bar in bars:
        height = float(bar.get_height())
        axis.annotate(
            f"{height:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _plot_overall_judge_scores(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    dimensions = data["dimensions"]
    labels = [_display_dimension(dimension) for dimension in dimensions]
    x_values = list(range(len(dimensions)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.7, 4.8))
    for index, system in enumerate(SYSTEM_ORDER):
        by_dimension = (data["summaries"][system].get("overall") or {}).get("by_dimension") or {}
        offset = (index - (len(SYSTEM_ORDER) - 1) / 2) * width
        values = [_dim_stats(by_dimension, dimension)["mean"] or 0.0 for dimension in dimensions]
        bars = axis.bar(
            [x + offset for x in x_values],
            values,
            width,
            label=SYSTEM_SHORT_LABELS[system],
            color=SYSTEM_COLOURS[system],
        )
        _annotate_bars(axis, bars)
    axis.set_title("Overall LLM-as-judge scores")
    axis.set_ylabel("Mean score")
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures_dir, "01_overall_judge_scores", dpi)
    plt.close(figure)


def _plot_question_type_scores(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    dimensions = data["dimensions"]
    labels = [_display_dimension(dimension) for dimension in dimensions]
    question_types = data["question_types"]
    x_values = list(range(len(question_types)))
    width = 0.24
    figure, axes = plt.subplots(1, len(dimensions), figsize=(5.1 * len(dimensions), 4.9), sharey=True)
    if len(dimensions) == 1:
        axes = [axes]
    for axis, dimension, label in zip(axes, dimensions, labels):
        for index, system in enumerate(SYSTEM_ORDER):
            by_type = data["summaries"][system]["by_question_type"]
            offset = (index - (len(SYSTEM_ORDER) - 1) / 2) * width
            values = [
                _dim_stats(by_type[question_type].get("by_dimension"), dimension)["mean"] or 0.0
                for question_type in question_types
            ]
            bars = axis.bar(
                [x + offset for x in x_values],
                values,
                width,
                color=SYSTEM_COLOURS[system],
                label=SYSTEM_SHORT_LABELS[system],
            )
            _annotate_bars(axis, bars)
        axis.set_title(label)
        axis.set_xticks(x_values, [question_type.replace("_", "\n") for question_type in question_types])
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(0, 1.11)
    axes[0].set_ylabel("Mean score")
    axes[-1].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    figure.suptitle("LLM-as-judge scores by question type", y=1.02, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, figures_dir, "02_judge_scores_by_question_type", dpi)
    plt.close(figure)


def _plot_context_tokens(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    rows = _overall_rows(data)
    figure, axis = plt.subplots(figsize=(6.7, 4.7))
    bars = axis.bar(
        [SYSTEM_SHORT_LABELS[row["system"]] for row in rows],
        [row["mean_context_tokens"] for row in rows],
        color=[SYSTEM_COLOURS[row["system"]] for row in rows],
        width=0.56,
    )
    _annotate_bars(axis, bars, decimals=0)
    budget = rows[0]["context_budget_tokens"]
    axis.axhline(budget, color="#4D4D4D", linestyle="--", linewidth=1.2, label=f"Shared budget ({budget:,})")
    axis.set_title("Mean judge context tokens per item")
    axis.set_ylabel("Whitespace tokens")
    axis.set_ylim(0, max(budget * 1.12, max(row["mean_context_tokens"] for row in rows) * 1.12))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    _save_figure(figure, figures_dir, "03_mean_context_tokens", dpi)
    plt.close(figure)


def _plot_error_rates(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    dimensions = data["dimensions"]
    labels = [_display_dimension(dimension) for dimension in dimensions]
    x_values = list(range(len(dimensions)))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.7, 4.6))
    for index, system in enumerate(SYSTEM_ORDER):
        by_dimension = (data["summaries"][system].get("overall") or {}).get("by_dimension") or {}
        offset = (index - (len(SYSTEM_ORDER) - 1) / 2) * width
        values = [_dim_stats(by_dimension, dimension)["error_rate"] for dimension in dimensions]
        bars = axis.bar(
            [x + offset for x in x_values],
            values,
            width,
            label=SYSTEM_SHORT_LABELS[system],
            color=SYSTEM_COLOURS[system],
        )
        _annotate_bars(axis, bars)
    axis.set_title("Judge call / parse error rate by dimension")
    axis.set_ylabel("Error rate")
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0, 1.0)
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures_dir, "04_error_rates_by_dimension", dpi)
    plt.close(figure)


def _plot_paired_comparison(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    """Plot the bootstrap mean-difference with CI for each paired comparison."""
    plt = _pyplot()
    for comparison in data["paired_comparisons"]:
        treatment = str(comparison.get("treatment") or "")
        control = str(comparison.get("control") or "")
        by_dimension = comparison.get("by_dimension") or {}
        dimensions = [dim for dim in data["dimensions"] if dim in by_dimension]
        if not dimensions:
            continue
        labels = [_display_dimension(dimension) for dimension in dimensions]
        diffs = [float(by_dimension[dimension]["mean_difference"]) for dimension in dimensions]
        lowers = [float(by_dimension[dimension]["bootstrap_ci"]["lower"]) for dimension in dimensions]
        uppers = [float(by_dimension[dimension]["bootstrap_ci"]["upper"]) for dimension in dimensions]
        errors = [[d - lo for d, lo in zip(diffs, lowers)], [up - d for d, up in zip(diffs, uppers)]]

        y_values = list(range(len(dimensions)))
        figure, axis = plt.subplots(figsize=(7.2, 1.1 + 0.7 * len(dimensions)))
        colours = ["#009E73" if diff >= 0 else "#D55E00" for diff in diffs]
        axis.barh(y_values, diffs, xerr=errors, color=colours, height=0.5, capsize=4, ecolor="#333333")
        axis.axvline(0, color="#4D4D4D", linewidth=1.0)
        axis.set_yticks(y_values, labels)
        axis.set_xlabel(f"Mean difference ({SYSTEM_SHORT_LABELS.get(treatment, treatment)} − "
                         f"{SYSTEM_SHORT_LABELS.get(control, control)}), with 95% bootstrap CI")
        axis.set_title(
            f"{SYSTEM_SHORT_LABELS.get(treatment, treatment)} vs "
            f"{SYSTEM_SHORT_LABELS.get(control, control)}: judge score deltas"
        )
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        _save_figure(figure, figures_dir, f"05_paired_{treatment}_vs_{control}", dpi)
        plt.close(figure)


def write_figures(data: Mapping[str, Any], output_dir: Path, dpi: int) -> None:
    """Write publication-ready PNG and SVG figures under ``output_dir/figures``."""
    figures_dir = output_dir / "figures"
    _plot_overall_judge_scores(data, figures_dir, dpi)
    _plot_question_type_scores(data, figures_dir, dpi)
    _plot_context_tokens(data, figures_dir, dpi)
    _plot_error_rates(data, figures_dir, dpi)
    _plot_paired_comparison(data, figures_dir, dpi)


# ════════════════════════════════════════════
# ANALYSIS INDEX
# ════════════════════════════════════════════


def write_report_index(data: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path) -> None:
    """Write the report landing page that links all generated artifacts."""
    overall = tables["overall_judge_scores"]
    notes = data["notes"]
    protocol_equal = data["protocol_equal"]

    overview_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("n_items", "n", _integer),
    ]
    overview_columns += [
        (f"{dimension}_mean", _display_dimension(dimension), _percent)
        for dimension in data["dimensions"]
    ]
    overview_columns += [
        ("mean_context_tokens", "Mean context tokens", _integer),
        ("truncation_rate", "Truncation rate", _percent),
    ]

    evidence_status = str(notes.get("evidence_status") or "")
    primary_comparison = str(notes.get("primary_comparison") or "")
    b1_proxy = str(notes.get("b1_proxy") or "")
    judge_gold_isolation = str(notes.get("judge_gold_isolation") or "")

    lines = [
        "# LLM-as-Judge Results Analysis",
        "",
        "This bundle was generated from the fixed judge-evaluation summary in `judge_eval/`. "
        "It is a reporting-only step; it does not call a judge model or rerun retrieval or generation.",
        "",
        "## Protocol check",
        "",
        f"- Items per system: **{overall[0]['n_items']}**",
        f"- Shared context budget: **{data['context_budget_tokens']:,} tokens**",
        f"- Shared comparison-critical judge protocol: **{'yes' if protocol_equal else 'no'}**",
        "",
        f"> {evidence_status}" if evidence_status else "",
        "",
        "## Overall judge scores",
        "",
        _markdown_table(overall, overview_columns),
        "",
        "![Overall judge scores](figures/01_overall_judge_scores.png)",
        "",
        f"> **Primary comparison.** {primary_comparison}" if primary_comparison else "",
        "",
        f"> **B1 caveat.** {b1_proxy}" if b1_proxy else "",
        "",
        f"> **Gold isolation.** {judge_gold_isolation}" if judge_gold_isolation else "",
        "",
        "## By question type",
        "",
        "See [by_question_type.md](by_question_type.md) and "
        "[by_question_type.csv](by_question_type.csv) for the complete table.",
        "",
        "![Judge scores by question type](figures/02_judge_scores_by_question_type.png)",
        "",
        "## Context tokens and judge coverage",
        "",
        "![Mean context tokens](figures/03_mean_context_tokens.png)",
        "",
        "![Error rates by dimension](figures/04_error_rates_by_dimension.png)",
        "",
        "See [error_rates_by_dimension.md](error_rates_by_dimension.md) for the exact counts "
        "behind each error rate.",
        "",
        "## Paired comparisons",
        "",
        "See [paired_comparisons.md](paired_comparisons.md) for bootstrap mean differences and "
        "95% confidence intervals; per-comparison charts are under `figures/05_paired_*.png`.",
        "",
        "## Output files",
        "",
        "| Artifact | Purpose |",
        "|---|---|",
        "| [overall_judge_scores.csv](overall_judge_scores.csv) / [overall_judge_scores.md](overall_judge_scores.md) | Headline judge scores and error rates per system. |",
        "| [by_question_type.csv](by_question_type.csv) / [by_question_type.md](by_question_type.md) | Per-question-type judge scores. |",
        "| [error_rates_by_dimension.csv](error_rates_by_dimension.csv) / [error_rates_by_dimension.md](error_rates_by_dimension.md) | Judge call/parse coverage and error counts per dimension. |",
        "| [context_statistics.csv](context_statistics.csv) | Context-token statistics fed to the judge per system. |",
        "| [paired_comparisons.csv](paired_comparisons.csv) / [paired_comparisons.md](paired_comparisons.md) | Bootstrap mean differences with 95% CIs. |",
        "| [figures/](figures/) | Every chart in both PNG and SVG. |",
        "",
    ]
    # Drop stray blank lines left by conditional caveat lines with no content.
    cleaned_lines = [line for line in lines if line != ""] 
    # Re-insert blank-line spacing: rebuild with a single blank line between
    # non-empty blocks, preserving explicit "" entries used for spacing above.
    final_lines: List[str] = []
    for line in lines:
        if line == "" and final_lines and final_lines[-1] == "":
            continue
        final_lines.append(line)
    (output_dir / "judge_results_report.md").write_text("\n".join(final_lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dpi < 72:
        print("--dpi must be at least 72", file=sys.stderr)
        return 2

    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir / "analysis"
    try:
        data = load_artifacts(input_dir)
        tables = write_tables(data, output_dir)
        write_figures(data, output_dir, args.dpi)
        write_report_index(data, tables, output_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"judge results report: {exc}", file=sys.stderr)
        return 2

    print(f"Done. Judge tables and figures written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
