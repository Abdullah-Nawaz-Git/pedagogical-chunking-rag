"""Create tables and figures from the fixed retrieval-evaluation artifacts.

This is a post-processing step: it does not query an index or regenerate any
retrieval results.  It reads the three system artifacts already written by the
retrieval experiments under ``retrieval_eval/`` and creates a compact analysis
bundle that is suitable for inspection and inclusion in a report.

Run from the repository root:

    python -m experiments.retrieval_results_report

The default writes to ``retrieval_eval/analysis/``.  The input filenames are
deliberately fixed to match :class:`ragkit.retrieval.schemas.RetrievalOutputLayout`:

* ``config_used_{proposed,b2,b1}.json``
* ``retrieval_summary_{proposed,b2,b1}.json``
* ``retrieval_records_{proposed,b2,b1}.jsonl``
* ``retrieval_comparison.json``

Gold Unit Recall is charted only for Proposed and B2.  B1 has no source-block
provenance, so its page-overlap metric is labelled ``Gold Page Recall (proxy)``
and is never averaged with or plotted as unit-level recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


# Match the order used by ragkit.config.RETRIEVAL_SYSTEM_ORDER without making
# this reporting script depend on the retrieval package at import time.
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the analysis-only reporting step."""
    parser = argparse.ArgumentParser(
        description=(
            "Write tables and figures from retrieval_eval's fixed B1, B2, and "
            "Proposed retrieval artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("retrieval_eval"),
        help="Directory containing the fixed retrieval artifacts (default: retrieval_eval).",
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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _as_float(value: Any, *, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric {field}; received {value!r}") from exc


def _hit_keys(configs: Mapping[str, Dict[str, Any]]) -> List[str]:
    """Return reported Hit@k keys in numeric rank order."""
    keys = {
        f"hit@{int(k)}"
        for config in configs.values()
        for k in (config.get("retrieval") or {}).get("hit_at_ks", [])
    }
    if not keys:
        raise ValueError("No retrieval.hit_at_ks values found in config_used files")
    return sorted(keys, key=lambda key: int(key.partition("@")[2]))


def _question_type_order(summary: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> List[str]:
    """Use the canonical summary order, retaining unexpected types if present."""
    ordered = list((summary.get("by_question_type") or {}).keys())
    for record in records:
        question_type = str(record.get("question_type") or "")
        if question_type and question_type not in ordered:
            ordered.append(question_type)
    if not ordered:
        raise ValueError("No question types found in retrieval summary or records")
    return ordered


def load_artifacts(input_dir: Path) -> Dict[str, Any]:
    """Load the fixed artifacts and fail early if they cannot be compared safely."""
    configs: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Dict[str, Any]] = {}
    records: Dict[str, List[Dict[str, Any]]] = {}

    for system in SYSTEM_ORDER:
        configs[system] = _read_json(input_dir / f"config_used_{system}.json")
        summaries[system] = _read_json(input_dir / f"retrieval_summary_{system}.json")
        records[system] = _read_jsonl(input_dir / f"retrieval_records_{system}.jsonl")

        configured_system = str((configs[system].get("system") or {}).get("name") or "")
        summarized_system = str(summaries[system].get("system") or "")
        if configured_system != system or summarized_system != system:
            raise ValueError(
                f"System identity mismatch for {system}: config={configured_system!r}, "
                f"summary={summarized_system!r}"
            )
        if any(str(row.get("system") or "") != system for row in records[system]):
            raise ValueError(f"Not every record in retrieval_records_{system}.jsonl is labelled {system!r}")

        expected_n = int((summaries[system].get("overall") or {}).get("n") or 0)
        if expected_n != len(records[system]):
            raise ValueError(
                f"Record count mismatch for {system}: summary says {expected_n}, "
                f"file contains {len(records[system])}"
            )

    comparison = _read_json(input_dir / "retrieval_comparison.json")
    comparison_systems = set(str(name) for name in comparison.get("systems") or [])
    if comparison_systems != set(SYSTEM_ORDER):
        raise ValueError(
            "retrieval_comparison.json must contain exactly Proposed, B2, and B1; "
            f"found {sorted(comparison_systems)!r}"
        )

    first_system = SYSTEM_ORDER[0]
    expected_ids = [str(row.get("qa_id") or "") for row in records[first_system]]
    if not all(expected_ids) or len(set(expected_ids)) != len(expected_ids):
        raise ValueError(f"{first_system} records must contain unique, non-empty qa_id values")
    expected_type_by_id = {
        str(row["qa_id"]): str(row.get("question_type") or "")
        for row in records[first_system]
    }
    for system in SYSTEM_ORDER[1:]:
        ids = [str(row.get("qa_id") or "") for row in records[system]]
        if set(ids) != set(expected_ids) or len(set(ids)) != len(ids):
            raise ValueError(f"QA ids in {system} do not match {first_system} exactly")
        type_by_id = {str(row["qa_id"]): str(row.get("question_type") or "") for row in records[system]}
        if type_by_id != expected_type_by_id:
            raise ValueError(f"Question-type labels differ between {first_system} and {system}")

    top_ks = {
        int((config.get("retrieval") or {}).get("top_k") or 0)
        for config in configs.values()
    }
    if len(top_ks) != 1 or next(iter(top_ks)) < 1:
        raise ValueError(f"The three experiments must have one positive shared top-k; found {sorted(top_ks)!r}")

    return {
        "configs": configs,
        "summaries": summaries,
        "records": records,
        "comparison": comparison,
        "hit_keys": _hit_keys(configs),
        "question_types": _question_type_order(summaries[first_system], records[first_system]),
        "qa_ids": expected_ids,
        "top_k": next(iter(top_ks)),
    }


# ════════════════════════════════════════════
# TABLE DATA
# ════════════════════════════════════════════


def _system_label(summary: Mapping[str, Any]) -> str:
    system = str(summary.get("system") or "")
    return str(summary.get("system_label") or SYSTEM_SHORT_LABELS.get(system, system))


def _recall_metric(summary: Mapping[str, Any]) -> str:
    return str((summary.get("gold") or {}).get("recall_metric") or "gold_recall")


def _overall_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one raw, CSV-friendly aggregate row for each system."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        overall = summary["overall"]
        config = data["configs"][system]
        metric = _recall_metric(summary)
        row: Dict[str, Any] = {
            "system": system,
            "system_label": _system_label(summary),
            "questions": int(overall.get("n") or 0),
            "mrr": _as_float(overall.get("mrr"), field=f"{system} overall mrr"),
            "gold_unit_recall": (
                _as_float(overall.get(metric), field=f"{system} {metric}")
                if metric == "gold_unit_recall"
                else None
            ),
            "gold_page_recall_proxy": (
                _as_float(overall.get(metric), field=f"{system} {metric}")
                if metric == "gold_page_recall_proxy"
                else None
            ),
            "gold_recall_metric": metric,
            "mean_context_tokens": _as_float(
                overall.get("context_token_count_mean"),
                field=f"{system} mean context tokens",
            ),
            "total_context_tokens": int(overall.get("context_token_count_total") or 0),
            "context_budget_tokens": int(
                ((config.get("answer") or {}).get("context_budget_tokens")) or 0
            ),
            "context_truncated_questions": sum(
                1 for record in data["records"][system] if bool(record.get("context_truncated"))
            ),
            "corpus_chunk_count": int(summary.get("corpus_chunk_count") or 0),
        }
        for key in data["hit_keys"]:
            row[key.replace("@", "_at_")] = _as_float(
                overall.get(key), field=f"{system} overall {key}"
            )
        rows.append(row)
    return rows


def _by_question_type_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create one row per system and question type, with recall kept separate."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        summary = data["summaries"][system]
        metric = _recall_metric(summary)
        by_type = summary.get("by_question_type") or {}
        for question_type in data["question_types"]:
            stats = by_type.get(question_type)
            if not isinstance(stats, dict):
                raise ValueError(f"{system} summary is missing question type {question_type!r}")
            row: Dict[str, Any] = {
                "system": system,
                "system_label": _system_label(summary),
                "question_type": question_type,
                "questions": int(stats.get("n") or 0),
                "mrr": _as_float(stats.get("mrr"), field=f"{system} {question_type} mrr"),
                "gold_unit_recall": (
                    _as_float(stats.get(metric), field=f"{system} {question_type} {metric}")
                    if metric == "gold_unit_recall"
                    else None
                ),
                "gold_page_recall_proxy": (
                    _as_float(stats.get(metric), field=f"{system} {question_type} {metric}")
                    if metric == "gold_page_recall_proxy"
                    else None
                ),
                "gold_recall_metric": metric,
                "mean_context_tokens": _as_float(
                    stats.get("context_token_count_mean"),
                    field=f"{system} {question_type} mean context tokens",
                ),
            }
            for key in data["hit_keys"]:
                row[key.replace("@", "_at_")] = _as_float(
                    stats.get(key), field=f"{system} {question_type} {key}"
                )
            rows.append(row)
    return rows


def _paired_question_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Create a wide audit table: every system's outcome for each QA item."""
    records_by_id = {
        system: {str(row["qa_id"]): row for row in data["records"][system]}
        for system in SYSTEM_ORDER
    }
    rows: List[Dict[str, Any]] = []
    for qa_id in data["qa_ids"]:
        source = records_by_id["proposed"][qa_id]
        row: Dict[str, Any] = {
            "qa_id": qa_id,
            "question_type": source.get("question_type", ""),
            "difficulty": source.get("difficulty", ""),
            "answer_mode": source.get("answer_mode", ""),
            "required_diagram": bool(source.get("required_diagram")),
            "required_formula": bool(source.get("required_formula")),
        }
        for system in SYSTEM_ORDER:
            record = records_by_id[system][qa_id]
            prefix = f"{system}_"
            row[f"{prefix}first_gold_rank"] = record.get("first_gold_rank")
            row[f"{prefix}mrr"] = _as_float(record.get("reciprocal_rank"), field=f"{qa_id} {system} mrr")
            row[f"{prefix}gold_recall_metric"] = record.get("gold_recall_kind", "")
            row[f"{prefix}gold_recall"] = _as_float(record.get("gold_recall"), field=f"{qa_id} {system} recall")
            row[f"{prefix}context_tokens"] = int(record.get("context_token_count") or 0)
            row[f"{prefix}context_truncated"] = bool(record.get("context_truncated"))
            for key in data["hit_keys"]:
                row[f"{prefix}{key.replace('@', '_at_')}"] = bool(record.get(key))
        rows.append(row)
    return rows


def _top_k_miss_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return auditable detail for questions with no relevant result in top-k."""
    top_k_key = f"hit@{data['top_k']}"
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        for record in data["records"][system]:
            if bool(record.get(top_k_key)):
                continue
            rows.append(
                {
                    "system": system,
                    "qa_id": record.get("qa_id", ""),
                    "question_type": record.get("question_type", ""),
                    "difficulty": record.get("difficulty", ""),
                    "gold_recall_metric": record.get("gold_recall_kind", ""),
                    "gold_recall": _as_float(
                        record.get("gold_recall"), field=f"{system} top-k miss recall"
                    ),
                    "gold_targets_total": int(record.get("gold_targets_total") or 0),
                    "gold_target_ids": "; ".join(str(value) for value in record.get("gold_target_ids") or []),
                    "context_tokens": int(record.get("context_token_count") or 0),
                    "context_truncated": bool(record.get("context_truncated")),
                    "retrieved_chunk_ids": "; ".join(
                        str(value) for value in record.get("retrieved_chunk_ids") or []
                    ),
                }
            )
    return rows


def _unit_level_delta_rows(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Compare Proposed with B2 only, where both systems support unit gold."""
    proposed = data["summaries"]["proposed"]
    b2 = data["summaries"]["b2"]
    if _recall_metric(proposed) != "gold_unit_recall" or _recall_metric(b2) != "gold_unit_recall":
        raise ValueError("Proposed and B2 must both report gold_unit_recall")

    metrics = [*data["hit_keys"], "mrr", "gold_unit_recall"]
    rows: List[Dict[str, Any]] = []
    for question_type, proposed_stats, b2_stats in [
        ("overall", proposed["overall"], b2["overall"]),
        *[
            (
                question_type,
                proposed["by_question_type"][question_type],
                b2["by_question_type"][question_type],
            )
            for question_type in data["question_types"]
        ],
    ]:
        for metric in metrics:
            proposed_value = _as_float(proposed_stats.get(metric), field=f"Proposed {question_type} {metric}")
            b2_value = _as_float(b2_stats.get(metric), field=f"B2 {question_type} {metric}")
            rows.append(
                {
                    "question_type": question_type,
                    "metric": metric,
                    "proposed": proposed_value,
                    "b2": b2_value,
                    "proposed_minus_b2": proposed_value - b2_value,
                }
            )
    return rows


def _rank_outcomes(data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Count first-relevant-result outcomes for the ranked retrieval chart."""
    rows: List[Dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        records = data["records"][system]
        hit_at_1 = sum(1 for row in records if row.get("first_gold_rank") == 1)
        hit_between = sum(
            1
            for row in records
            if isinstance(row.get("first_gold_rank"), int)
            and 1 < int(row["first_gold_rank"]) <= data["top_k"]
        )
        rows.append(
            {
                "system": system,
                "questions": len(records),
                "rank_1": hit_at_1,
                f"rank_2_to_{data['top_k']}": hit_between,
                "not_in_top_k": len(records) - hit_at_1 - hit_between,
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


def _display_hit_key(key: str) -> str:
    return key.upper()


def write_tables(data: Mapping[str, Any], output_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Write CSV tables and the standalone Markdown versions of key tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = _overall_rows(data)
    by_type = _by_question_type_rows(data)
    paired = _paired_question_rows(data)
    misses = _top_k_miss_rows(data)
    deltas = _unit_level_delta_rows(data)
    outcomes = _rank_outcomes(data)

    tables = {
        "overall_metrics": overall,
        "by_question_type": by_type,
        "per_question_system_comparison": paired,
        "top_k_misses": misses,
        "proposed_vs_b2_unit_level_deltas": deltas,
        "rank_outcomes": outcomes,
    }
    for name, rows in tables.items():
        _write_csv(output_dir / f"{name}.csv", rows)

    overall_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("questions", "n", _integer),
    ]
    overall_columns += [
        (key.replace("@", "_at_"), _display_hit_key(key), _percent)
        for key in data["hit_keys"]
    ]
    overall_columns += [
        ("mrr", "MRR", _decimal),
        ("gold_unit_recall", "Gold Unit Recall", _percent),
        ("gold_page_recall_proxy", "Gold Page Recall (proxy)", _percent),
        ("mean_context_tokens", "Mean context tokens", _integer),
        ("context_truncated_questions", "Context truncated", _integer),
        ("corpus_chunk_count", "Corpus chunks", _integer),
    ]
    (output_dir / "overall_metrics.md").write_text(
        "# Overall retrieval metrics\n\n"
        + _markdown_table(overall, overall_columns)
        + "\n\n"
        + "> B1's Gold Page Recall is a page-overlap proxy and is not comparable "
        "to the Gold Unit Recall of Proposed and B2.\n",
        encoding="utf-8",
    )

    type_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("question_type", "Question type", _text),
        ("questions", "n", _integer),
    ]
    type_columns += [
        (key.replace("@", "_at_"), _display_hit_key(key), _percent)
        for key in data["hit_keys"]
    ]
    type_columns += [
        ("mrr", "MRR", _decimal),
        ("gold_unit_recall", "Gold Unit Recall", _percent),
        ("gold_page_recall_proxy", "Gold Page Recall (proxy)", _percent),
        ("mean_context_tokens", "Mean context tokens", _integer),
    ]
    (output_dir / "by_question_type.md").write_text(
        "# Retrieval metrics by question type\n\n"
        + _markdown_table(by_type, type_columns)
        + "\n\n"
        + "> B1's recall column is a page-overlap proxy, not unit-level recall.\n",
        encoding="utf-8",
    )

    delta_columns = [
        ("question_type", "Question type", _text),
        ("metric", "Metric", _text),
        ("proposed", "Proposed", _percent),
        ("b2", "B2", _percent),
        ("proposed_minus_b2", "Proposed − B2", _percent),
    ]
    # MRR is conventionally expressed as a decimal rather than a percentage.
    delta_markdown_rows = [dict(row) for row in deltas]
    for row in delta_markdown_rows:
        if row["metric"] == "mrr":
            row["proposed"] = f"{row['proposed']:.3f}"
            row["b2"] = f"{row['b2']:.3f}"
            row["proposed_minus_b2"] = f"{row['proposed_minus_b2']:+.3f}"
    (output_dir / "proposed_vs_b2_unit_level_deltas.md").write_text(
        "# Proposed vs B2 — unit-level comparison\n\n"
        + _markdown_table(
            delta_markdown_rows,
            [
                ("question_type", "Question type", _text),
                ("metric", "Metric", _text),
                ("proposed", "Proposed", _text),
                ("b2", "B2", _text),
                ("proposed_minus_b2", "Proposed − B2", _text),
            ],
        )
        + "\n",
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


def _plot_overall_ranked_metrics(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    metric_keys = [*data["hit_keys"], "mrr"]
    metric_labels = [*[_display_hit_key(key) for key in data["hit_keys"]], "MRR"]
    x_values = list(range(len(metric_keys)))
    width = 0.22
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    for index, system in enumerate(SYSTEM_ORDER):
        overall = data["summaries"][system]["overall"]
        offset = (index - (len(SYSTEM_ORDER) - 1) / 2) * width
        bars = axis.bar(
            [x + offset for x in x_values],
            [_as_float(overall.get(key), field=f"{system} {key}") for key in metric_keys],
            width,
            label=SYSTEM_SHORT_LABELS[system],
            color=SYSTEM_COLOURS[system],
        )
        _annotate_bars(axis, bars)
    axis.set_title("Overall ranked-retrieval performance")
    axis.set_ylabel("Score")
    axis.set_xticks(x_values, metric_labels)
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures_dir, "01_overall_ranked_retrieval_metrics", dpi)
    plt.close(figure)


def _plot_unit_recall(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    """Plot only systems whose gold mapping resolves instructional units."""
    plt = _pyplot()
    systems = [
        system
        for system in SYSTEM_ORDER
        if _recall_metric(data["summaries"][system]) == "gold_unit_recall"
    ]
    values = [
        _as_float(data["summaries"][system]["overall"].get("gold_unit_recall"), field=f"{system} unit recall")
        for system in systems
    ]
    figure, axis = plt.subplots(figsize=(5.8, 4.7))
    bars = axis.bar(
        [SYSTEM_SHORT_LABELS[system] for system in systems],
        values,
        color=[SYSTEM_COLOURS[system] for system in systems],
        width=0.56,
    )
    _annotate_bars(axis, bars)
    axis.set_title("Gold Unit Recall")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5,
        0.01,
        "B1 excluded: its page-overlap proxy is not unit-level recall.",
        ha="center",
        fontsize=8.5,
        color="#4D4D4D",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    _save_figure(figure, figures_dir, "02_gold_unit_recall", dpi)
    plt.close(figure)


def _plot_question_type_metrics(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    metrics = [*data["hit_keys"], "mrr"]
    labels = [*[_display_hit_key(key) for key in data["hit_keys"]], "MRR"]
    question_types = data["question_types"]
    x_values = list(range(len(question_types)))
    width = 0.22
    figure, axes = plt.subplots(1, len(metrics), figsize=(5.1 * len(metrics), 4.9), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric, label in zip(axes, metrics, labels):
        for index, system in enumerate(SYSTEM_ORDER):
            stats_by_type = data["summaries"][system]["by_question_type"]
            offset = (index - (len(SYSTEM_ORDER) - 1) / 2) * width
            bars = axis.bar(
                [x + offset for x in x_values],
                [
                    _as_float(stats_by_type[question_type].get(metric), field=f"{system} {question_type} {metric}")
                    for question_type in question_types
                ],
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
    axes[0].set_ylabel("Score")
    axes[-1].legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    figure.suptitle("Ranked retrieval by question type", y=1.02, fontweight="bold")
    figure.tight_layout()
    _save_figure(figure, figures_dir, "03_ranked_metrics_by_question_type", dpi)
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
    axis.set_title("Mean context tokens per query")
    axis.set_ylabel("Whitespace tokens")
    axis.set_ylim(0, max(budget * 1.12, max(row["mean_context_tokens"] for row in rows) * 1.12))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    figure.tight_layout()
    _save_figure(figure, figures_dir, "04_mean_context_tokens", dpi)
    plt.close(figure)


def _plot_rank_outcomes(data: Mapping[str, Any], figures_dir: Path, dpi: int) -> None:
    plt = _pyplot()
    rows = _rank_outcomes(data)
    top_k = data["top_k"]
    segments = ["rank_1", f"rank_2_to_{top_k}", "not_in_top_k"]
    labels = ["Relevant at rank 1", f"Relevant at ranks 2–{top_k}", f"Not in top-{top_k}"]
    colours = ["#009E73", "#56B4E9", "#999999"]
    figure, axis = plt.subplots(figsize=(7.4, 4.6))
    for row_index, row in enumerate(rows):
        left = 0
        for segment, label, colour in zip(segments, labels, colours):
            value = int(row[segment]) / int(row["questions"])
            axis.barh(
                SYSTEM_SHORT_LABELS[row["system"]],
                value,
                left=left,
                color=colour,
                height=0.58,
                label=label if row_index == 0 else None,
            )
            if value >= 0.08:
                axis.text(left + value / 2, row_index, f"{100 * value:.0f}%", ha="center", va="center", fontsize=9)
            left += value
    axis.set_title("First relevant result within the retrieval depth")
    axis.set_xlabel("Questions")
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1))
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    figure.tight_layout()
    _save_figure(figure, figures_dir, "05_first_relevant_rank_outcomes", dpi)
    plt.close(figure)


def write_figures(data: Mapping[str, Any], output_dir: Path, dpi: int) -> None:
    """Write publication-ready PNG and SVG figures under ``output_dir/figures``."""
    figures_dir = output_dir / "figures"
    _plot_overall_ranked_metrics(data, figures_dir, dpi)
    _plot_unit_recall(data, figures_dir, dpi)
    _plot_question_type_metrics(data, figures_dir, dpi)
    _plot_context_tokens(data, figures_dir, dpi)
    _plot_rank_outcomes(data, figures_dir, dpi)


# ════════════════════════════════════════════
# ANALYSIS INDEX
# ════════════════════════════════════════════


def write_report_index(data: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]], output_dir: Path) -> None:
    """Write the report landing page that links all generated artifacts."""
    overall = tables["overall_metrics"]
    comparison = data["comparison"]
    protocol_equal = bool(comparison.get("protocol_equal_across_systems"))
    protocol = data["summaries"]["proposed"].get("protocol") or {}
    top_k = data["top_k"]

    overview_columns: List[tuple[str, str, Formatter]] = [
        ("system", "System", _text),
        ("questions", "n", _integer),
    ]
    overview_columns += [
        (key.replace("@", "_at_"), _display_hit_key(key), _percent)
        for key in data["hit_keys"]
    ]
    overview_columns += [
        ("mrr", "MRR", _decimal),
        ("mean_context_tokens", "Mean context tokens", _integer),
    ]

    unit_rows = [row for row in overall if row["gold_unit_recall"] is not None]
    unit_columns = [
        ("system", "System", _text),
        ("gold_unit_recall", "Gold Unit Recall", _percent),
    ]

    b1_note = str((data["summaries"]["b1"].get("gold") or {}).get("disclaimer") or "")
    lines = [
        "# Retrieval Results Analysis",
        "",
        "This bundle was generated from the fixed retrieval artifacts in `retrieval_eval/`. "
        "It is a reporting-only step; it does not rerun embeddings, retrieval, or answer generation.",
        "",
        "## Protocol check",
        "",
        f"- Questions per system: **{overall[0]['questions']}**",
        f"- Retrieval depth: **top-{top_k}**",
        f"- Query model: `{protocol.get('query_embedding_model', '—')}`",
        f"- Shared comparison-critical protocol: **{'yes' if protocol_equal else 'no'}**",
        "",
        "## Overall ranked retrieval",
        "",
        _markdown_table(overall, overview_columns),
        "",
        "![Overall ranked retrieval metrics](figures/01_overall_ranked_retrieval_metrics.png)",
        "",
        "## Gold Unit Recall",
        "",
        _markdown_table(unit_rows, unit_columns),
        "",
        "![Gold Unit Recall](figures/02_gold_unit_recall.png)",
        "",
        f"> **B1 caveat.** {b1_note}",
        "",
        "## By question type",
        "",
        "See [by_question_type.md](by_question_type.md) and "
        "[by_question_type.csv](by_question_type.csv) for the complete table.",
        "",
        "![Ranked metrics by question type](figures/03_ranked_metrics_by_question_type.png)",
        "",
        "## Context and first-relevant-rank outcomes",
        "",
        "![Mean context tokens](figures/04_mean_context_tokens.png)",
        "",
        "![First relevant rank outcomes](figures/05_first_relevant_rank_outcomes.png)",
        "",
        "## Output files",
        "",
        "| Artifact | Purpose |",
        "|---|---|",
        "| [overall_metrics.csv](overall_metrics.csv) / [overall_metrics.md](overall_metrics.md) | Headline metrics, recall separated by granularity, context use, and corpus size. |",
        "| [by_question_type.csv](by_question_type.csv) / [by_question_type.md](by_question_type.md) | Per-question-type retrieval and context metrics. |",
        "| [proposed_vs_b2_unit_level_deltas.csv](proposed_vs_b2_unit_level_deltas.csv) | Proposed − B2 deltas where both methods use true unit-level gold. |",
        "| [per_question_system_comparison.csv](per_question_system_comparison.csv) | Wide, QA-by-QA audit trail for all three systems. |",
        f"| [top_k_misses.csv](top_k_misses.csv) | Every question with no relevant result in top-{top_k}, including retrieved chunk IDs. |",
        "| [rank_outcomes.csv](rank_outcomes.csv) | Counts used in the first-relevant-rank outcome figure. |",
        "| [figures/](figures/) | Every chart in both PNG and SVG. |",
        "",
    ]
    (output_dir / "retrieval_results_report.md").write_text("\n".join(lines), encoding="utf-8")


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
        print(f"retrieval results report: {exc}", file=sys.stderr)
        return 2

    print(f"Done. Retrieval tables and figures written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
