"""
ragkit.judge.schemas
════════════════════

Data contracts + the output-directory layout for LLM-as-judge scoring.

This is the judge counterpart of ``ragkit.retrieval.schemas`` and stays a
dependency-free contracts layer: it defines the ONE atomic score-record schema
(one row per ``qa_id`` × ``system`` × ``metric``), the fixed output paths, the
resume-key helper, and config (de)serialisation. JSON/JSONL/CSV writing is reused
verbatim from ``ragkit.qa.schemas`` so Arabic round-trips identically across
every pipeline.

An atomic (item, system, metric) record is deliberately the unit of work: it
makes the run resumable at metric granularity, lets a single malformed judge
reply fail transparently on its own row, and keeps the CSV flat and auditable.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config as cfg

# Reuse the QA pipeline's UTF-8 JSON/JSONL/CSV helpers (ensure_ascii=False, CSV
# BOM) so Arabic, RTL text, and LaTeX survive on disk unchanged.
from ..qa.schemas import (  # noqa: F401 - re-exported for judge callers
    iter_jsonl,
    load_chunks,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)

# Canonical dimension + system order, re-exported for aggregation code.
JUDGE_DIMENSIONS: tuple[str, ...] = cfg.JUDGE_DIMENSIONS
JUDGE_SYSTEM_ORDER: tuple[str, ...] = cfg.JUDGE_SYSTEM_ORDER


# ════════════════════════════════════════════
# SCORE-RECORD STATUSES
# ════════════════════════════════════════════

# A dimension was scored and its score passed range validation.
STATUS_SCORED = "scored"
# A generation-only dimension (faithfulness / answer_relevancy) had no generated
# answer to score — recorded, not silently dropped.
STATUS_SKIPPED_NO_ANSWER = "skipped_no_generated_answer"
# A context-dependent dimension had no retrieved context at all.
STATUS_SKIPPED_NO_CONTEXT = "skipped_no_context"
# The judge replied but the JSON could not be parsed (raw preserved).
STATUS_PARSE_ERROR = "parse_error"
# The provider call itself failed after exhausting retries (raw/error preserved).
STATUS_API_ERROR = "api_error"
# JSON parsed but the score was out of the [0, 1] range or non-numeric.
STATUS_INVALID_SCORE = "invalid_score"
# The reply was truncated (unterminated JSON) but a valid numeric score was
# recovered from the raw text. The value enters the aggregates; the raw text is
# preserved so the verdict stays auditable.
STATUS_RECOVERED = "recovered"

# Statuses that carry a usable numeric score into the aggregates.
SCORED_STATUSES: frozenset = frozenset({STATUS_SCORED, STATUS_RECOVERED})
# Statuses that represent a hard failure (as opposed to a legitimate skip).
FAILURE_STATUSES: frozenset = frozenset(
    {STATUS_PARSE_ERROR, STATUS_API_ERROR, STATUS_INVALID_SCORE}
)


# ════════════════════════════════════════════
# PER-ITEM (ATOMIC) RECORD SCHEMA
# ════════════════════════════════════════════

# Full column set, in report order, for judge_scores.jsonl / .csv.
RECORD_COLUMNS: tuple[str, ...] = (
    # ── identity ─────────────────────────────────────────────────────────
    "qa_id",
    "system",
    "system_label",
    "experimental_role",
    "metric",
    "metric_display",
    # ── question metadata ────────────────────────────────────────────────
    "question_type",
    "difficulty",
    "answer_mode",
    "required_diagram",
    "required_formula",
    "question_ar",
    # ── the exact judge inputs (only those this metric was shown) ─────────
    "reference_answer_ar",
    "generated_answer_ar",
    "answer_available",
    "judged_context_chunk_ids",
    "context_source",
    "context_token_count",
    "context_token_budget",
    "context_truncated",
    "context_char_count",
    "context_estimated_model_tokens",
    # ── retrieval result (audit context, NOT shown to the judge) ──────────
    "retrieved_chunk_ids",
    "retrieved_ranks",
    "retrieved_scores",
    # ── the judge's structured output ─────────────────────────────────────
    "score",
    "rationale",
    "confidence",
    "judge_status",
    "parse_ok",
    "retries_used",
    # ── judge provenance ──────────────────────────────────────────────────
    "judge_provider",
    "judge_model",
    "prompt_version",
    "judge_temperature",
    "timestamp_utc",
    # ── gold-mapping metadata (reporting only; NEVER a judge input) ───────
    "gold_granularity",
    "mapping_type",
    "mapping_method",
    "mapping_status",
    "mapping_caveat",
    # ── audit trail ──────────────────────────────────────────────────────
    "warnings",
    "raw_response",
)


def record_key(record: Dict[str, Any]) -> tuple[str, str, str]:
    """The resume key identifying one atomic unit of work."""
    return (
        str(record.get("qa_id") or ""),
        str(record.get("system") or ""),
        str(record.get("metric") or ""),
    )


def sort_key(record: Dict[str, Any]) -> tuple[Any, ...]:
    """Deterministic ledger order: system, qa_id, then canonical metric order."""
    metric = str(record.get("metric") or "")
    try:
        metric_rank = JUDGE_DIMENSIONS.index(metric)
    except ValueError:
        metric_rank = len(JUDGE_DIMENSIONS)
    try:
        system_rank = JUDGE_SYSTEM_ORDER.index(str(record.get("system") or ""))
    except ValueError:
        system_rank = len(JUDGE_SYSTEM_ORDER)
    return (system_rank, str(record.get("qa_id") or ""), metric_rank)


# ════════════════════════════════════════════
# OUTPUT-DIRECTORY LAYOUT
# ════════════════════════════════════════════


class JudgeOutputLayout:
    """Centralises every judge artifact path under the output directory.

    Unlike the retrieval layout, judge outputs are NOT suffixed per system: the
    per-item ledger, the summary, and the manifest each hold every system so the
    cross-system paired comparison (the paper's key result) reads a single file.
    Per-system judge runs merge into the same ledger by resume key.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure(self) -> "JudgeOutputLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def scores_jsonl(self) -> Path:
        return self.root / "judge_scores.jsonl"

    @property
    def scores_csv(self) -> Path:
        return self.root / "judge_scores.csv"

    @property
    def summary_json(self) -> Path:
        return self.root / "judge_summary.json"

    @property
    def summary_md(self) -> Path:
        return self.root / "judge_summary.md"

    @property
    def run_manifest(self) -> Path:
        return self.root / "judge_run_manifest.json"

    @property
    def llm_calls_jsonl(self) -> Path:
        """Structured per-attempt judge LLM call logs."""
        return self.root / "judge_llm_calls.jsonl"

    def config_used(self, system: str) -> Path:
        return self.root / f"judge_config_used_{system}.json"


# ════════════════════════════════════════════
# CONFIG FILE LOADING / SERIALISATION
# ════════════════════════════════════════════


def load_judge_config(
    system: cfg.JudgeSystemConfig,
    path: Optional[str | Path] = None,
    **overrides: Any,
) -> cfg.JudgeExperimentConfig:
    """Build a ``JudgeExperimentConfig`` from defaults + optional file + CLI.

    Precedence is defaults < file < ``overrides`` (the CLI), matching
    ``ragkit.retrieval.schemas.load_retrieval_config``. ``None`` overrides are
    ignored so an unset CLI flag never clobbers a file value.
    """
    config = cfg.JudgeExperimentConfig(system=system)
    if path is not None:
        from ..qa.schemas import _apply_overrides, _read_config_file

        data = _read_config_file(Path(path))
        # ``system`` is identity, not a knob: never overridable by file.
        data.pop("system", None)
        config = _apply_overrides(config, data)
    applied = {k: v for k, v in overrides.items() if v is not None}
    if applied:
        config = replace(config, **applied)
    return config


def config_to_dict(config: cfg.JudgeExperimentConfig) -> Dict[str, Any]:
    """Serialise a ``JudgeExperimentConfig`` (nested dataclasses) to a dict."""

    def _convert(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return _convert(config)
