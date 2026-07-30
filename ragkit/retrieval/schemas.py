"""
ragkit.retrieval.schemas
════════════════════════

Data contracts + the output-directory layout for retrieval evaluation.

This module is the retrieval counterpart of ``ragkit.qa.schemas`` and stays a
dependency-free contracts layer: it defines the ONE per-question record schema
that every system writes, the output paths, and the config (de)serialisation
helpers. JSONL/CSV/JSON writing itself is reused verbatim from
``ragkit.qa.schemas`` so Arabic round-trips identically across both pipelines.

The record schema is deliberately shared by all three systems. Only the *values*
of ``gold_granularity`` / ``gold_recall_kind`` differ, so B1's page-overlap proxy
is visible in every row instead of being silently folded into a "recall" column.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config as cfg

# Reuse the QA pipeline's UTF-8 JSON/JSONL/CSV helpers rather than duplicating
# them: identical on-disk conventions (ensure_ascii=False, utf-8-sig for CSV).
from ..qa.schemas import (  # noqa: F401 - re-exported for retrieval callers
    iter_jsonl,
    load_chunks,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)

# Question types, in canonical order, re-exported for aggregation code.
QUESTION_TYPES: tuple[str, ...] = cfg.QA_QUESTION_TYPES


# ════════════════════════════════════════════
# PER-QUESTION RECORD SCHEMA
# ════════════════════════════════════════════

# The full per-question column set, in report order. Written to
# ``retrieval_records_<system>.jsonl`` and ``.csv`` for every system.
RECORD_COLUMNS: tuple[str, ...] = (
    # ── identity ─────────────────────────────────────────────────────────
    "qa_id",
    "system",
    "question_type",
    "difficulty",
    "answer_mode",
    "required_diagram",
    "required_formula",
    # ── retrieval result ─────────────────────────────────────────────────
    "retrieved_chunk_ids",
    "retrieved_scores",
    "retrieved_ranks_relevant",
    "first_gold_rank",
    # ── ranked-retrieval metrics ─────────────────────────────────────────
    "hit@1",
    "hit@5",
    "reciprocal_rank",
    # ── gold recall (semantics depend on the system's provenance) ─────────
    "gold_granularity",
    "gold_recall_kind",
    "gold_recall",
    "gold_targets_total",
    "gold_targets_covered",
    "gold_target_ids",
    "gold_relevant_chunk_ids",
    "mapping_status",
    # ── generator context accounting ─────────────────────────────────────
    "context_chunk_ids",
    "context_token_count",
    "context_token_budget",
    "context_truncated",
    "context_char_count",
    "context_estimated_model_tokens",
    # ── answer generation (optional) ──────────────────────────────────────
    "answer_reference_ar",
    "generated_answer_ar",
    "answer_provider",
    "answer_model",
    "answer_prompt_version",
    "answer_error",
    # ── provenance / caveats ─────────────────────────────────────────────
    "warnings",
)

# Columns whose semantics change per system. Kept explicit so reporting code can
# label B1's proxy without special-casing B1 anywhere in the control flow.
RECALL_KIND_UNIT = "gold_unit_recall"
RECALL_KIND_PAGE_PROXY = "gold_page_recall_proxy"


def recall_kind_for(system: cfg.RetrievalSystemConfig) -> str:
    """Return the honest recall label for a system (unit vs page proxy)."""
    return RECALL_KIND_UNIT if system.is_unit_level else RECALL_KIND_PAGE_PROXY


# ════════════════════════════════════════════
# OUTPUT-DIRECTORY LAYOUT
# ════════════════════════════════════════════


class RetrievalOutputLayout:
    """Centralises every retrieval artifact path under the output directory.

    Per-system files are suffixed with the system name so the three experiments
    never overwrite one another, matching how the ingestion runs keep separate
    cache directories and the QA stage keeps separate ``gold_mapping_*`` files.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure(self) -> "RetrievalOutputLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    # Run provenance
    def config_used(self, system: str) -> Path:
        return self.root / f"config_used_{system}.json"

    # Per-question detail
    def records_jsonl(self, system: str) -> Path:
        return self.root / f"retrieval_records_{system}.jsonl"

    def records_csv(self, system: str) -> Path:
        return self.root / f"retrieval_records_{system}.csv"

    # Per-system aggregates
    def summary_json(self, system: str) -> Path:
        return self.root / f"retrieval_summary_{system}.json"

    def summary_md(self, system: str) -> Path:
        return self.root / f"retrieval_summary_{system}.md"

    # Cross-system aggregates (written by the ``compare`` command)
    @property
    def comparison_json(self) -> Path:
        return self.root / "retrieval_comparison.json"

    @property
    def comparison_md(self) -> Path:
        return self.root / "retrieval_comparison.md"


# ════════════════════════════════════════════
# CONFIG FILE LOADING / SERIALISATION
# ════════════════════════════════════════════


def load_retrieval_config(
    system: cfg.RetrievalSystemConfig,
    path: Optional[str | Path] = None,
    **overrides: Any,
) -> cfg.RetrievalExperimentConfig:
    """Build a ``RetrievalExperimentConfig`` from defaults + optional file + CLI.

    Precedence is defaults < file < ``overrides`` (the CLI), matching
    ``ragkit.qa.schemas.load_qa_config``. ``None`` overrides are ignored so an
    unset CLI flag never clobbers a file value.
    """
    config = cfg.RetrievalExperimentConfig(system=system)
    if path is not None:
        # Reuse the QA config-file reader (YAML optional, JSON always available).
        from ..qa.schemas import _apply_overrides, _read_config_file

        data = _read_config_file(Path(path))
        # ``system`` is identity, not a knob: it may never be overridden by file.
        data.pop("system", None)
        config = _apply_overrides(config, data)
    applied = {k: v for k, v in overrides.items() if v is not None}
    if applied:
        config = replace(config, **applied)
    return config


def config_to_dict(config: cfg.RetrievalExperimentConfig) -> Dict[str, Any]:
    """Serialise a ``RetrievalExperimentConfig`` (nested dataclasses) to a dict."""

    def _convert(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return _convert(config)
