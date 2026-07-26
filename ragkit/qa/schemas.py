"""
ragkit.qa.schemas
═════════════════

Data contracts + small IO helpers shared by every QA stage.

Records are exchanged between stages as plain JSON-serialisable ``dict``s (one
JSON object per JSONL line) so they round-trip losslessly and remain easy to
inspect on disk. This module centralises:

    * the canonical field-name sets and enums,
    * the ``QAConfig`` file loader (YAML or JSON overrides of the frozen
      defaults defined in ``ragkit.config``),
    * UTF-8 JSON / JSONL / CSV writers (always ``ensure_ascii=False`` so Arabic
      is preserved verbatim),
    * a ``source_payload`` builder that snapshots the Proposed-chunk fields the
      prompts need,
    * the output-path layout under the QA output directory.

Nothing here talks to an LLM or applies selection logic; those live in their own
modules so this file stays a dependency-free contracts layer.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from .. import config as cfg

# Re-export for convenience so callers can ``from ragkit.qa.schemas import QUESTION_TYPES``.
QUESTION_TYPES: tuple[str, ...] = cfg.QA_QUESTION_TYPES

# Candidate lifecycle statuses written to qa_candidates.jsonl.
STATUS_GENERATED = "generated"
STATUS_PARSE_FAILED = "parse_failed"
STATUS_GENERATION_FAILED = "generation_failed"
STATUS_VALIDATED = "validated"

# Fields the model is allowed to set (everything else is copied from the task).
LLM_OUTPUT_FIELDS: tuple[str, ...] = (
    "question_ar",
    "question_en",
    "answer_reference_ar",
    "difficulty",
    "answer_mode",
    "required_diagram",
    "required_formula",
)


# ════════════════════════════════════════════
# OUTPUT-DIRECTORY LAYOUT
# ════════════════════════════════════════════


class QAOutputLayout:
    """Centralises every artifact path under the QA output directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def ensure(self) -> "QAOutputLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self

    # Stage 0 / config
    @property
    def config_used(self) -> Path:
        return self.root / "config_used.json"

    # Stage 1 — selection
    @property
    def source_selection_plan(self) -> Path:
        return self.root / "source_selection_plan.jsonl"

    # Stage 3 — generation
    @property
    def qa_candidates(self) -> Path:
        return self.root / "qa_candidates.jsonl"

    # Stage 4 — validation
    @property
    def qa_validated(self) -> Path:
        return self.root / "qa_validated.jsonl"

    @property
    def qa_rejections(self) -> Path:
        return self.root / "qa_rejections.csv"

    # Stage 5 — finalize
    @property
    def qa_dataset_jsonl(self) -> Path:
        return self.root / "qa_dataset_v1.jsonl"

    @property
    def qa_dataset_csv(self) -> Path:
        return self.root / "qa_dataset_v1.csv"

    @property
    def qa_dataset_summary_json(self) -> Path:
        return self.root / "qa_dataset_summary.json"

    @property
    def qa_dataset_summary_md(self) -> Path:
        return self.root / "qa_dataset_summary.md"

    @property
    def qa_deficit_report(self) -> Path:
        return self.root / "qa_deficit_report.md"

    # Stage 6 — gold mapping
    @property
    def gold_mapping_b1(self) -> Path:
        return self.root / "gold_mapping_b1.jsonl"

    @property
    def gold_mapping_b2(self) -> Path:
        return self.root / "gold_mapping_b2.jsonl"

    @property
    def gold_mapping_proposed(self) -> Path:
        return self.root / "gold_mapping_proposed.jsonl"

    @property
    def gold_mapping_summary_md(self) -> Path:
        return self.root / "gold_mapping_summary.md"

    @property
    def unmapped_qa_items(self) -> Path:
        return self.root / "unmapped_qa_items.csv"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"


# ════════════════════════════════════════════
# CONFIG FILE LOADING (YAML / JSON OVERRIDES)
# ════════════════════════════════════════════


def _read_config_file(path: Path) -> Dict[str, Any]:
    """Read a YAML or JSON config file into a plain dict.

    YAML support is optional: it is only imported when a ``.yaml``/``.yml`` file
    is actually supplied, so the package has no hard dependency on PyYAML for the
    default (no ``--config``) path or for JSON configs.
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyYAML is required to read YAML config files; install pyyaml or "
                "pass a JSON config instead."
            ) from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text) if text.strip() else {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level")
    return data


def _apply_overrides(obj: Any, data: Dict[str, Any]) -> Any:
    """Recursively apply a dict of overrides onto a frozen dataclass instance."""
    if not is_dataclass(obj):
        return data
    field_names = {f.name for f in fields(obj)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = _apply_overrides(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return replace(obj, **kwargs)


def load_qa_config(
    path: Optional[str | Path] = None,
    *,
    seed: Optional[int] = None,
    provider: Optional[str] = None,
) -> cfg.QAConfig:
    """Build a ``QAConfig`` from defaults, optionally overridden by a file.

    ``seed`` and ``provider`` (typically from CLI flags) take precedence over
    both the defaults and the file so command-line intent always wins.
    """
    config = cfg.QAConfig()
    if path is not None:
        config = _apply_overrides(config, _read_config_file(Path(path)))
    if seed is not None:
        config = replace(config, random_seed=seed)
    if provider is not None:
        config = replace(config, provider=provider)
    return config


def config_to_dict(config: cfg.QAConfig) -> Dict[str, Any]:
    """Serialise a ``QAConfig`` (with nested dataclasses) to a plain dict."""

    def _convert(value: Any) -> Any:
        if is_dataclass(value):
            return {f.name: _convert(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, tuple):
            return list(value)
        return value

    return _convert(config)


# ════════════════════════════════════════════
# JSON / JSONL / CSV IO (UTF-8, ensure_ascii=False)
# ════════════════════════════════════════════


def write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(path: Path | str) -> List[Dict[str, Any]]:
    """Load a chunks.json list produced by one of the ingest experiments."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of chunks in {path}")
    return data


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> Path:
    """Write rows to a UTF-8 CSV with a BOM so Excel renders Arabic correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _csv_cell(row.get(c, "")) for c in columns})
    return path


def _csv_cell(value: Any) -> Any:
    """Flatten lists/dicts into compact JSON so CSV cells stay single-valued."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


# ════════════════════════════════════════════
# SOURCE-PAYLOAD SNAPSHOT
# ════════════════════════════════════════════

# The Proposed-chunk fields preserved into each task's ``source_payload`` so the
# prompt builder and validator never need the original chunks.json again.
SOURCE_PAYLOAD_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "source_block_id",
    "source_block_ids",
    "source_page_numbers",
    "source_block_coverage",
    "page_number",
    "page_range",
    "content_type",
    "unit_number",
    "lesson_number",
    "lesson_title_ar",
    "lesson_title_en",
    "heading_ar",
    "main_text_ar",
    "math_expressions",
    "diagrams",
    "diagram_image_urls",
    "named_elements",
    "has_diagram",
    "has_math",
)


def build_source_payload(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Snapshot the chunk fields needed downstream, with safe defaults."""
    payload: Dict[str, Any] = {}
    for key in SOURCE_PAYLOAD_FIELDS:
        payload[key] = chunk.get(key)
    # Normalise the collection fields so downstream code never sees ``None``.
    payload["source_block_ids"] = list(chunk.get("source_block_ids") or [])
    payload["source_page_numbers"] = list(chunk.get("source_page_numbers") or [])
    payload["math_expressions"] = list(chunk.get("math_expressions") or [])
    payload["diagrams"] = list(chunk.get("diagrams") or [])
    payload["diagram_image_urls"] = list(chunk.get("diagram_image_urls") or [])
    payload["named_elements"] = dict(chunk.get("named_elements") or {})
    payload["source_block_coverage"] = dict(chunk.get("source_block_coverage") or {})
    return payload
