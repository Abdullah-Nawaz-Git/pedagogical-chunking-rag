"""
ragkit.retrieval.corpus
═══════════════════════

Loads the frozen evaluation inputs: the QA dataset, the per-system gold mapping,
and the per-system chunk cache.

Nothing here regenerates or rewrites an artifact — the QA dataset and the three
``gold_mapping_*.jsonl`` files are the source of truth and are read verbatim.
Chunk IDs and provenance fields are preserved exactly as they appear in
``chunks.json``, because those IDs are the anchors gold matching relies on.

The one piece of interpretation this module performs is turning a gold mapping
into a system-appropriate :class:`GoldTargets`:

    * Proposed / B2 → targets are **source-block ids** (true instructional units),
      taken from ``relevant_chunks_by_gold_block``.
    * B1 → targets are **page numbers**, taken from ``relevant_chunks_by_gold_page``.
      This is the page-overlap proxy from ``ragkit.qa.gold_mapping``; it is never
      relabelled as unit-level relevance.

Which branch applies is decided by the system's ``gold_granularity`` config, not
by an ``if system == "b1"`` check in the evaluation loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.retrieval.corpus")


# ════════════════════════════════════════════
# GOLD TARGETS
# ════════════════════════════════════════════


@dataclass(frozen=True)
class GoldTargets:
    """The gold relevance for one QA item in one system.

    ``target_ids`` are source-block ids for unit-level systems and page numbers
    (as strings) for B1's page proxy. ``chunks_by_target`` maps each target to the
    chunk ids the gold mapping deemed relevant for it, which is what Gold Unit /
    Gold Page recall is computed against.
    """

    qa_id: str
    system: str
    granularity: str
    mapping_method: str
    mapping_status: str
    target_ids: tuple[str, ...]
    chunks_by_target: Dict[str, Set[str]]
    all_relevant_chunk_ids: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_gold(self) -> bool:
        """True when at least one target has at least one relevant chunk."""
        return any(self.chunks_by_target.get(t) for t in self.target_ids)


def _targets_from_mapping(
    mapping: Dict[str, Any],
    system: cfg.RetrievalSystemConfig,
) -> GoldTargets:
    """Convert one gold-mapping record into ``GoldTargets`` for ``system``.

    The mapping key that holds relevance is chosen from the system's configured
    gold granularity, so unit-level and proxy systems flow through the same code
    path with different data.
    """
    if system.is_unit_level:
        by_target = mapping.get("relevant_chunks_by_gold_block") or {}
        # Preserve the mapping's own required-target order for determinism.
        ordered = [str(b) for b in (mapping.get("required_gold_source_block_ids") or [])]
    else:
        by_target = mapping.get("relevant_chunks_by_gold_page") or {}
        ordered = [str(p) for p in (mapping.get("required_gold_page_numbers") or [])]

    # Include any target present in the mapping payload but absent from the
    # required list, so nothing is silently dropped.
    for key in by_target:
        if str(key) not in ordered:
            ordered.append(str(key))

    chunks_by_target: Dict[str, Set[str]] = {}
    for target in ordered:
        hits = by_target.get(target) or by_target.get(_maybe_int(target)) or []
        chunk_ids: Set[str] = set()
        for hit in hits:
            # A hit only counts when the mapping marked it relevant.
            if isinstance(hit, dict) and hit.get("is_relevant") and hit.get("chunk_id"):
                chunk_ids.add(str(hit["chunk_id"]))
        chunks_by_target[target] = chunk_ids

    warnings = tuple(str(w) for w in (mapping.get("warnings") or []))
    return GoldTargets(
        qa_id=str(mapping.get("qa_id")),
        system=system.name,
        granularity=system.gold_granularity,
        mapping_method=str(mapping.get("mapping_method") or ""),
        mapping_status=str(mapping.get("mapping_status") or ""),
        target_ids=tuple(ordered),
        chunks_by_target=chunks_by_target,
        all_relevant_chunk_ids=tuple(
            str(c) for c in (mapping.get("all_relevant_chunk_ids") or [])
        ),
        warnings=warnings,
    )


def _maybe_int(value: str) -> Any:
    """Return ``value`` as an int when possible (B1 page keys may be numeric)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


# ════════════════════════════════════════════
# EVALUATION CORPUS
# ════════════════════════════════════════════


@dataclass
class EvaluationCorpus:
    """Everything one retrieval run needs, loaded once and reused per question."""

    system: cfg.RetrievalSystemConfig
    qa_items: List[Dict[str, Any]]
    gold_by_qa_id: Dict[str, GoldTargets]
    chunks_by_id: Dict[str, Dict[str, Any]]

    def chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Look up a chunk by the exact id stored in the cache / index."""
        return self.chunks_by_id.get(chunk_id)

    def gold(self, qa_id: str) -> Optional[GoldTargets]:
        return self.gold_by_qa_id.get(qa_id)


def load_corpus(config: cfg.RetrievalExperimentConfig) -> EvaluationCorpus:
    """Load the QA dataset, the system's gold mapping, and its chunk cache."""
    system = config.system
    qa_dir = Path(config.qa_dataset_dir)

    qa_path = qa_dir / config.qa_dataset_filename
    qa_items = schemas.read_jsonl(qa_path)
    if not qa_items:
        raise FileNotFoundError(
            f"No QA dataset at {qa_path}. The retrieval experiments read the "
            "frozen dataset produced by 'python -m experiments.qa_dataset finalize'."
        )

    gold_path = qa_dir / system.gold_mapping_filename
    gold_rows = schemas.read_jsonl(gold_path)
    if not gold_rows:
        raise FileNotFoundError(
            f"No gold mapping at {gold_path}. Run "
            "'python -m experiments.qa_dataset map-gold' first."
        )

    gold_by_qa_id: Dict[str, GoldTargets] = {}
    for row in gold_rows:
        # Guard against evaluating a system against another system's mapping.
        row_system = str(row.get("system") or "")
        if row_system and row_system != system.name:
            raise ValueError(
                f"{gold_path} contains records for system {row_system!r}, "
                f"but this run evaluates {system.name!r}."
            )
        targets = _targets_from_mapping(row, system)
        gold_by_qa_id[targets.qa_id] = targets

    chunks = schemas.load_chunks(system.chunks_path)
    # Preserve chunk ids and provenance exactly; index by the id used in Pinecone.
    chunks_by_id = {
        str(chunk["chunk_id"]): chunk for chunk in chunks if chunk.get("chunk_id")
    }

    missing_gold = [qa["qa_id"] for qa in qa_items if qa["qa_id"] not in gold_by_qa_id]
    if missing_gold:
        logger.warning(
            "%d QA item(s) have no %s gold mapping and will be scored as misses: %s",
            len(missing_gold), system.name, ", ".join(missing_gold[:5]),
        )

    logger.info(
        "Loaded corpus for %s: qa_items=%d gold_mappings=%d chunks=%d",
        system.name, len(qa_items), len(gold_by_qa_id), len(chunks_by_id),
    )
    return EvaluationCorpus(
        system=system,
        qa_items=qa_items,
        gold_by_qa_id=gold_by_qa_id,
        chunks_by_id=chunks_by_id,
    )
