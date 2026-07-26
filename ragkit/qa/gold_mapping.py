"""
ragkit.qa.gold_mapping
═══════════════════════

Stage 6 — map each final QA item onto relevant chunks in B1, B2, and Proposed.

A gold mapping answers: *for this QA item, which chunks in a given system should
count as relevant during retrieval evaluation?* This stage never modifies the
QA dataset — it only reads it and the three chunk corpora.

    * Proposed / B2 use **source-block** relevance (a chunk is relevant to a gold
      block if it contains that block, subject to a coverage threshold).
    * B1 uses a **page-overlap proxy** because OCR chunks retain no source-block
      provenance; the resulting metric is Gold *Page* Recall, not Gold Unit
      Recall, and every B1 record carries a warning saying so.

Items that cannot be fully mapped are recorded in ``unmapped_qa_items.csv`` with
high-severity warnings; they are never silently removed or rewritten.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.qa.map")

_B1_WARNING = (
    "B1 lacks source-block provenance. Mapping uses page-range overlap only, "
    "not instructional-unit relevance."
)


# ════════════════════════════════════════════
# SHARED HELPERS
# ════════════════════════════════════════════


def _page_range(chunk: Dict[str, Any]) -> Tuple[int, int]:
    """Inclusive page range for a chunk, falling back to [page_number]*2."""
    pr = chunk.get("page_range")
    if isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, int) for x in pr):
        return int(pr[0]), int(pr[1])
    page = chunk.get("page_number")
    if isinstance(page, int):
        return page, page
    return -1, -1  # sentinel: never overlaps a real page


def _coverage_for_block(chunk: Dict[str, Any], block_id: str) -> Optional[float]:
    coverage = chunk.get("source_block_coverage") or {}
    if block_id in coverage:
        try:
            return float(coverage[block_id])
        except (TypeError, ValueError):
            return None
    return None


def _b2_threshold(question_type: str, config: cfg.QAConfig) -> float:
    if question_type == "worked_example_reasoning":
        return config.b2_mapping.worked_example_min_source_block_coverage
    return config.b2_mapping.ordinary_min_source_block_coverage


# ════════════════════════════════════════════
# PROPOSED MAPPING (source-block match)
# ════════════════════════════════════════════


def map_proposed_item(
    qa: Dict[str, Any],
    proposed_chunks: List[Dict[str, Any]],
    config: cfg.QAConfig,
) -> Dict[str, Any]:
    """Map one QA item onto Proposed chunks by source-block containment."""
    threshold = config.proposed_mapping.minimum_source_block_coverage
    required_blocks = list(qa.get("gold_source_block_ids") or [])
    by_block: Dict[str, List[Dict[str, Any]]] = {}
    all_relevant: List[str] = []
    warnings: List[str] = []

    for block_id in required_blocks:
        hits: List[Dict[str, Any]] = []
        for chunk in proposed_chunks:
            if block_id not in (chunk.get("source_block_ids") or []):
                continue
            coverage = _coverage_for_block(chunk, block_id)
            # If coverage is present it must meet the threshold; if absent, a
            # containment match is accepted (pedagogical chunks consume blocks whole).
            is_relevant = coverage is None or coverage >= threshold
            if is_relevant:
                hits.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "coverage": coverage if coverage is not None else 1.0,
                        "is_relevant": True,
                    }
                )
        by_block[block_id] = hits
        for hit in hits:
            if hit["chunk_id"] not in all_relevant:
                all_relevant.append(hit["chunk_id"])
        if not hits:
            warnings.append(f"no Proposed chunk found for gold block {block_id}")

    status = "complete" if all(by_block.get(b) for b in required_blocks) and required_blocks else "incomplete"
    return {
        "qa_id": qa["qa_id"],
        "system": "proposed",
        "mapping_method": "source_block_match",
        "required_gold_source_block_ids": required_blocks,
        "relevant_chunks_by_gold_block": by_block,
        "all_relevant_chunk_ids": all_relevant,
        "mapping_status": status,
        "warnings": warnings,
    }


# ════════════════════════════════════════════
# B2 MAPPING (source-block coverage)
# ════════════════════════════════════════════


def map_b2_item(
    qa: Dict[str, Any],
    b2_chunks: List[Dict[str, Any]],
    config: cfg.QAConfig,
) -> Dict[str, Any]:
    """Map one QA item onto B2 fixed windows by source-block coverage threshold."""
    threshold = _b2_threshold(qa.get("question_type"), config)
    required_blocks = list(qa.get("gold_source_block_ids") or [])
    is_diagram = qa.get("question_type") == "diagram_dependent"
    by_block: Dict[str, List[Dict[str, Any]]] = {}
    all_relevant: List[str] = []
    warnings: List[str] = []

    for block_id in required_blocks:
        hits: List[Dict[str, Any]] = []
        for chunk in b2_chunks:
            if block_id not in (chunk.get("source_block_ids") or []):
                continue
            coverage = _coverage_for_block(chunk, block_id)
            if coverage is None:
                # Block is listed but has no coverage figure — cannot confirm the
                # threshold, so flag rather than assert relevance.
                warnings.append(
                    f"B2 chunk {chunk.get('chunk_id')} lists block {block_id} without coverage"
                )
                continue
            is_relevant = coverage >= threshold
            entry = {
                "chunk_id": chunk.get("chunk_id"),
                "coverage": coverage,
                "threshold": threshold,
                "is_relevant": is_relevant,
            }
            if is_diagram:
                entry["has_diagram"] = bool(chunk.get("has_diagram"))
                entry["has_diagram_image_urls"] = bool(chunk.get("diagram_image_urls"))
                if not chunk.get("diagram_image_urls"):
                    warnings.append(
                        f"B2 chunk {chunk.get('chunk_id')} has only page-level diagram "
                        "references; exact diagram provenance is not guaranteed."
                    )
            if is_relevant:
                hits.append(entry)
        by_block[block_id] = hits
        for hit in hits:
            if hit["chunk_id"] not in all_relevant:
                all_relevant.append(hit["chunk_id"])
        if not hits:
            warnings.append(f"no B2 chunk met coverage {threshold} for gold block {block_id}")

    status = "complete" if all(by_block.get(b) for b in required_blocks) and required_blocks else "incomplete"
    return {
        "qa_id": qa["qa_id"],
        "system": "b2",
        "mapping_method": "source_block_coverage",
        "required_gold_source_block_ids": required_blocks,
        "relevant_chunks_by_gold_block": by_block,
        "all_relevant_chunk_ids": all_relevant,
        "mapping_status": status,
        "warnings": warnings,
    }


# ════════════════════════════════════════════
# B1 MAPPING (page-overlap proxy)
# ════════════════════════════════════════════


def map_b1_item(
    qa: Dict[str, Any],
    b1_chunks: List[Dict[str, Any]],
    _config: cfg.QAConfig,
) -> Dict[str, Any]:
    """Map one QA item onto B1 OCR windows by inclusive page-range overlap."""
    required_pages = [p for p in (qa.get("gold_page_numbers") or []) if isinstance(p, int)]
    by_page: Dict[str, List[Dict[str, Any]]] = {}
    all_relevant: List[str] = []
    warnings: List[str] = [_B1_WARNING]

    for page in required_pages:
        hits: List[Dict[str, Any]] = []
        for chunk in b1_chunks:
            lo, hi = _page_range(chunk)
            if lo <= page <= hi:
                hits.append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "page_range": [lo, hi],
                        "is_relevant": True,
                    }
                )
        by_page[str(page)] = hits
        for hit in hits:
            if hit["chunk_id"] not in all_relevant:
                all_relevant.append(hit["chunk_id"])
        if not hits:
            warnings.append(f"no B1 chunk overlaps gold page {page}")

    status = "proxy_page_level" if required_pages and all(by_page.get(str(p)) for p in required_pages) else "incomplete"
    return {
        "qa_id": qa["qa_id"],
        "system": "b1",
        "mapping_method": "page_overlap_proxy",
        "required_gold_page_numbers": required_pages,
        "required_gold_source_block_ids": list(qa.get("gold_source_block_ids") or []),
        "relevant_chunks_by_gold_page": by_page,
        "all_relevant_chunk_ids": all_relevant,
        "mapping_status": status,
        "warnings": warnings,
    }


# ════════════════════════════════════════════
# STAGE ORCHESTRATION + CHECKS
# ════════════════════════════════════════════

_UNMAPPED_COLUMNS = (
    "qa_id",
    "system",
    "question_type",
    "severity",
    "detail",
)


def _check_unmapped(qa: Dict[str, Any], mapping: Dict[str, Any], system: str) -> List[Dict[str, Any]]:
    """Return unmapped rows: a gold block/page with no relevant chunk."""
    rows: List[Dict[str, Any]] = []
    if system == "b1":
        for page, hits in (mapping.get("relevant_chunks_by_gold_page") or {}).items():
            if not hits:
                rows.append(
                    {
                        "qa_id": qa["qa_id"],
                        "system": system,
                        "question_type": qa.get("question_type"),
                        "severity": "high",
                        "detail": f"no B1 chunk overlaps gold page {page}",
                    }
                )
    else:
        for block, hits in (mapping.get("relevant_chunks_by_gold_block") or {}).items():
            if not hits:
                rows.append(
                    {
                        "qa_id": qa["qa_id"],
                        "system": system,
                        "question_type": qa.get("question_type"),
                        "severity": "high",
                        "detail": f"no {system} chunk relevant for gold block {block}",
                    }
                )
    return rows


def run_gold_mapping(
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute Stage 6, writing the three mapping files + summary + unmapped CSV."""
    layout.ensure()
    qa_items = schemas.read_jsonl(layout.qa_dataset_jsonl)
    if not qa_items:
        raise FileNotFoundError(
            f"No final dataset at {layout.qa_dataset_jsonl}. Run 'finalize' first."
        )

    proposed_chunks = schemas.load_chunks(config.proposed_chunks_path)
    b2_chunks = schemas.load_chunks(config.b2_chunks_path)
    b1_chunks = schemas.load_chunks(config.b1_chunks_path)

    proposed_maps: List[Dict[str, Any]] = []
    b2_maps: List[Dict[str, Any]] = []
    b1_maps: List[Dict[str, Any]] = []
    unmapped_rows: List[Dict[str, Any]] = []

    for qa in qa_items:
        pmap = map_proposed_item(qa, proposed_chunks, config)
        bmap2 = map_b2_item(qa, b2_chunks, config)
        bmap1 = map_b1_item(qa, b1_chunks, config)
        proposed_maps.append(pmap)
        b2_maps.append(bmap2)
        b1_maps.append(bmap1)
        unmapped_rows.extend(_check_unmapped(qa, pmap, "proposed"))
        unmapped_rows.extend(_check_unmapped(qa, bmap2, "b2"))
        unmapped_rows.extend(_check_unmapped(qa, bmap1, "b1"))

    def _complete(maps: List[Dict[str, Any]], ok_statuses: set) -> int:
        return sum(1 for m in maps if m["mapping_status"] in ok_statuses)

    summary = {
        "total_qa_items": len(qa_items),
        "proposed_complete": _complete(proposed_maps, {"complete"}),
        "b2_complete": _complete(b2_maps, {"complete"}),
        "b1_page_level": _complete(b1_maps, {"proxy_page_level"}),
        "unmapped_rows": len(unmapped_rows),
    }

    if dry_run:
        logger.info("[dry-run] mapping summary: %s", summary)
        return summary

    schemas.write_jsonl(layout.gold_mapping_proposed, proposed_maps)
    schemas.write_jsonl(layout.gold_mapping_b2, b2_maps)
    schemas.write_jsonl(layout.gold_mapping_b1, b1_maps)
    schemas.write_csv(layout.unmapped_qa_items, unmapped_rows, _UNMAPPED_COLUMNS)

    md = [
        "# Gold Mapping Summary",
        "",
        f"Total QA items: **{summary['total_qa_items']}**",
        "",
        "| System | Method | Fully mapped |",
        "|---|---|---:|",
        f"| Proposed | source_block_match | {summary['proposed_complete']} |",
        f"| B2 | source_block_coverage | {summary['b2_complete']} |",
        f"| B1 | page_overlap_proxy | {summary['b1_page_level']} |",
        "",
        f"Unmapped (block/page with no relevant chunk) rows: **{summary['unmapped_rows']}** "
        f"(see `{layout.unmapped_qa_items.name}`).",
        "",
        "> B1 mapping is a page-overlap proxy: its metric is Gold Page Recall, "
        "not Gold Unit Recall, because OCR chunks retain no source-block provenance.",
        "",
    ]
    layout.gold_mapping_summary_md.write_text("\n".join(md), encoding="utf-8")

    logger.info(
        "Stage 6 complete: proposed=%d/%d b2=%d/%d b1=%d/%d unmapped_rows=%d",
        summary["proposed_complete"], len(qa_items),
        summary["b2_complete"], len(qa_items),
        summary["b1_page_level"], len(qa_items),
        summary["unmapped_rows"],
    )
    return summary
