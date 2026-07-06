"""
ragkit.chunk.pedagogical
═════════════════════════

Stage 2 (proposed) — boundary-aware pedagogical chunking with block merge.

This is the proposed system's chunker. Instead of slicing text mechanically, it
treats each extracted content block (an example, theorem, exercise, …) as a
pedagogical unit and merges blocks that a single instructional unit spans across
pages. The result respects lesson boundaries and keeps worked solutions intact.

``new_chunk_from_block`` is intentionally public: baseline B2 reuses it (via
``ragkit.represent.build_representation_stream``) to build the proposed-style
representation for a single block, so extraction+representation stay identical
while only the chunker differs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ..cache import append_log


# ════════════════════════════════════════════
# NAMED-ELEMENT HELPERS
# ════════════════════════════════════════════


def _empty_named_elements() -> Dict[str, List[Any]]:
    """Return a fresh, empty named-elements container."""
    return {"theorems": [], "definitions": [], "vocabulary": [], "standards": []}


def _merge_named_elements(dst: Dict[str, List[Any]], src: Dict[str, Any]) -> None:
    """
    Merge named elements from a source block into a destination accumulator,
    deduplicating each list in the process.
    """
    for k in ("theorems", "definitions", "standards"):
        for item in (src.get(k) or []):
            if item and item not in dst[k]:
                dst[k].append(item)
    for v in (src.get("vocabulary") or []):
        if v and v not in dst["vocabulary"]:
            dst["vocabulary"].append(v)


def _attach_diagram_urls(
    chunk: Dict[str, Any],
    block_diagrams: List[Dict[str, Any]],
    page_url_manifest: List[Dict[str, Any]],
) -> None:
    """
    Look up each diagram's Cloudinary URL from the page URL manifest and
    append it to the chunk's diagram_image_urls list (no duplicates).
    """
    by_idx = {d["diagram_index"]: d for d in page_url_manifest}
    for diag in block_diagrams:
        d_idx = diag.get("diagram_index")
        if d_idx is None:
            continue
        if d_idx in by_idx:
            url = by_idx[d_idx]["cloudinary_url"]
            if url not in chunk["diagram_image_urls"]:
                chunk["diagram_image_urls"].append(url)


def _make_chunk_id(
    semester: int,
    unit_number: Optional[int],
    lesson_number: Optional[str],
    content_type: str,
    first_page: int,
    block_index: int,
) -> str:
    """
    Build a human-readable, unique chunk ID.
    Format: s<semester>-u<unit>-l<lesson>-<type>-p<page>-b<block>
    Example: s2-u5-l5-1-example-p22-b3
    """
    u = str(unit_number) if unit_number is not None else "X"
    l = (lesson_number or "X").replace(".", "-")
    return f"s{semester}-u{u}-l{l}-{content_type}-p{first_page}-b{block_index}"


def make_source_block_id(
    semester: int,
    unit_number: Optional[int],
    lesson_number: Optional[str],
    page_number: int,
    block_index: int,
) -> str:
    """
    Build a STABLE, deterministic identifier for an original Gemini-extracted
    content block, used for retrieval-evaluation provenance.

    Format: s<semester>-u<unit>-l<lesson>-p<page>-b<block>
    Example: s2-u4-l4-1-p15-b3

    Unlike ``_make_chunk_id`` this deliberately omits ``content_type`` so the
    ID is tied to the block's position in the document, not to how it was
    later classified. If unit or lesson metadata is absent a stable "X"
    fallback is used rather than failing.
    """
    u = str(unit_number) if unit_number is not None else "X"
    l = (lesson_number or "X").replace(".", "-")
    return f"s{semester}-u{u}-l{l}-p{page_number}-b{block_index}"


def new_chunk_from_block(
    page: Dict[str, Any],
    block: Dict[str, Any],
    semester: int,
    page_url_manifest: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create a new chunk dict seeded with data from a single content block.

    This is called when a block has no open carry-over chunk to merge into.
    (Public because baseline B2 reuses it to render the proposed representation
    for an individual block.)
    """
    page_number = int(page["page_number"])
    block_index = int(block.get("block_index", 0))
    content_type = block.get("content_type") or "other"

    # Stable provenance ID for the ORIGINAL extracted block this chunk seeds from.
    source_block_id = make_source_block_id(
        semester,
        page.get("unit_number"),
        page.get("lesson_number"),
        page_number,
        block_index,
    )

    chunk: Dict[str, Any] = {
        "chunk_id": _make_chunk_id(
            semester,
            page.get("unit_number"),
            page.get("lesson_number"),
            content_type,
            page_number,
            block_index,
        ),
        # Provenance: which original Gemini block(s) contributed to this chunk.
        # A pedagogical chunk consumes each contributing block in full, so its
        # coverage of every source block is 1.0.
        "source_block_id": source_block_id,
        "source_block_ids": [source_block_id],
        "source_page_numbers": [page_number],
        "source_block_coverage": {source_block_id: 1.0},
        "page_number": page_number,
        # page_range tracks the first and last page spanned by this chunk
        "page_range": [page_number, page_number],
        "content_type": content_type,
        "unit_number": page.get("unit_number"),
        "unit_title_ar": page.get("unit_title_ar"),
        "lesson_number": page.get("lesson_number"),
        "lesson_title_ar": page.get("lesson_title_ar"),
        "lesson_title_en": page.get("lesson_title_en"),
        "standard": page.get("standard"),
        "heading_ar": block.get("heading_ar"),
        "problem_numbers": list(block.get("problem_numbers") or []),
        "main_text_ar": block.get("text_ar") or "",
        "math_expressions": list(block.get("math_expressions") or []),
        "diagrams": list(block.get("diagrams") or []),
        "diagram_image_urls": [],
        "named_elements": _empty_named_elements(),
        "has_diagram": bool(block.get("diagrams")),
        "has_math": bool(block.get("math_expressions")),
        "standards": [],
        "is_other": content_type == "other",
        "extraction_confidence": 1.0,
    }

    _merge_named_elements(chunk["named_elements"], block.get("named_elements") or {})
    _attach_diagram_urls(chunk, chunk["diagrams"], page_url_manifest)
    return chunk


def _merge_block_into_chunk(
    chunk: Dict[str, Any],
    page: Dict[str, Any],
    block: Dict[str, Any],
    page_url_manifest: List[Dict[str, Any]],
    semester: int,
) -> None:
    """
    Append a continuation block's content into an existing open chunk.

    Used when a block is flagged continued_from_prev_page — its content
    is joined to the chunk started on the preceding page.
    """
    page_number = int(page["page_number"])
    block_index = int(block.get("block_index", 0))
    txt = block.get("text_ar") or ""

    # Record the continuation block's provenance. The continuation is consumed
    # in full, so its coverage of this source block is 1.0.
    cont_source_block_id = make_source_block_id(
        semester,
        page.get("unit_number"),
        page.get("lesson_number"),
        page_number,
        block_index,
    )
    if cont_source_block_id not in chunk["source_block_ids"]:
        chunk["source_block_ids"].append(cont_source_block_id)
        chunk["source_block_coverage"][cont_source_block_id] = 1.0
    if page_number not in chunk["source_page_numbers"]:
        chunk["source_page_numbers"].append(page_number)

    # Append text with a paragraph separator
    if txt:
        chunk["main_text_ar"] = (chunk["main_text_ar"] + "\n\n" + txt).strip()

    # Accumulate math expressions, deduplicating
    for expr in block.get("math_expressions") or []:
        if expr not in chunk["math_expressions"]:
            chunk["math_expressions"].append(expr)

    # Accumulate diagrams, deduplicating
    for diag in block.get("diagrams") or []:
        if diag not in chunk["diagrams"]:
            chunk["diagrams"].append(diag)

    _merge_named_elements(chunk["named_elements"], block.get("named_elements") or {})

    # Extend the page range to include this continuation page
    chunk["page_range"][1] = page_number
    chunk["has_diagram"] = chunk["has_diagram"] or bool(block.get("diagrams"))
    chunk["has_math"] = chunk["has_math"] or bool(block.get("math_expressions"))
    _attach_diagram_urls(chunk, block.get("diagrams") or [], page_url_manifest)


def _finalize_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the extraction_confidence score for a chunk and flatten standards.

    Confidence is reduced when key metadata fields are absent or the text is
    very short, giving downstream consumers a signal about data quality.
    """
    # Flatten the standards list from named_elements to the top level
    chunk["standards"] = list(chunk["named_elements"]["standards"])

    score = 1.0
    if not chunk.get("lesson_number"):
        score -= 0.2
    if chunk.get("unit_number") is None:
        score -= 0.2
    if not chunk.get("main_text_ar"):
        score -= 0.4
    elif len(chunk["main_text_ar"]) < 50:
        score -= 0.2
    chunk["extraction_confidence"] = max(0.0, score)
    return chunk


# ══════════════════════════════════════════════
# STAGE 2 — PEDAGOGICAL CHUNKING WITH BLOCK MERGE
# ══════════════════════════════════════════════


def build_chunks(
    extractions_dir: Path,
    diagram_urls_dir: Path,
    cache_dir: Path,
    semester: int,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert per-page extraction JSON files into a flat list of pedagogical chunks.

    The core challenge here is that a single instructional block (e.g. a worked
    example) can span multiple pages. We track "open" chunks keyed by
    (lesson_number, content_type, heading_ar). When a block is flagged as
    continues_to_next_page its chunk stays open; when the continuation block
    arrives it is merged in. Any still-open chunks at the end are closed.

    The resulting list is written to cache_dir/chunks.json and returned.
    """
    from ..render import page_in_range

    low_confidence_log = cache_dir / "low_confidence.log"

    # Load all extraction JSON files for pages in range
    page_files = sorted(extractions_dir.glob("page_*.json"))
    pages: List[Dict[str, Any]] = []
    for pf in page_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                pn = int(data.get("page_number", 0))
                if page_in_range(pn, start_page, end_page):
                    pages.append(data)
        except Exception:
            continue

    pages.sort(key=lambda p: int(p.get("page_number", 0)))

    # open_chunks: chunks that have continues_to_next_page=True and await continuation
    # last_carry: maps lesson_number → merge_key for fuzzy fallback when keys drift
    open_chunks: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    last_carry: Dict[Any, Tuple[Any, Any, Any]] = {}
    closed_chunks: List[Dict[str, Any]] = []

    for page in tqdm(pages, desc="Stage 2 — chunking"):
        page_number = int(page.get("page_number", 0))
        key = f"{page_number:04d}"

        # Load the Cloudinary URL manifest for this page (may be empty)
        manifest_path = diagram_urls_dir / f"page_{key}_urls.json"
        page_url_manifest: List[Dict[str, Any]] = []
        if manifest_path.exists():
            try:
                page_url_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                page_url_manifest = []

        blocks = page.get("content_blocks") or []
        blocks = sorted(blocks, key=lambda b: int(b.get("block_index", 0)))

        for block in blocks:
            content_type = block.get("content_type") or "other"
            heading_ar = block.get("heading_ar")
            lesson_number = page.get("lesson_number")

            # The merge key uniquely identifies a logical content block across pages
            merge_key = (lesson_number, content_type, heading_ar)

            continued = bool(block.get("continued_from_prev_page"))
            continues = bool(block.get("continues_to_next_page"))

            if continued:
                # This block continues from a previous page — find the open chunk
                if merge_key not in open_chunks:
                    # Fallback: try matching by lesson_number alone when the key drifted
                    fallback = last_carry.get(lesson_number)
                    if fallback and fallback in open_chunks:
                        logging.info(
                            "page %d block %s: key mismatch, remapping %s -> %s",
                            page_number, block.get("block_index"), merge_key, fallback,
                        )
                        merge_key = fallback
                    else:
                        # No open chunk found — create an orphan chunk and move on
                        logging.warning(
                            "page %d block %s: no open chunk and no carry for lesson=%s; "
                            "creating orphan chunk",
                            page_number, block.get("block_index"), lesson_number,
                        )
                        chunk = new_chunk_from_block(page, block, semester, page_url_manifest)
                        if continues:
                            open_chunks[merge_key] = chunk
                            last_carry[lesson_number] = merge_key
                        else:
                            closed_chunks.append(chunk)
                        continue

                # Merge this continuation block into the existing open chunk
                chunk = open_chunks[merge_key]
                _merge_block_into_chunk(chunk, page, block, page_url_manifest, semester)
                if not continues:
                    # Block is fully closed — move it from open to closed
                    closed_chunks.append(open_chunks.pop(merge_key))
                    last_carry.pop(lesson_number, None)
            else:
                # Fresh block — close any stale open chunk with the same key
                if merge_key in open_chunks:
                    closed_chunks.append(open_chunks.pop(merge_key))

                chunk = new_chunk_from_block(page, block, semester, page_url_manifest)
                if continues:
                    # Keep open for continuation on the next page
                    open_chunks[merge_key] = chunk
                    last_carry[lesson_number] = merge_key
                else:
                    closed_chunks.append(chunk)

    # Close any chunks that never received a continuation (e.g. pipeline ended mid-book)
    for chunk in open_chunks.values():
        closed_chunks.append(chunk)
    open_chunks.clear()

    # Compute confidence scores and flatten standards
    finalized = [_finalize_chunk(c) for c in closed_chunks]

    # Log any chunks below the confidence threshold for review
    for c in finalized:
        if c["extraction_confidence"] < 0.6:
            append_log(
                low_confidence_log,
                f"{c['chunk_id']} confidence={c['extraction_confidence']:.2f}",
            )

    # Write the full chunk list to disk for inspection
    chunks_path = cache_dir / "chunks.json"
    chunks_path.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return finalized
