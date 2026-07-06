"""
ragkit.chunk.fixed
══════════════════

Fixed-window (512-token + overlap) chunking — used by baselines B1 and B2, and
the textbook RAG default.

The mechanics are identical for both baselines; only the INPUT differs:
    * B1 slides the window over a page-tagged stream of raw OCR tokens.
    * B2 slides the same window over the proposed structured representation.

So the chunker is a single ``fixed_token_chunks`` that operates on a page-tagged
``(token, page_number)`` stream. ``build_ocr_stream`` flattens B1's per-page OCR
text into that stream; B2 gets its stream from ``represent.build_representation_stream``.

Tokens are approximated by whitespace-separated words — a deliberately simple,
reproducible definition appropriate for these baselines. Because every token
carries its source page, each emitted chunk records the exact page range it
covers without using any document structure.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..config import FixedChunkConfig


def build_ocr_stream(page_texts: Dict[int, str]) -> List[Tuple[str, int]]:
    """
    Flatten per-page OCR text into a single page-tagged token stream spanning
    the whole document in page order. Each element is (token, page_number).
    """
    stream: List[Tuple[str, int]] = []
    for page_number in sorted(page_texts.keys()):
        for token in page_texts[page_number].split():
            stream.append((token, page_number))
    return stream


def fixed_token_chunks(
    stream: List[Tuple],
    config: FixedChunkConfig,
    source_block_total_units: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Slide a fixed ``chunk_size`` token window across the tagged stream,
    advancing by (chunk_size - overlap) tokens each step so consecutive chunks
    share ``overlap`` tokens of context.

    The stream elements may be either:
      * ``(token, page_number)``                    — B1 (raw OCR), or
      * ``(token, page_number, source_block_id)``   — B2 (representation stream).

    Every chunk records page-level provenance (``source_page_numbers``,
    ``source_page_unit_counts``, ``whitespace_unit_count``). When the stream
    additionally carries ``source_block_id`` (B2), each chunk also records
    block-level provenance:

      * ``source_block_ids``        — ordered unique blocks contributing ≥1 unit.
      * ``source_block_unit_counts``— units from each block inside this window.
      * ``source_block_coverage``   — units-in-window / total-units-in-stream for
                                       each block, clamped to 1.0 (requires
                                       ``source_block_total_units``).
    """
    chunk_size = config.chunk_size_tokens
    overlap = config.overlap_tokens

    chunks: List[Dict[str, Any]] = []
    if not stream:
        return chunks

    # Detect whether the stream carries source_block_id (3-tuple) or not.
    has_blocks = len(stream[0]) >= 3
    totals = source_block_total_units or {}

    # The stride is how far the window advances; overlap < chunk_size guarantees
    # forward progress.
    step = max(1, chunk_size - overlap)

    chunk_ordinal = 0
    for start in range(0, len(stream), step):
        window = stream[start : start + chunk_size]
        if not window:
            break

        tokens = [item[0] for item in window]
        pages = [item[1] for item in window]
        text = " ".join(tokens).strip()

        # Skip windows that ended up empty after stripping.
        if not text:
            # If we've reached the tail and produced nothing, stop.
            if start + chunk_size >= len(stream):
                break
            continue

        # Page-level provenance (computed for every stream shape).
        source_page_numbers: List[int] = []
        source_page_unit_counts: Dict[str, int] = {}
        for p in pages:
            pk = str(p)
            source_page_unit_counts[pk] = source_page_unit_counts.get(pk, 0) + 1
            if p not in source_page_numbers:
                source_page_numbers.append(p)

        chunk: Dict[str, Any] = {
            "ordinal": chunk_ordinal,
            "text": text,
            "page_range": [min(pages), max(pages)],
            "whitespace_unit_count": len(tokens),
            "source_page_numbers": source_page_numbers,
            "source_page_unit_counts": source_page_unit_counts,
        }

        # Block-level provenance — only when the stream carries source_block_id.
        if has_blocks:
            block_ids = [item[2] for item in window]
            source_block_ids: List[str] = []
            source_block_unit_counts: Dict[str, int] = {}
            for b in block_ids:
                source_block_unit_counts[b] = source_block_unit_counts.get(b, 0) + 1
                if b not in source_block_ids:
                    source_block_ids.append(b)

            source_block_coverage: Dict[str, float] = {}
            for b, count in source_block_unit_counts.items():
                total = totals.get(b, 0)
                coverage = (count / total) if total else 0.0
                # Clamp to 1.0 to guard against rounding / double-counted units.
                source_block_coverage[b] = min(1.0, coverage)

            chunk["source_block_ids"] = source_block_ids
            chunk["source_block_unit_counts"] = source_block_unit_counts
            chunk["source_block_coverage"] = source_block_coverage

        chunks.append(chunk)
        chunk_ordinal += 1

        # The last window has been emitted once it reaches the end of the stream.
        if start + chunk_size >= len(stream):
            break

    return chunks


def make_ocr_records(
    raw_chunks: List[Dict[str, Any]],
    semester: int,
    chunk_id_prefix: str = "b1",
    content_type: str = "ocr_text",
) -> List[Dict[str, Any]]:
    """
    Convert raw fixed-window OCR chunks into the chunk dict shape that
    ``index.upsert_to_pinecone`` / ``index.sanitize_metadata`` expect.

    OCR baselines have no structured metadata (no unit/lesson/diagram/math
    information), so those fields are intentionally left empty/zero. The
    metadata writer tolerates these defaults.
    """
    records: List[Dict[str, Any]] = []
    for ch in raw_chunks:
        first_page = int(ch["page_range"][0])
        records.append({
            # Unique, human-readable id for this baseline and chunk.
            "chunk_id": f"{chunk_id_prefix}-s{semester}-p{first_page}-c{ch['ordinal']}",
            "page_number": first_page,
            "page_range": ch["page_range"],
            # B1 keeps PAGE-level provenance only: raw OCR has no reliable
            # pedagogical source blocks to point at.
            "source_page_numbers": ch.get("source_page_numbers", []),
            "source_page_unit_counts": ch.get("source_page_unit_counts", {}),
            "whitespace_unit_count": ch.get("whitespace_unit_count", 0),
            # Flat, structureless content type — this is raw OCR text.
            "content_type": content_type,
            # No structured metadata is available from plain OCR.
            "unit_number": None,
            "unit_title_ar": "",
            "lesson_number": "",
            "lesson_title_ar": "",
            "lesson_title_en": "",
            "heading_ar": "",
            "problem_numbers": [],
            "math_expressions": [],
            "diagrams": [],
            "diagram_image_urls": [],
            "standards": [],
            "has_diagram": False,
            "has_math": False,
            # The text that will be embedded == the raw OCR window itself.
            "main_text_ar": ch["text"],
            # OCR confidence is unknown here; mark as the baseline default.
            "extraction_confidence": 1.0,
        })
    return records


def make_stream_records(
    raw_chunks: List[Dict[str, Any]],
    page_meta: Dict[int, Dict[str, Any]],
    block_meta: Dict[str, Dict[str, Any]],
    semester: int,
    chunk_id_prefix: str = "b3",
    content_type: str = "fixed_window",
) -> List[Dict[str, Any]]:
    """
    Convert fixed-window chunks built over the proposed representation stream
    into the dict shape the upsert stage expects, attaching the best-available
    structured metadata.

    Metadata is resolved as:
      - unit/lesson identity → taken from the chunk's FIRST page.
      - diagram urls / standards / has_diagram / has_math → aggregated ONLY from
        the source blocks that actually contributed text to this window (via
        ``block_meta``), NOT from every page the window happens to span. This
        avoids a window inheriting a diagram from a page when only one sentence
        from that page appears in the window.

    Each record also carries the window's full provenance (source block IDs,
    per-block unit counts and coverage, source pages, whitespace unit count).
    """
    records: List[Dict[str, Any]] = []
    for ch in raw_chunks:
        first_page = int(ch["page_range"][0])

        # Identity metadata comes from the first page in the window.
        base = page_meta.get(first_page, {})

        # Aggregate list/boolean metadata from ONLY the contributing source blocks.
        source_block_ids: List[str] = ch.get("source_block_ids", [])
        diagram_urls: List[str] = []
        standards: List[str] = []
        has_diagram = False
        has_math = False
        for sbid in source_block_ids:
            bm = block_meta.get(sbid)
            if not bm:
                continue
            has_diagram = has_diagram or bm.get("has_diagram", False)
            has_math = has_math or bm.get("has_math", False)
            for url in bm.get("diagram_image_urls") or []:
                if url not in diagram_urls:
                    diagram_urls.append(url)
            for std in bm.get("standards") or []:
                if std not in standards:
                    standards.append(std)

        records.append({
            "chunk_id": f"{chunk_id_prefix}-s{semester}-p{first_page}-c{ch['ordinal']}",
            "page_number": first_page,
            "page_range": ch["page_range"],
            # Fixed windows have no single pedagogical type — label them plainly.
            "content_type": content_type,
            "unit_number": base.get("unit_number"),
            "unit_title_ar": base.get("unit_title_ar", ""),
            "lesson_number": base.get("lesson_number", ""),
            "lesson_title_ar": base.get("lesson_title_ar", ""),
            "lesson_title_en": base.get("lesson_title_en", ""),
            "heading_ar": "",
            "problem_numbers": [],
            "math_expressions": [],
            "diagrams": [],
            "diagram_image_urls": diagram_urls,
            "standards": standards,
            "has_diagram": has_diagram,
            "has_math": has_math,
            "main_text_ar": ch["text"],
            "extraction_confidence": 1.0,
            # Provenance — which original Gemini blocks this window covers.
            "source_block_ids": source_block_ids,
            "source_block_unit_counts": ch.get("source_block_unit_counts", {}),
            "source_block_coverage": ch.get("source_block_coverage", {}),
            "source_page_numbers": ch.get("source_page_numbers", []),
            "whitespace_unit_count": ch.get("whitespace_unit_count", 0),
        })
    return records
