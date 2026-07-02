"""
ragkit.represent
════════════════

Stage 3A — turn a chunk into the text string that will be embedded.

``build_embedding_text`` is the proposed system's representation. Baseline B2
reuses it verbatim (via ``build_representation_stream``) so that B2 embeds
byte-for-byte the same representation strings as the proposed system — the only
thing B2 changes is the chunker. Holding representation constant this way means
any B2-vs-proposed difference is attributable purely to chunking strategy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .cache import append_log
from .config import RepresentationConfig


# ════════════════════════════════════════════
# STAGE 3A — EMBEDDING TEXT CONSTRUCTION
# ════════════════════════════════════════════


def build_embedding_text(
    chunk: Dict[str, Any],
    cache_dir: Path,
    config: RepresentationConfig,
) -> str:
    """
    Build the text string that will be embedded for a chunk.

    The string concatenates (in order):
      - Unit title
      - Lesson title (Arabic and English)
      - Block heading
      - Main Arabic text
      - LaTeX math expressions (prefixed with "المعادلة:")
      - Diagram descriptions with their visible labels

    If the resulting string exceeds the embedding model's token limit
    (estimated at ``config.chars_per_token`` chars/token), it is truncated and
    the chunk ID is appended to a truncated_chunks.log for review.
    """
    lines: List[str] = []

    if chunk.get("unit_title_ar"):
        lines.append(f"الوحدة: {chunk['unit_title_ar']}")
    if chunk.get("lesson_title_ar"):
        lines.append(f"الدرس: {chunk['lesson_title_ar']}")
    if chunk.get("lesson_title_en"):
        lines.append(f"Lesson: {chunk['lesson_title_en']}")
    if chunk.get("heading_ar"):
        lines.append(chunk["heading_ar"])
    if chunk.get("main_text_ar"):
        lines.append(chunk["main_text_ar"])

    # Add each LaTeX expression as a separate line
    for expr in chunk.get("math_expressions") or []:
        if expr:
            lines.append(f"المعادلة: {expr}")

    # Add diagram descriptions with their visible label lists
    for diag in chunk.get("diagrams") or []:
        desc = diag.get("description") or ""
        labels = diag.get("labels") or []
        labels_str = "، ".join(str(l) for l in labels)
        lines.append(f"[شكل هندسي: {desc}. التسميات: {labels_str}]")

    text = "\n".join(l for l in lines if l)

    # Truncate if the estimated token count exceeds the model limit
    if len(text) / config.chars_per_token > config.max_tokens:
        max_chars = int(config.max_tokens * config.chars_per_token)
        truncated = text[:max_chars].rstrip() + "…"
        append_log(cache_dir / "truncated_chunks.log", chunk["chunk_id"])
        return truncated

    return text


def write_embedding_text_cache(
    cache_dir: Path,
    chunks: List[Dict[str, Any]],
    texts: List[str],
) -> Path:
    """
    Write embedding texts to a JSONL file for offline inspection and debugging.

    Each line is a JSON object with chunk_id, content_type, page_range, and text.
    """
    out_path = cache_dir / "embedding_texts.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk, text in zip(chunks, texts):
            record = {
                "chunk_id": chunk.get("chunk_id"),
                "content_type": chunk.get("content_type"),
                "page_range": chunk.get("page_range"),
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


# ════════════════════════════════════════════
# B3 — PROPOSED REPRESENTATION AS A FLAT STREAM
# ════════════════════════════════════════════


def build_representation_stream(
    extractions_dir: Path,
    diagram_urls_dir: Path,
    cache_dir: Path,
    semester: int,
    config: RepresentationConfig,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> Tuple[
    List[Tuple[str, int, str]],
    Dict[int, Dict[str, Any]],
    Dict[str, int],
    Dict[str, Dict[str, Any]],
]:
    """
    Render every content block into proposed-style embedding text and flatten
    the results into one page-tagged token stream.

    This reuses ``new_chunk_from_block()`` + ``build_embedding_text()`` so each
    block's representation is IDENTICAL to what the proposed pipeline would
    embed — the difference is purely that here we do NOT merge blocks into
    pedagogical chunks; we just concatenate them in reading order.

    Each emitted token additionally carries the stable ``source_block_id`` of
    the ORIGINAL Gemini block that produced it, so fixed windows downstream can
    report exactly which pedagogical blocks they cover (and how completely).

    Returns:
      stream                   — list of (token, page_number, source_block_id)
                                 covering the whole document.
      page_meta                — page_number -> aggregated metadata for that page
                                 (unit/lesson titles, diagram urls, has_math, …).
      source_block_total_units — source_block_id -> total whitespace units that
                                 block contributed to the COMPLETE stream. Needed
                                 to compute per-window coverage fractions.
      block_meta               — source_block_id -> that block's own metadata
                                 (diagram urls, standards, has_diagram, has_math,
                                 identity), so a fixed window can aggregate
                                 metadata from ONLY the blocks it actually covers
                                 instead of inheriting everything on a page.
    """
    from .render import page_in_range
    from .chunk.pedagogical import new_chunk_from_block

    # Load all extraction JSON files for pages in range, sorted by page number.
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

    stream: List[Tuple[str, int, str]] = []
    page_meta: Dict[int, Dict[str, Any]] = {}
    # source_block_id -> total whitespace units contributed across the whole stream.
    source_block_total_units: Dict[str, int] = {}
    # source_block_id -> that block's own metadata (used for per-window aggregation).
    block_meta: Dict[str, Dict[str, Any]] = {}

    for page in tqdm(pages, desc="Stage B3.A — build representation"):
        page_number = int(page.get("page_number", 0))
        key = f"{page_number:04d}"

        # Load this page's Cloudinary URL manifest (same source the proposed
        # pipeline uses to attach diagram image links to the representation).
        manifest_path = diagram_urls_dir / f"page_{key}_urls.json"
        page_url_manifest: List[Dict[str, Any]] = []
        if manifest_path.exists():
            try:
                page_url_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                page_url_manifest = []

        # Per-page metadata aggregates accumulated across the page's blocks.
        meta = {
            "unit_number": page.get("unit_number"),
            "unit_title_ar": page.get("unit_title_ar") or "",
            "lesson_number": page.get("lesson_number") or "",
            "lesson_title_ar": page.get("lesson_title_ar") or "",
            "lesson_title_en": page.get("lesson_title_en") or "",
            "diagram_image_urls": [],
            "standards": [],
            "has_diagram": False,
            "has_math": False,
        }

        blocks = page.get("content_blocks") or []
        blocks = sorted(blocks, key=lambda b: int(b.get("block_index", 0)))

        for block in blocks:
            # "other" blocks (title pages, answer keys, …) carry no instructional
            # content and are excluded from embedding by the proposed pipeline —
            # exclude them here too so the representation matches.
            if (block.get("content_type") or "other") == "other":
                continue

            # Build a proposed-style chunk for this single block, then render it
            # with the SAME representation function the proposed pipeline uses.
            pseudo_chunk = new_chunk_from_block(page, block, semester, page_url_manifest)
            source_block_id = pseudo_chunk["source_block_id"]
            rep_text = build_embedding_text(pseudo_chunk, cache_dir, config)

            # Append the representation's tokens, tagged with this page number and
            # the stable ID of the original block that produced them.
            block_units = 0
            for token in rep_text.split():
                stream.append((token, page_number, source_block_id))
                block_units += 1

            # Accumulate this block's total contribution to the complete stream.
            source_block_total_units[source_block_id] = (
                source_block_total_units.get(source_block_id, 0) + block_units
            )

            # Record this block's OWN metadata so a fixed window can aggregate
            # diagrams/standards/flags from only the blocks it actually covers.
            bm = block_meta.setdefault(
                source_block_id,
                {
                    "page_number": page_number,
                    "unit_number": page.get("unit_number"),
                    "unit_title_ar": page.get("unit_title_ar") or "",
                    "lesson_number": page.get("lesson_number") or "",
                    "lesson_title_ar": page.get("lesson_title_ar") or "",
                    "lesson_title_en": page.get("lesson_title_en") or "",
                    "diagram_image_urls": [],
                    "standards": [],
                    "has_diagram": False,
                    "has_math": False,
                },
            )
            bm["has_diagram"] = bm["has_diagram"] or bool(pseudo_chunk.get("has_diagram"))
            bm["has_math"] = bm["has_math"] or bool(pseudo_chunk.get("has_math"))
            for url in pseudo_chunk.get("diagram_image_urls") or []:
                if url not in bm["diagram_image_urls"]:
                    bm["diagram_image_urls"].append(url)
            for std in pseudo_chunk.get("standards") or []:
                if std not in bm["standards"]:
                    bm["standards"].append(std)

            # Roll the block's signals up into the page-level metadata aggregate.
            meta["has_diagram"] = meta["has_diagram"] or bool(pseudo_chunk.get("has_diagram"))
            meta["has_math"] = meta["has_math"] or bool(pseudo_chunk.get("has_math"))
            for url in pseudo_chunk.get("diagram_image_urls") or []:
                if url not in meta["diagram_image_urls"]:
                    meta["diagram_image_urls"].append(url)
            for std in pseudo_chunk.get("standards") or []:
                if std not in meta["standards"]:
                    meta["standards"].append(std)

        page_meta[page_number] = meta

    return stream, page_meta, source_block_total_units, block_meta
