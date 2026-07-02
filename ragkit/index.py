"""
ragkit.index
════════════

Stage 3C — Pinecone index management and upsert.

Shared by every experiment unchanged. Each experiment targets its OWN index
(via a distinct env var, see ``ExperimentConfig.index_env_var``) so results
never collide, but the index creation and upsert logic is identical so the
storage layer is held constant across the comparison.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from .config import IndexConfig


def sanitize_metadata(
    chunk: Dict[str, Any],
    config: IndexConfig,
    content_text: str = "",
) -> Dict[str, Any]:
    """
    Build a flat metadata dict suitable for storage in a Pinecone vector record.

    All values are coerced to types Pinecone accepts (str, int, float, bool,
    list of str). Long text fields are truncated to ``config.max_metadata_text_chars``
    to stay within Pinecone's per-vector metadata size limit.
    """
    limit = config.max_metadata_text_chars

    def _truncate_text(text: str) -> str:
        """Truncate text to limit chars, appending '...' if truncated."""
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    return {
        "page_number": int(chunk["page_number"]),
        "page_range_start": int(chunk["page_range"][0]),
        "page_range_end": int(chunk["page_range"][1]),
        # unit_number defaults to 0 when unknown (Pinecone requires a scalar, not null)
        "unit_number": int(chunk["unit_number"]) if chunk.get("unit_number") is not None else 0,
        "lesson_number": chunk.get("lesson_number") or "",
        "unit_title_ar": chunk.get("unit_title_ar") or "",
        "lesson_title_ar": chunk.get("lesson_title_ar") or "",
        "lesson_title_en": chunk.get("lesson_title_en") or "",
        "heading_ar": chunk.get("heading_ar") or "",
        # Store the full embedding text so retrieval results are self-contained
        "content_text_ar": _truncate_text(content_text),
        "content_type": chunk.get("content_type") or "other",
        "has_diagram": bool(chunk.get("has_diagram")),
        "has_math": bool(chunk.get("has_math")),
        "diagram_image_urls": list(chunk.get("diagram_image_urls") or []),
        "standards": list(chunk.get("standards") or []),
        # problem_numbers stored as strings to satisfy Pinecone's list-of-strings type
        "problem_numbers": [str(p) for p in (chunk.get("problem_numbers") or [])],
        "subject": "mathematics",
        "language": "ar",
        "extraction_confidence": float(chunk.get("extraction_confidence") or 0.0),
        # Lightweight provenance only — the full source_block_* dicts live in the
        # local *_provenance.jsonl files to avoid Pinecone's metadata-size limit.
        "source_block_count": len(chunk.get("source_block_ids") or []),
    }


def ensure_pinecone_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    config: IndexConfig,
) -> Any:
    """
    Return a handle to a Pinecone index, creating it first if it does not exist.

    New indexes are created as serverless using the cloud/region/metric from
    ``config``. After creation the function polls until the index reports
    ready=True (up to 60 seconds) before returning.
    """
    existing = {ix["name"] for ix in pc.list_indexes()}
    if index_name not in existing:
        logging.info("Creating Pinecone index %s (dim=%d)", index_name, dimension)
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=config.metric,
            spec=ServerlessSpec(cloud=config.cloud, region=config.region),
        )
        # Poll until the index is ready to accept upserts
        for _ in range(30):
            desc = pc.describe_index(index_name)
            if getattr(desc, "status", {}).get("ready"):
                break
            time.sleep(2)
    return pc.Index(index_name)


def upsert_to_pinecone(
    index: Any,
    chunks: List[Dict[str, Any]],
    vectors: List[List[float]],
    config: IndexConfig,
    texts: Optional[List[str]] = None,
) -> int:
    """
    Upsert chunk vectors and metadata into Pinecone in batches.

    Chunks with empty vectors are skipped. Returns the total number of vectors
    successfully upserted.
    """
    batch_size = config.upsert_batch_size

    # Build the list of Pinecone upsert items, pairing each chunk with its vector
    items = []
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        if not vec:
            continue
        content_text = ""
        if texts is not None and idx < len(texts):
            content_text = texts[idx] or ""
        items.append({
            "id": chunk["chunk_id"],
            "values": vec,
            "metadata": sanitize_metadata(chunk, config, content_text),
        })

    upserted = 0
    for i in tqdm(range(0, len(items), batch_size), desc="Stage 3C — Pinecone upsert"):
        batch = items[i : i + batch_size]
        index.upsert(vectors=batch)
        upserted += len(batch)
    return upserted
