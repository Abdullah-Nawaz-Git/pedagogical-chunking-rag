"""
ragkit.render
═════════════

Stage 1 — render a range of PDF pages to PNG images.

This stage is shared by every experiment unchanged: identical rendering keeps
the page images (and therefore everything downstream) constant across the
proposed system and all baselines. The page-range helpers (``page_in_range`` /
``key_in_range``) live here too because they are used by several later stages
to honour ``--start-page`` / ``--end-page``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import fitz                        # PyMuPDF — renders PDF pages to images
from tqdm import tqdm


# ════════════════════════════════════════════
# PAGE-RANGE HELPERS
# ════════════════════════════════════════════


def page_in_range(page_number: int, start_page: int, end_page: Optional[int]) -> bool:
    """Return True if page_number falls within the [start_page, end_page] range."""
    if page_number < start_page:
        return False
    if end_page is not None and page_number > end_page:
        return False
    return True


def key_in_range(key: str, start_page: int, end_page: Optional[int]) -> bool:
    """Return True if the zero-padded page key (e.g. "0015") is within range."""
    try:
        return page_in_range(int(key), start_page, end_page)
    except ValueError:
        return False


# ════════════════════════════════════════════
# STAGE 1 — PDF RENDERING
# ════════════════════════════════════════════


def render_pdf_pages(
    pdf_path: Path,
    pages_dir: Path,
    dpi: int,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """
    Render a range of PDF pages to PNG images at the given DPI.

    Already-rendered pages (PNG + dimensions entry) are skipped, making this
    stage idempotent — safe to re-run after a partial failure.

    Returns a dict mapping zero-padded page keys (e.g. "0015") to their pixel
    dimensions {"width": W, "height": H}, which the crop stage needs later.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    dims_path = pages_dir / "page_dimensions.json"

    # Load any previously persisted dimensions so we can skip already-done pages
    page_dimensions: Dict[str, Dict[str, int]] = {}
    if dims_path.exists():
        try:
            page_dimensions = json.loads(dims_path.read_text(encoding="utf-8"))
        except Exception:
            page_dimensions = {}

    doc = fitz.open(str(pdf_path))

    # PyMuPDF uses 72 DPI internally; scale factor converts to target DPI
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    try:
        total = doc.page_count
        effective_end = total if end_page is None else min(end_page, total)

        if start_page > total:
            logging.warning(
                "start_page %d exceeds PDF page count %d; nothing to render.",
                start_page, total,
            )

        # page indices in PyMuPDF are 0-based; convert from 1-based page numbers
        target_indices = range(max(0, start_page - 1), effective_end)

        for i in tqdm(list(target_indices), desc="Stage 1 — render PDF pages"):
            page_number = i + 1
            key = f"{page_number:04d}"
            png_path = pages_dir / f"page_{key}.png"

            # Skip if already rendered and dimensions recorded
            if png_path.exists() and key in page_dimensions:
                continue

            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(str(png_path))
            page_dimensions[key] = {"width": pix.width, "height": pix.height}
    finally:
        doc.close()

    # Persist updated dimensions to disk
    dims_path.write_text(
        json.dumps(page_dimensions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return page_dimensions
