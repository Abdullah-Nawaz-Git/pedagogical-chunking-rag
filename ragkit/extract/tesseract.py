"""
ragkit.extract.tesseract
═════════════════════════

Tesseract Arabic OCR — the extraction stage for baselines B1 and B2.

This is the cheapest, most naive extraction a practitioner would build: dump
raw text per page with no structure, no math handling, no diagram awareness,
no pedagogical metadata. B1 and B2 use IDENTICAL OCR extraction; they differ
only in the chunker applied afterwards, so this single shared implementation
keeps the extraction variable constant between them.

The system-level ``tesseract`` binary plus the Arabic language pack (``ara``)
must be installed separately — see the README / requirements.txt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from PIL import Image                 # used to open the rendered page PNGs for OCR
from tqdm import tqdm

# Tesseract OCR Python bindings.
import pytesseract

from ..cache import append_log
from ..config import TesseractExtractionConfig


def ocr_pages(
    pages_dir: Path,
    ocr_dir: Path,
    cache_dir: Path,
    config: TesseractExtractionConfig,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> Dict[int, str]:
    """
    Run Tesseract Arabic OCR over every rendered page PNG in the given range.

    The raw OCR text for each page is cached to ocr_dir/page_<key>.txt so the
    (slow) OCR step is idempotent and re-runnable after a partial failure.

    Returns a dict mapping page_number -> raw OCR text (ordered by page when
    iterated, because we sort the inputs).
    """
    from ..render import key_in_range

    ocr_dir.mkdir(parents=True, exist_ok=True)
    ocr_failed_log = cache_dir / "ocr_failed.log"

    # Collect the page PNGs produced by Stage 1 (rendering), filtered to range.
    page_pngs = sorted(pages_dir.glob("page_*.png"))
    page_pngs = [
        p for p in page_pngs
        if key_in_range(p.stem.replace("page_", ""), start_page, end_page)
    ]

    page_texts: Dict[int, str] = {}

    for png_path in tqdm(page_pngs, desc="Stage OCR — Tesseract"):
        key = png_path.stem.replace("page_", "")
        page_number = int(key)
        txt_path = ocr_dir / f"page_{key}.txt"

        # Idempotency: reuse a previously cached OCR result if present.
        if txt_path.exists():
            page_texts[page_number] = txt_path.read_text(encoding="utf-8")
            continue

        try:
            # Open the page image and hand it to Tesseract with the Arabic model.
            with Image.open(png_path) as img:
                text = pytesseract.image_to_string(img, lang=config.ocr_lang)
        except Exception as e:
            # OCR is best-effort: log the failure and treat the page as empty.
            append_log(ocr_failed_log, f"page {key}: OCR error: {e}")
            text = ""

        txt_path.write_text(text, encoding="utf-8")
        page_texts[page_number] = text

    return page_texts
