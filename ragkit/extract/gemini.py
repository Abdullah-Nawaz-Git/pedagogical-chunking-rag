"""
ragkit.extract.gemini
══════════════════════

Stage 1.5 / 1.6 — structured extraction with Gemini, plus diagram crop & upload.

Used by the proposed system and by baseline B2 (which holds extraction +
representation constant and varies only the chunker). This module owns:

    * the extraction prompt and the Pydantic schema it must satisfy,
    * per-page structured-JSON extraction (``extract_pages``),
    * forward-fill of unit/lesson metadata (``normalize_extractions_metadata``),
    * cropping each diagram out of the page PNG and uploading it to Cloudinary
      (``crop_and_upload``), writing a per-page URL manifest consumed downstream.

The OCR baseline (B1) does NOT use this module — see ``extract.tesseract``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image              # Pillow — crops diagram regions from PNGs
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

# Google GenAI SDK (Vertex AI)
from google import genai
from google.genai import types

# Cloudinary — cloud storage for cropped diagram images
import cloudinary
import cloudinary.uploader

from ..cache import append_log
from ..config import GeminiExtractionConfig


# ════════════════════════════════════════════
# EXTRACTION PROMPT
# ════════════════════════════════════════════

# This prompt is sent to Gemini alongside each page image.
# It instructs the model to return a single structured JSON object that
# faithfully mirrors every instructional block found on the page.
EXTRACTION_PROMPT = """You are processing a page from an Arabic high school mathematics textbook. 
Analyze the ENTIRE page and return ONE valid JSON object.

CRITICAL RULE: A single page almost always contains MULTIPLE distinct
instructional blocks. You MUST identify EVERY block on the page separately.
Do NOT collapse multiple blocks into one.

Return this exact schema — no text before or after the JSON:

{
  "page_number": <integer>,
  "unit_number": <integer or null>,
  "unit_title_ar": <string or null>,
  "lesson_number": <string e.g. "5-1" or null>,
  "lesson_title_ar": <string or null>,
  "lesson_title_en": <string or null>,
  "standard": <string e.g. "10.1.1" or null — from معيار الدرس>,
  "content_blocks": [
    {
      "block_index": <integer, 0-based, top-to-bottom right-to-left order>,
      "content_type": <one of the types listed below>,
      "heading_ar": <exact Arabic heading of this block e.g. "مثال 1"
                     or "تدرّب" — null if no heading>,
      "problem_numbers": <list of integers if this block contains numbered
                          problems e.g. [15,16,17] — empty list otherwise>,
      "text_ar": <full Arabic prose of this block, preserving paragraph
                  breaks with \\n. Include ALL text: problem statements,
                  solution steps, instructions, side notes, callout boxes.
                  If the block ends with يتبع في الصفحة التالية or any
                  continuation indicator, include that phrase verbatim.>,
      "math_expressions": <list of LaTeX strings for ALL equations,
                           expressions, and formulas in this block.
                           Use \\sqrt[n]{x} for radicals, \\frac{a}{b}
                           for fractions. Include every step in worked
                           solutions, not just the final answer.>,
      "diagrams": [
        {
          "diagram_index": <integer matching the bounding box index
                            from the page — null if unknown>,
            "box_2d": [y1, x1, y2, x2],
          "description": <Arabic description of the figure/graph/shape>,
          "labels": <list of all letters, numbers, angles, tick marks
                     visible on the diagram>
        }
      ],
      "named_elements": {
        "theorems": <list of Arabic theorem names>,
        "definitions": <list of Arabic terms being defined>,
        "vocabulary": [
          {"term_ar": <Arabic>, "term_en": <English or null>}
        ],
        "standards": <list of standard codes e.g. ["10.7.3"]>
      },
      "continues_to_next_page": <boolean — true ONLY if this exact block
                                 visually signals continuation: the word
                                 يتبع, an arrow, or clearly mid-sentence
                                 at the bottom of the page>,
      "continued_from_prev_page": <boolean — true if this block's heading
                                   or content indicates it started on the
                                   prior page>
    }
  ]
}

VALID content_type VALUES:
  "lesson_intro"            — استطيع / معيار الدرس / مصطلحات sidebar
  "explore"                 — استكشف وبزّر منطقيًا
  "essential_question"      — السؤال الأساس
  "theorem"                 — نظرية with formal statement and proof
  "definition"              — تعريف of a mathematical term
  "example"                 — مثال N (fully worked solution)
  "exercise"                — تدرّب (practice problems, no solutions shown)
  "reinforce"               — عزّز فهمك
  "error_analysis"          — حلّل الخطأ
  "reasoning"               — فكّر وثابر في الحل
  "higher_order_thinking"   — مهارات التفكير العليا
  "look_for_relationships"  — ابحث عن العلاقات
  "review"                  — مراجعة
  "stem_project"            — مشروع STEM
  "other"                   — index pages, table of contents, references,
                               title pages, blank pages, unit openers,
                               answer keys, appendices, or anything that
                               does not fit the above types

BLOCK SPLITTING RULES — follow exactly:
1. Every visually distinct section with its own colored header or box
   is a separate block.
2. Each numbered مثال is its own block, even if multiple appear on
   the same page.
3. تدرّب and عزّز فهمك are ALWAYS separate blocks even if adjacent.
4. A sidebar callout (فكّر وثابر في الحل, side notes) is its own block.
5. The vocabulary/مصطلحات list in the lesson sidebar is part of
   "lesson_intro" — do NOT make it a separate block.
6. The right column is processed before the left column.
   Within each column, process top to bottom.
7. Blocks of type "other" should still have text_ar fully extracted.
   Do not skip their content.

DIAGRAM INSTRUCTIONS:
- For geometric figures: describe the shape, all marked angles, all
  side lengths, which angle is the right angle if applicable.
- For coordinate graphs: describe the function, axis ranges, key
  points, and curve shape.
- Set diagram_index to match the spatial order of figures on the page
    (0 = first figure encountered reading right-to-left, top-to-bottom).
    This index links the block's diagram description to the cropped
    image file uploaded to Cloudinary.
- box_2d is required for each diagram and must be [y1, x1, y2, x2]
    integers on a 0–1000 scale (y1 = top edge × 1000, x1 = left × 1000, etc.).
- y1 < y2 and x1 < x2 always. Use the tightest box that fully contains the diagram.
- Labels should include every letter, number, degree marking, and
  tick mark visible.
  
UNIT AND LESSON EXTRACTION RULES:
1. Always scan the FOOTER BAR (bottom strip of the page) first.
   It contains either:
   - "الوحدة N <title>" → extract unit_number and unit_title_ar
   - "الدرس N-N <title>" → extract lesson_number and lesson_title_ar
   Both can appear on the same page if present in both header and footer.
2. Also check the TOP HEADER or any colored sidebar strip for
   lesson/unit identifiers — some pages show lesson number there.
3. lesson_number format: always "X-Y" string (e.g., "1-4"), never integer.
4. If the footer says "الدرس 1-4", the unit_number is the first digit (1),
   so infer unit_number = 1 even if "الوحدة" doesn't appear explicitly.
5. unit_title_ar and lesson_title_ar should be the full Arabic title
   excluding the prefix word (الوحدة / الدرس) and number.

If any field is not present on this page, use null or [].
"""


# ════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════

# These models validate the JSON returned by Gemini for each page.
# They mirror the schema described in EXTRACTION_PROMPT above.

class Vocab(BaseModel):
    term_ar: Optional[str] = None
    term_en: Optional[str] = None


class NamedElements(BaseModel):
    theorems: List[str] = Field(default_factory=list)
    definitions: List[str] = Field(default_factory=list)
    vocabulary: List[Vocab] = Field(default_factory=list)
    standards: List[str] = Field(default_factory=list)


class Diagram(BaseModel):
    diagram_index: Optional[int] = None
    # box_2d: [y1, x1, y2, x2] on a 0–1000 scale; converted to 0–1 floats later
    box_2d: Optional[List[float]] = None
    description: Optional[str] = None
    labels: List[Any] = Field(default_factory=list)


class ContentBlock(BaseModel):
    block_index: int
    content_type: str
    heading_ar: Optional[str] = None
    problem_numbers: List[int] = Field(default_factory=list)
    text_ar: Optional[str] = None
    math_expressions: List[str] = Field(default_factory=list)
    diagrams: List[Diagram] = Field(default_factory=list)
    named_elements: NamedElements = Field(default_factory=NamedElements)
    continues_to_next_page: bool = False
    continued_from_prev_page: bool = False


class PageExtraction(BaseModel):
    page_number: int
    unit_number: Optional[int] = None
    unit_title_ar: Optional[str] = None
    lesson_number: Optional[str] = None
    lesson_title_ar: Optional[str] = None
    lesson_title_en: Optional[str] = None
    standard: Optional[str] = None
    content_blocks: List[ContentBlock] = Field(default_factory=list)


# ════════════════════════════════════════════
# JSON UTILITIES
# ════════════════════════════════════════════


def strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` markdown fences that Gemini sometimes wraps output in."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(text: str) -> Any:
    """Strip code fences then parse as JSON."""
    text = strip_code_fence(text)
    return json.loads(text)


def _write_bbox_sidecar(
    parsed_page: Dict[str, Any],
    out_path: Path,
    validation_log: Optional[Path] = None,
) -> None:
    """
    Extract bounding-box data from a parsed page dict and write a sidecar JSON file.

    The sidecar is consumed by the crop-and-upload stage (Stage 1.6).
    Each entry records the normalised (0–1) coordinates of one diagram so the
    crop stage can compute pixel coordinates from the page dimensions.

    Any diagram that is missing a valid box_2d field is skipped and, if a
    validation_log path is provided, an entry is written to that log.
    """
    results: List[Dict[str, Any]] = []
    missing_boxes: List[str] = []
    blocks = parsed_page.get("content_blocks") or []

    for block in blocks:
        block_index = block.get("block_index")
        for diag in block.get("diagrams") or []:
            if not isinstance(diag, dict):
                continue
            d_idx = diag.get("diagram_index")
            box = diag.get("box_2d")

            # Validate: box must be a list of exactly four numeric values
            if not isinstance(box, list) or len(box) != 4:
                if validation_log is not None:
                    missing_boxes.append(f"block={block_index} diagram_index={d_idx}")
                continue

            try:
                y1_raw, x1_raw, y2_raw, x2_raw = [float(v) for v in box]
            except (TypeError, ValueError):
                if validation_log is not None:
                    missing_boxes.append(f"block={block_index} diagram_index={d_idx}")
                continue

            # Convert from 0–1000 scale to normalised 0–1 floats and clamp
            x1 = max(0.0, min(x1_raw / 1000.0, 1.0))
            y1 = max(0.0, min(y1_raw / 1000.0, 1.0))
            x2 = max(0.0, min(x2_raw / 1000.0, 1.0))
            y2 = max(0.0, min(y2_raw / 1000.0, 1.0))

            # Skip degenerate boxes where top >= bottom or left >= right
            if x1 >= x2 or y1 >= y2:
                continue

            try:
                diagram_index = int(diag.get("diagram_index"))
            except Exception:
                diagram_index = len(results)

            results.append({
                "diagram_index": diagram_index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "diagram_type": "other_figure",
                "description": str(diag.get("description") or ""),
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Log any diagrams that were skipped due to missing box data
    if validation_log is not None and missing_boxes:
        page_number = parsed_page.get("page_number")
        for entry in missing_boxes:
            append_log(validation_log, f"page {page_number}: missing box_2d ({entry})")


# ════════════════════════════════════════════
# STAGE 1.5 — STRUCTURED EXTRACTION (Gemini)
# ════════════════════════════════════════════


def _process_extract_page(
    png_path: Path,
    extractions_dir: Path,
    bbox_dir: Path,
    cache_dir: Path,
    cloud_client: genai.Client,
    failed_log: Path,
    config: GeminiExtractionConfig,
) -> None:
    """
    Run structured extraction on a single page image using Gemini.

    Sends the PNG and EXTRACTION_PROMPT to the model and expects a JSON
    response conforming to the PageExtraction schema. On success:
      - The raw parsed JSON is written to extractions_dir/<stem>.json
      - A bbox sidecar is written to bbox_dir/page_<key>_bboxes.json

    Failures (parse errors, validation errors, API errors) are logged to
    failed_log and the function returns without writing output, leaving the
    page eligible for a re-run on the next pipeline invocation.
    """
    key = png_path.stem.replace("page_", "")
    page_number = int(key)
    out_path = extractions_dir / f"{png_path.stem}.json"
    bbox_out_path = bbox_dir / f"page_{key}_bboxes.json"

    # Idempotency: skip if both output files already exist
    if out_path.exists() and bbox_out_path.exists():
        return

    try:
        with open(png_path, "rb") as f:
            img_bytes = f.read()

        # Send the page image + extraction prompt to Gemini
        try:
            response = cloud_client.models.generate_content(
                model=config.flash_model,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    types.Part.from_text(text=EXTRACTION_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=config.temperature,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=config.thinking_budget,
                    ),
                ),
            )
        except Exception as e:
            append_log(failed_log, f"page {key}: request error: {e}")
            return

        # Extract the text payload from the response
        raw = getattr(response, "text", None) or ""

        # Parse the JSON response
        try:
            parsed = safe_json_loads(raw)
        except Exception as e:
            append_log(failed_log, f"page {key}: json parse error: {e}")
            return

        # The response must be a JSON object, not an array or scalar
        if not isinstance(parsed, dict):
            append_log(failed_log, f"page {key}: response is not a JSON object")
            return

        # Gemini sometimes returns {"error": "..."} for problematic pages
        if "error" in parsed:
            append_log(failed_log, f"page {key}: model error: {parsed['error']}")
            return

        # Always stamp the page_number so downstream code can rely on it
        parsed["page_number"] = page_number

        # Validate the response against the Pydantic schema
        try:
            PageExtraction.model_validate(parsed)
        except ValidationError as ve:
            append_log(failed_log, f"page {key}: validation error: {ve}")
            return

        # Persist the extraction JSON
        out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write the bounding-box sidecar for the crop stage
        _write_bbox_sidecar(parsed, bbox_out_path, cache_dir / "bbox_validation.log")

    except Exception as e:
        append_log(failed_log, f"page {key}: {e}")


def extract_pages(
    cloud_client: genai.Client,
    pages_dir: Path,
    extractions_dir: Path,
    bbox_dir: Path,
    cache_dir: Path,
    config: GeminiExtractionConfig,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> None:
    """
    Run _process_extract_page sequentially over all pending page PNGs in range.

    Already-extracted pages (both .json and _bboxes.json present) are skipped.
    """
    from ..render import key_in_range

    extractions_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    failed_log = cache_dir / "failed_pages.log"

    # Gather all page PNGs within the requested range
    page_pngs = sorted(pages_dir.glob("page_*.png"))
    page_pngs = [
        p for p in page_pngs
        if key_in_range(p.stem.replace("page_", ""), start_page, end_page)
    ]

    # Filter to only pages missing at least one output file
    pending = [
        p for p in page_pngs
        if not (
            (extractions_dir / f"{p.stem}.json").exists()
            and (bbox_dir / f"page_{p.stem.replace('page_', '')}_bboxes.json").exists()
        )
    ]

    if not pending:
        logging.info("Stage 1.5: all pages already extracted, skipping.")
        return

    for p in tqdm(pending, desc=f"Stage 1.5 — extraction ({config.flash_model})"):
        _process_extract_page(
            p, extractions_dir, bbox_dir, cache_dir, cloud_client, failed_log, config,
        )


def resolve_metadata(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Forward-fill unit and lesson metadata across a sorted list of page dicts.

    Pages that lack unit_number or lesson_number inherit the most recently
    seen values from earlier pages. This compensates for pages (diagrams,
    mid-lesson pages, etc.) where the textbook does not repeat the header.
    """
    last_unit_number = None
    last_unit_title_ar = None
    last_lesson_number = None
    last_lesson_title_ar = None
    last_lesson_title_en = None

    for page in sorted(pages, key=lambda p: int(p.get("page_number", 0))):
        if page.get("unit_number"):
            # Update the running unit metadata
            last_unit_number = page["unit_number"]
            last_unit_title_ar = page.get("unit_title_ar")
        else:
            # Inherit from previous page
            page["unit_number"] = last_unit_number
            page["unit_title_ar"] = last_unit_title_ar

        if page.get("lesson_number"):
            # Update the running lesson metadata
            last_lesson_number = page["lesson_number"]
            last_lesson_title_ar = page.get("lesson_title_ar")
            last_lesson_title_en = page.get("lesson_title_en")
        else:
            # Inherit from previous page
            page["lesson_number"] = last_lesson_number
            page["lesson_title_ar"] = last_lesson_title_ar
            page["lesson_title_en"] = last_lesson_title_en

    return pages


def normalize_extractions_metadata(
    extractions_dir: Path,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> int:
    """
    Forward-fill unit/lesson metadata across pages.

    Many pages in Arabic textbooks do not repeat the unit or lesson header.
    This function calls resolve_metadata() to propagate the last-seen values
    forward through the page sequence, then re-writes each JSON file in place.

    Returns the number of files updated.
    """
    from ..render import page_in_range

    page_files = sorted(extractions_dir.glob("page_*.json"))
    pages: List[Dict[str, Any]] = []
    path_by_page: Dict[int, Path] = {}

    for pf in page_files:
        try:
            data = json.loads(pf.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            pn = int(data.get("page_number", 0))
            if page_in_range(pn, start_page, end_page):
                pages.append(data)
                path_by_page[pn] = pf
        except Exception:
            continue

    if not pages:
        return 0

    pages.sort(key=lambda p: int(p.get("page_number", 0)))
    resolved = resolve_metadata(pages)

    updated = 0
    for page in resolved:
        pn = int(page.get("page_number", 0))
        out_path = path_by_page.get(pn)
        if not out_path:
            continue
        out_path.write_text(
            json.dumps(page, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        updated += 1

    return updated


# ════════════════════════════════════════════
# STAGE 1.6 — CROP & UPLOAD TO CLOUDINARY
# ════════════════════════════════════════════


def _process_crop_upload_page(
    bbox_file: Path,
    pages_dir: Path,
    diagrams_dir: Path,
    diagram_urls_dir: Path,
    page_dimensions: Dict[str, Dict[str, int]],
    semester: int,
    upload_failed_log: Path,
) -> None:
    """
    Crop all diagrams from a single page and upload each one to Cloudinary.

    For each diagram bounding box recorded in the sidecar JSON:
      1. Convert normalised coordinates to pixel coordinates using the page dims.
      2. Add 12 px of padding on all sides to avoid cutting off edges.
      3. Save the cropped image to diagrams_dir.
      4. Upload to Cloudinary under math-textbook/semester-<N>/.
      5. Record the resulting URL in a per-page manifest JSON.

    The manifest is consumed by the chunking stage to embed Cloudinary URLs
    directly into Pinecone metadata so the RAG system can reference images.
    """
    m = re.match(r"page_(\d+)_bboxes\.json", bbox_file.name)
    if not m:
        return
    key = m.group(1)

    # Idempotency: skip if this page's URL manifest already exists
    manifest_path = diagram_urls_dir / f"page_{key}_urls.json"
    if manifest_path.exists():
        return

    try:
        bboxes = json.loads(bbox_file.read_text(encoding="utf-8"))
    except Exception:
        bboxes = []

    # Write an empty manifest for pages with no diagrams so they are not re-processed
    if not bboxes:
        manifest_path.write_text("[]", encoding="utf-8")
        return

    if key not in page_dimensions:
        append_log(upload_failed_log, f"page {key}: missing page dimensions")
        manifest_path.write_text("[]", encoding="utf-8")
        return

    page_w = page_dimensions[key]["width"]
    page_h = page_dimensions[key]["height"]
    page_png = pages_dir / f"page_{key}.png"

    if not page_png.exists():
        append_log(upload_failed_log, f"page {key}: missing PNG")
        manifest_path.write_text("[]", encoding="utf-8")
        return

    manifest: List[Dict[str, Any]] = []

    try:
        img = Image.open(page_png)
    except Exception as e:
        append_log(upload_failed_log, f"page {key}: cannot open PNG: {e}")
        manifest_path.write_text("[]", encoding="utf-8")
        return

    try:
        for bbox in bboxes:
            try:
                d_idx = int(bbox["diagram_index"])

                # Convert normalised 0–1 coordinates to pixel coordinates
                px1 = int(bbox["x1"] * page_w)
                py1 = int(bbox["y1"] * page_h)
                px2 = int(bbox["x2"] * page_w)
                py2 = int(bbox["y2"] * page_h)

                # Add 12-pixel padding on each side, clamped to image bounds
                x1_pad = max(0, px1 - 12)
                y1_pad = max(0, py1 - 12)
                x2_pad = min(page_w, px2 + 12)
                y2_pad = min(page_h, py2 + 12)

                # Skip if padding collapsed the box
                if x2_pad <= x1_pad or y2_pad <= y1_pad:
                    continue

                crop_path = diagrams_dir / f"page_{key}_d{d_idx}.png"

                # Only re-crop if the file doesn't already exist locally
                if not crop_path.exists():
                    cropped = img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
                    cropped.save(crop_path)

                # Upload to Cloudinary; overwrite=True makes this idempotent
                result = cloudinary.uploader.upload(
                    str(crop_path),
                    folder=f"math-textbook/semester-{semester}",
                    public_id=f"page_{key}_d{d_idx}",
                    resource_type="image",
                    overwrite=True,
                )
                public_url = result["secure_url"]

                manifest.append({
                    "diagram_index": d_idx,
                    "diagram_type": bbox.get("diagram_type", "other_figure"),
                    "description": bbox.get("description", ""),
                    "local_path": str(crop_path),
                    "cloudinary_url": public_url,
                })
            except Exception as e:
                append_log(
                    upload_failed_log,
                    f"page {key} diagram {bbox.get('diagram_index')}: {e}",
                )
                continue
    finally:
        img.close()

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def crop_and_upload(
    pages_dir: Path,
    bbox_dir: Path,
    diagrams_dir: Path,
    diagram_urls_dir: Path,
    page_dimensions: Dict[str, Dict[str, int]],
    semester: int,
    cache_dir: Path,
    start_page: int = 1,
    end_page: Optional[int] = None,
) -> None:
    """
    Iterate over all bbox sidecar files in the requested page range and
    process each page sequentially (crop + Cloudinary upload).
    """
    diagrams_dir.mkdir(parents=True, exist_ok=True)
    diagram_urls_dir.mkdir(parents=True, exist_ok=True)
    upload_failed_log = cache_dir / "upload_failed.log"

    # Collect all bbox sidecar files, filtered to the requested page range
    bbox_files = sorted(bbox_dir.glob("page_*_bboxes.json"))

    def _bbox_key(p: Path) -> Optional[str]:
        m = re.match(r"page_(\d+)_bboxes\.json", p.name)
        return m.group(1) if m else None

    bbox_files = [
        p for p in bbox_files
        if (k := _bbox_key(p)) is not None and key_in_range(k, start_page, end_page)
    ]

    # Filter to only pages whose URL manifest has not yet been written
    pending = [
        p for p in bbox_files
        if not (diagram_urls_dir / f"page_{_bbox_key(p)}_urls.json").exists()
    ]

    if not pending:
        logging.info("Stage 1.6: all pages already uploaded, skipping.")
        return

    for p in tqdm(pending, desc="Stage 1.6 — crop & upload"):
        _process_crop_upload_page(
            p, pages_dir, diagrams_dir, diagram_urls_dir,
            page_dimensions, semester, upload_failed_log,
        )
