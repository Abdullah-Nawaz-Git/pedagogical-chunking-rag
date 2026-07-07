"""
ragkit.cache
════════════

Cache-directory handling and logging helpers shared by every stage.

A "cache" here is just a directory on disk that holds the intermediate
artefacts of a run (rendered PNGs, per-page extraction JSON, OCR text, diagram
crops, URL manifests, logs). Every stage reads/writes inside it, and every
stage is idempotent: re-running after a partial failure skips work that has
already been persisted. ``CacheLayout`` centralises the sub-directory names so
the orchestrator and the individual stages never disagree about paths.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheLayout:
    """The on-disk sub-directory layout under a single cache root.

    Not every experiment uses every sub-directory (the OCR baselines have no
    ``extractions``/``bboxes``/diagram directories, for instance) but computing
    the paths up-front is cheap and keeps the orchestration code uniform.
    """

    root: Path

    @property
    def pages(self) -> Path:
        """Rendered page PNGs."""
        return self.root / "pages"

    @property
    def extractions(self) -> Path:
        """Per-page structured JSON (Gemini extraction)."""
        return self.root / "extractions"

    @property
    def bboxes(self) -> Path:
        """Per-page diagram bounding-box sidecar JSONs."""
        return self.root / "bboxes"

    @property
    def diagrams(self) -> Path:
        """Cropped diagram PNGs."""
        return self.root / "diagrams"

    @property
    def diagram_urls(self) -> Path:
        """Per-page Cloudinary URL manifests."""
        return self.root / "diagram_urls"

    @property
    def ocr(self) -> Path:
        """Raw Tesseract OCR text per page (OCR baselines)."""
        return self.root / "ocr"

    @property
    def page_dimensions(self) -> Path:
        """JSON map of page key -> pixel dimensions, written by the renderer."""
        return self.pages / "page_dimensions.json"


def setup_logging(cache_dir: Path) -> None:
    """Configure logging to write to both a file and stdout."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cache_dir / "pipeline.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def append_log(path: Path, message: str) -> None:
    """Append a single line to an error/warning log file, creating it if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")
