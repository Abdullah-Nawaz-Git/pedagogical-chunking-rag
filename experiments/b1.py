"""Baseline B1: Tesseract OCR + fixed-window chunking.

The simplest baseline. Pages are rendered to images and run through Tesseract
OCR (Arabic), and the resulting raw text is split into fixed 512-token windows
with overlap. No structure, no diagrams. Writes to ``curriculum-highschool-b1``.

Run with:

    python -m experiments.b1 --pdf /Users/abdullah/Downloads/rag-refactored/MATH_Grade10_Semester2_QTR_AR.pdf --semester 2 --stop-after-texts --start-page 15 --end-page 236 
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.pipeline import run

# B1 isolates the value of structured extraction + pedagogical chunking by
# replacing both with the crudest possible alternative (OCR + fixed windows).
CONFIG = cfg.ExperimentConfig(
    name="b1",
    extraction=cfg.EXTRACTION_TESSERACT,
    chunking=cfg.CHUNKING_FIXED_OCR,
    index_env_var="PINECONE_INDEX_NAME_B1",
    default_index_name="curriculum-highschool-b1",
    default_cache_dir="cache_b1",
    uses_diagrams=False,
    chunk_content_type="text",
    chunk_id_prefix="b1",
)


def main() -> int:
    return run(CONFIG)


if __name__ == "__main__":
    sys.exit(main())
