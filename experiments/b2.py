"""Baseline B2: Gemini structured extraction + fixed-window chunking.

B2 shares the proposed pipeline's rich Gemini extraction (including diagrams),
but flattens each page's structured blocks into the same representation stream
used for embedding and then splits that stream into fixed 512-token windows
with overlap. Comparing B2 against the proposed pipeline isolates the value of
*pedagogical* chunking while holding extraction quality constant. Upserts to
``curriculum-highschool-b2`` Pinecone index.

Run with:

    python -m experiments.b2 --pdf /Users/abdullah/Downloads/rag-refactored/MATH_Grade10_Semester2_QTR_AR.pdf --semester 2 --stop-after-texts --start-page 15 --end-page 236 --skip-bbox
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.pipeline import run

# B2 holds extraction constant (Gemini, with diagrams) and varies only the
# chunker: fixed windows over the representation stream instead of pedagogical
# block-aware chunks.
CONFIG = cfg.ExperimentConfig(
    name="b2",
    extraction=cfg.EXTRACTION_GEMINI,
    chunking=cfg.CHUNKING_FIXED_STREAM,
    index_env_var="PINECONE_INDEX_NAME_B2",
    default_index_name="curriculum-highschool-b2",
    default_cache_dir="cache_b2",
    uses_diagrams=True,
    chunk_content_type="text",
    chunk_id_prefix="b2",
)


def main() -> int:
    return run(CONFIG)


if __name__ == "__main__":
    sys.exit(main())
