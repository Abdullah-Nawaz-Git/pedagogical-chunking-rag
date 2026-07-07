"""Proposed pipeline: VLM structured extraction + pedagogical chunking.

This is the main pipeline under study. Each PDF page is extracted with Gemini
vision into structured pedagogical blocks (with diagram crops uploaded to
Cloudinary), then chunked block-by-block so that each chunk corresponds to a
coherent pedagogical unit. Chunks are embedded with Gemini and upserted to the Pinecone index.

Run with:

    python -m experiments.proposed --pdf book.pdf --semester 1

See ``--help`` for the full set of options (page ranges, cache dir, resume
flags, etc.). All knobs default to the values that previously lived at the top
of ``ingest.py``.
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.pipeline import run

# The proposed experiment: Gemini extraction + pedagogical chunking, writing to
# the primary index. This is the only experiment that produces diagram crops.
CONFIG = cfg.ExperimentConfig(
    name="proposed",
    extraction=cfg.EXTRACTION_GEMINI,
    chunking=cfg.CHUNKING_PEDAGOGICAL,
    index_env_var="PINECONE_INDEX_NAME",
    default_index_name="curriculum-highschool",
    default_cache_dir="cache",
    uses_diagrams=True,
)


def main() -> int:
    return run(CONFIG)


if __name__ == "__main__":
    sys.exit(main())