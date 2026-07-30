"""Retrieval evaluation for baseline B2 (Gemini extraction + fixed windows).

Queries the B2 Pinecone index with every question in the frozen QA dataset and
scores the results against ``gold_mapping_b2.jsonl``. B2 windows retain
source-block provenance (with per-block coverage), so recall here is TRUE
instructional-unit recall (Gold Unit Recall) and is directly comparable to the
Proposed system's — which is the point of the B2 control.



Run with:

    python -m experiments.retrieval_b2

Offline smoke test (no credentials, deterministic):

    python -m experiments.retrieval_b2 --retriever local

See ``--help`` for the full set of options (``--top-k``, ``--generate-answers``,
``--output-dir``, ``--limit``, ``--dry-run``).
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.retrieval.runner import run

# B2 holds extraction and representation constant against Proposed and varies
# only the chunker, so the retrieval protocol must be byte-for-byte identical.
SYSTEM = cfg.RETRIEVAL_SYSTEM_B2


def main() -> int:
    return run(SYSTEM)


if __name__ == "__main__":
    sys.exit(main())
