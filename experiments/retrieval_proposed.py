"""Retrieval evaluation for the Proposed system (pedagogical chunks).

Queries the Proposed Pinecone index with every question in the frozen QA dataset
and scores the results against ``gold_mapping_proposed.jsonl``. Because Proposed
chunks carry source-block provenance, recall here is TRUE instructional-unit
recall (Gold Unit Recall).

Run with:

    python -m experiments.retrieval_proposed

Offline smoke test (no credentials, deterministic):

    python -m experiments.retrieval_proposed --retriever local

See ``--help`` for the full set of options (``--top-k``, ``--generate-answers``,
``--output-dir``, ``--limit``, ``--dry-run``).
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.retrieval.runner import run

# The Proposed retrieval experiment: same shared engine as B1/B2, differing only
# in which index is queried and which gold mapping defines relevance.
SYSTEM = cfg.RETRIEVAL_SYSTEM_PROPOSED


def main() -> int:
    return run(SYSTEM)


if __name__ == "__main__":
    sys.exit(main())
