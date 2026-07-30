"""Retrieval evaluation for baseline B1 (Tesseract OCR + fixed windows).

Queries the B1 Pinecone index with every question in the frozen QA dataset and
scores the results against ``gold_mapping_b1.jsonl``.

IMPORTANT — B1 is NOT unit-level. Raw OCR windows retain no source-block
provenance, so B1's gold relevance is the PAGE-OVERLAP PROXY produced by
``ragkit.qa.gold_mapping``. This run therefore reports
``gold_page_recall_proxy``, never ``gold_unit_recall``, and every record plus
every report carries the caveat explicitly (see
``ragkit.config.B1_PROXY_DISCLAIMER``). Hit@1, Hit@5, and MRR are computed with
the same code as the other systems, but against this proxy relevance set.

Run with:

    python -m experiments.retrieval_b1

Offline smoke test (no credentials, deterministic):

    python -m experiments.retrieval_b1 --retriever local

See ``--help`` for the full set of options (``--top-k``, ``--generate-answers``,
``--output-dir``, ``--limit``, ``--dry-run``).
"""

from __future__ import annotations

import sys

from ragkit import config as cfg
from ragkit.retrieval.runner import run

# B1 is the practitioner floor. Its config is the ONLY place the page-level
# gold granularity is declared; the shared engine/metrics code never special-cases it.
SYSTEM = cfg.RETRIEVAL_SYSTEM_B1


def main() -> int:
    return run(SYSTEM)


if __name__ == "__main__":
    sys.exit(main())
