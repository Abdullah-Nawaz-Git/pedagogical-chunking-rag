"""
ragkit.retrieval
════════════════

Retrieval evaluation for the Proposed / B1 / B2 comparison.

This sub-package consumes the FROZEN artifacts produced elsewhere and never
modifies them:

    * ``qa_dataset/qa_dataset_v1.jsonl``   — the frozen QA dataset,
    * ``qa_dataset/gold_mapping_*.jsonl``  — per-system gold relevance,
    * ``cache*/chunks.json``               — the ingested chunk corpora,
    * the three Pinecone indexes ingestion already built (read-only).

Design (one engine, one metrics layer, three thin entry points):

    ``schemas``  output layout + the single per-question record schema
    ``corpus``   loads the QA dataset, gold mappings, and chunk caches
    ``engine``   shared query embedding + top-k retrieval (Pinecone or offline)
    ``metrics``  Hit@k, MRR, gold recall, and aggregation
    ``answer``   shared context budget/accounting + answer generation
    ``report``   per-question, per-system, and cross-system reports
    ``runner``   the orchestrator + CLI

Systems differ ONLY through ``ragkit.config.RetrievalSystemConfig``: which index
to query, which chunk cache to read, and what gold granularity their provenance
supports. Proposed and B2 are scored against source-block (instructional-unit)
relevance; B1 is scored against the page-overlap PROXY from
``ragkit.qa.gold_mapping`` and is labelled as such on every record and in every
report, because OCR windows retain no source-block provenance.

Run one system, or all three plus a comparison:

    python -m experiments.retrieval_proposed
    python -m experiments.retrieval_b2
    python -m experiments.retrieval_b1
    python -m ragkit.retrieval.runner --system all
    python -m ragkit.retrieval.runner --system all --generate-answers --answer-provider vertex

Add ``--retriever local`` for a deterministic offline smoke test that needs no
credentials, and ``--generate-answers`` to also run the shared generator.
"""

from __future__ import annotations

__all__ = [
    "answer",
    "corpus",
    "engine",
    "metrics",
    "report",
    "runner",
    "schemas",
]
