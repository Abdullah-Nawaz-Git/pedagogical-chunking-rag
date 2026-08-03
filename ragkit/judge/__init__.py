"""
ragkit.judge
════════════

LLM-as-judge scoring inspired by RAGAS dimensions — a SECONDARY, end-to-end
evaluation that runs ALONGSIDE (never replacing) the deterministic retrieval
metrics in ``ragkit.retrieval``.

Why "inspired by", not "RAGAS scores"
-------------------------------------
The four dimensions below borrow their intent from RAGAS, but the prompts,
parsing, and aggregation are the project's own and are tuned for an Arabic
mathematics corpus. They are therefore reported as *LLM-as-judge scores inspired
by RAGAS dimensions*. Retrieval-native metrics (Hit@1, Hit@5, MRR, Gold-Unit /
Gold-Page recall) remain the PRIMARY evidence; the judge is corroborating,
model-dependent evidence and is labelled as such everywhere.

The four dimensions (each scored in isolation, 0.0–1.0 + rationale)
-------------------------------------------------------------------
    Context Recall     Did the retrieved contexts contain the information needed
                       to answer the question (per the reference answer)?
    Context Precision  Are useful contexts ranked above less-useful ones?
    Faithfulness       Is the generated answer supported by the retrieved context?
    Answer Relevancy   Does the generated answer directly answer the question?

What this package consumes (all FROZEN, never rewritten)
--------------------------------------------------------
    * ``qa_dataset/qa_dataset_v1.jsonl``      — question + reference answer,
    * ``retrieval_eval/retrieval_records_*``  — retrieved chunk ids/ranks/scores,
                                                 the frozen generator context, and
                                                 the generated answer (if any),
    * ``cache*/chunks.json``                  — chunk text for the contexts.

It never reruns QA generation, ingestion, embedding, indexing, or Pinecone
retrieval, and it never modifies QA records, gold mappings, chunk ids, or
provenance semantics.

Input isolation + no gold leakage
---------------------------------
Each dimension receives ONLY the inputs its ``JudgeDimensionSpec`` allows
(``ragkit.config.JUDGE_DIMENSION_SPECS``). No dimension is ever shown gold unit
ids, mapping labels, Hit@k values, mapping status, or any other gold-derived
relevance signal — those stay in the deterministic pipeline.

Module layout (one client, one prompt builder, one parser, one runner)
----------------------------------------------------------------------
    ``schemas``  output layout + the per-item record schema + config IO
    ``corpus``   loads QA + retrieval records + chunks; builds per-metric inputs
    ``prompts``  the four isolated prompts + shared judge guidance
    ``client``   pluggable judge providers (mock / Claude-Vertex / Gemini-Vertex)
    ``parser``   the shared JSON-only parser + score-range validation
    ``scoring``  per-item, per-metric scoring runner (resumable)
    ``metrics``  aggregation by system/metric/type/status + paired comparison
    ``report``   per-item + aggregate artifacts and the run manifest
    ``runner``   the orchestrator + CLI

Run one system, or all three:

    python -m experiments.judge_proposed
    python -m experiments.judge_b2
    python -m experiments.judge_b1
    python -m ragkit.judge.runner --system all

Add ``--provider mock`` for a deterministic offline smoke test that needs no
credentials, and ``--mode retrieval_only`` to score only the context dimensions.
"""

from __future__ import annotations

__all__ = [
    "client",
    "corpus",
    "metrics",
    "parser",
    "prompts",
    "report",
    "schemas",
    "scoring",
    "runner",
]
