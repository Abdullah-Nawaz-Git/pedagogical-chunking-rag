"""
ragkit.judge.corpus
═══════════════════

Loads the FROZEN inputs a judge run scores on top of, and reconstructs — for each
QA item — the exact same context the answer generator saw.

Inputs (all read verbatim, never rewritten):

    * ``qa_dataset/qa_dataset_v1.jsonl``           — question + reference answer,
    * ``retrieval_eval/retrieval_records_<sys>``   — retrieved chunk ids / scores,
                                                      the recorded generator context,
                                                      and the generated answer,
    * ``cache*/chunks.json``                       — chunk text for the contexts,
    * ``qa_dataset/gold_mapping_<sys>.jsonl``      — mapping method (reporting only).

Fair context handling
---------------------
The judge must not let one system look stronger merely because it received a
longer context. So the context is re-assembled from the retrieved chunks under a
FIXED shared token budget (``JudgeContextConfig.context_budget_tokens``),
preserving retrieval order and truncating the overflowing chunk at the token
level — the same algorithm the answer generator used. The reconstructed chunk set
is cross-checked against the ``context_chunk_ids`` the retrieval run recorded;
any divergence (or a chunk with no resolvable text) is logged as a warning rather
than hidden. Token counts and truncation are recorded on every item.

The gold mapping is loaded ONLY to attach mapping method/status/type as reporting
metadata — none of it is ever passed to the judge.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .. import config as cfg
from . import schemas

logger = logging.getLogger("ragkit.judge.corpus")


# ════════════════════════════════════════════
# FROZEN CONTEXT RECONSTRUCTION
# ════════════════════════════════════════════


@dataclass(frozen=True)
class JudgeContext:
    """The ranked contexts scored for one item, plus measured size + provenance."""

    context_texts: Tuple[str, ...]
    chunk_ids: Tuple[str, ...]
    token_count: int
    budget_tokens: int
    truncated: bool
    char_count: int
    estimated_model_tokens: int
    source: str
    warnings: Tuple[str, ...] = field(default_factory=tuple)


def assemble_judge_context(
    ranked: List[Tuple[str, str]],
    context_config: cfg.JudgeContextConfig,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], int, bool]:
    """Concatenate ranked ``(chunk_id, text)`` pairs under the shared token budget.

    This mirrors ``ragkit.retrieval.answer.assemble_context`` exactly — whitespace
    tokens, rank markers counted toward the budget, first overflowing chunk
    truncated (not dropped) — so the judge scores the same-sized window the
    generator produced. Returns per-chunk truncated texts, their ids, the total
    token count, and whether truncation occurred.
    """
    budget = context_config.context_budget_tokens
    used_tokens = 0
    texts: List[str] = []
    chunk_ids: List[str] = []
    truncated = False

    for rank, (chunk_id, text) in enumerate(ranked, start=1):
        if used_tokens >= budget:
            truncated = True
            break
        # The generator prefixed each chunk with a rank marker and counted its
        # tokens; replicate that so the budget maths is identical.
        header_tokens = f"[{rank}] {chunk_id}".split()
        body_tokens = (text or "").split()
        remaining = budget - used_tokens
        if len(header_tokens) >= remaining:
            truncated = True
            break
        allowance = remaining - len(header_tokens)
        if len(body_tokens) > allowance:
            body_tokens = body_tokens[:allowance]
            truncated = True
        texts.append(" ".join(body_tokens))
        chunk_ids.append(chunk_id)
        used_tokens += len(header_tokens) + len(body_tokens)

    # ``used_tokens`` includes the two-token ``[rank] chunk_id`` marker exactly
    # as ``retrieval.answer.assemble_context`` does.  Do not derive this from the
    # body texts: that would under-report the budget actually used by generation.
    return tuple(texts), tuple(chunk_ids), used_tokens, truncated


# ════════════════════════════════════════════
# PER-ITEM JUDGE INPUT
# ════════════════════════════════════════════


@dataclass(frozen=True)
class JudgeItem:
    """Everything the four dimensions need for one (qa_id, system), pre-isolation.

    The scoring runner slices this per dimension using ``JudgeDimensionSpec`` so no
    dimension sees an input it is not entitled to. The gold/mapping fields here are
    REPORTING metadata only and are never rendered into any prompt.
    """

    qa_id: str
    system: str
    # Question metadata (from the frozen QA dataset).
    question_ar: str
    question_type: str
    difficulty: str
    answer_mode: str
    required_diagram: bool
    required_formula: bool
    # Judge inputs.
    reference_answer_ar: str
    generated_answer_ar: str
    answer_available: bool
    context: JudgeContext
    # Retrieval audit context (NOT shown to the judge).
    retrieved_chunk_ids: Tuple[str, ...]
    retrieved_scores: Tuple[float, ...]
    retrieved_ranks: Tuple[int, ...]
    # Answer-generation provenance carried from the retrieval record.
    answer_provider: str
    answer_model: str
    # Gold-mapping metadata for reporting only.
    gold_granularity: str
    mapping_type: str
    mapping_method: str
    mapping_status: str
    mapping_caveat: str
    warnings: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class JudgeCorpus:
    """All per-item judge inputs for one system, loaded once."""

    system: cfg.JudgeSystemConfig
    items: List[JudgeItem]
    generated_answers_present: bool


def _chunk_text(chunk: Optional[Dict[str, Any]]) -> str:
    return str((chunk or {}).get("main_text_ar") or "")


def _chunk_full_text(chunk: Optional[Dict[str, Any]]) -> str:
    """Full representation text including math/diagrams when available.

    Falls back to ``main_text_ar`` when the chunk has no structured enrichment
    (B2/B1 fixed-window chunks). Always returns the empty string for a ``None``
    chunk (cache miss).
    """
    if not chunk:
        return ""
    text = _chunk_text(chunk)
    if not text:
        return ""
    # Reconstruct only when the chunk carries fields that main_text_ar alone drops.
    if any([chunk.get("math_expressions"), chunk.get("diagrams"),
            chunk.get("unit_title_ar"), chunk.get("lesson_title_ar"),
            chunk.get("heading_ar")]):
        lines: list[str] = []
        if chunk.get("unit_title_ar"):
            lines.append(f"الوحدة: {chunk['unit_title_ar']}")
        if chunk.get("lesson_title_ar"):
            lines.append(f"الدرس: {chunk['lesson_title_ar']}")
        if chunk.get("lesson_title_en"):
            lines.append(f"Lesson: {chunk['lesson_title_en']}")
        if chunk.get("heading_ar"):
            lines.append(chunk["heading_ar"])
        if chunk.get("main_text_ar"):
            lines.append(chunk["main_text_ar"])
        for expr in chunk.get("math_expressions") or []:
            if expr:
                lines.append(f"المعادلة: {expr}")
        for diag in chunk.get("diagrams") or []:
            desc = diag.get("description") or ""
            labels = diag.get("labels") or []
            labels_str = "، ".join(str(l) for l in labels)
            lines.append(f"[شكل هندسي: {desc}. التسميات: {labels_str}]")
        return "\n".join(l for l in lines if l)
    return text


def _load_mapping_methods(path: Path, system_name: str) -> Dict[str, str]:
    """Index mapping_method by qa_id from a gold mapping file (reporting only)."""
    methods: Dict[str, str] = {}
    for row in schemas.read_jsonl(path):
        qa_id = str(row.get("qa_id") or "")
        if qa_id:
            methods[qa_id] = str(row.get("mapping_method") or "")
    return methods


def load_judge_corpus(config: cfg.JudgeExperimentConfig) -> JudgeCorpus:
    """Load the QA dataset, this system's retrieval records, and its chunk cache."""
    system = config.system
    qa_dir = Path(config.qa_dataset_dir)

    # 1) Frozen QA dataset — the source of truth for question + reference answer.
    qa_path = qa_dir / config.qa_dataset_filename
    qa_rows = schemas.read_jsonl(qa_path)
    if not qa_rows:
        raise FileNotFoundError(
            f"No QA dataset at {qa_path}. The judge reads the frozen dataset "
            "produced by 'python -m experiments.qa_dataset finalize'."
        )
    qa_by_id = {str(row.get("qa_id")): row for row in qa_rows if row.get("qa_id")}

    # 2) Retrieval-result artifacts for this system (the judge builds ON these).
    records_path = (
        Path(config.retrieval_eval_dir) / f"retrieval_records_{system.name}.jsonl"
    )
    records = schemas.read_jsonl(records_path)
    if not records:
        raise FileNotFoundError(
            f"No retrieval records at {records_path}. Run the retrieval evaluation "
            f"first, e.g. 'python -m experiments.retrieval_{system.name} "
            "--generate-answers' (add --generate-answers to score faithfulness and "
            "answer relevancy)."
        )

    # 3) Chunk cache — resolves retrieved ids back to the text that was embedded.
    chunks = schemas.load_chunks(system.chunks_path)
    chunks_by_id = {
        str(c["chunk_id"]): c for c in chunks if c.get("chunk_id")
    }

    # 4) Gold mapping — mapping_method for reporting only (never a judge input).
    mapping_methods = _load_mapping_methods(
        qa_dir / system.gold_mapping_filename, system.name
    )

    items: List[JudgeItem] = []
    any_generated = False
    for record in records:
        qa_id = str(record.get("qa_id") or "")
        if str(record.get("system") or "") not in ("", system.name):
            raise ValueError(
                f"{records_path} contains a record for system "
                f"{record.get('system')!r}, but this run judges {system.name!r}."
            )
        qa = qa_by_id.get(qa_id, {})
        item_warnings: List[str] = []
        if system.mapping_caveat:
            item_warnings.append(system.mapping_caveat)
        if not qa:
            item_warnings.append(
                f"qa_id {qa_id} is absent from the frozen QA dataset; "
                "question/reference taken from the retrieval record where possible"
            )

        retrieved_ids = [str(c) for c in (record.get("retrieved_chunk_ids") or [])]
        retrieved_scores = [
            float(s) for s in (record.get("retrieved_scores") or [])
        ]
        ranked_texts: List[Tuple[str, str]] = []
        for chunk_id in retrieved_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                item_warnings.append(
                    f"retrieved chunk {chunk_id} not found in {system.chunks_path}; "
                    "scored with empty text"
                )
            ranked_texts.append((chunk_id, _chunk_full_text(chunk)))

        texts, ctx_chunk_ids, token_count, truncated = assemble_judge_context(
            ranked_texts, config.context
        )
        # The answer generator's stored accounting includes rank/id headers and
        # separators. Reconstruct it for provenance even though chunk ids are not
        # rendered in the judge prompt.
        generator_context_text = "\n\n".join(
            f"[{rank}] {chunk_id} {text}".rstrip()
            for rank, (chunk_id, text) in enumerate(zip(ctx_chunk_ids, texts), start=1)
        )
        char_count = len(generator_context_text)
        est_tokens = (
            (char_count + config.context.chars_per_token - 1)
            // config.context.chars_per_token
        )

        # Fairness cross-check against the context the generator actually used.
        recorded_context_ids = [
            str(c) for c in (record.get("context_chunk_ids") or [])
        ]
        context_source = "reassembled_shared_budget"
        if recorded_context_ids and list(ctx_chunk_ids) != recorded_context_ids:
            item_warnings.append(
                "reconstructed context chunk set differs from the retrieval run's "
                f"recorded context_chunk_ids ({len(ctx_chunk_ids)} vs "
                f"{len(recorded_context_ids)} chunks)"
            )
        elif recorded_context_ids:
            # Matching IDs alone are not enough to claim the generator context
            # was reproduced: also validate the recorded budget accounting when
            # the retrieval artifact contains it.
            recorded_tokens = record.get("context_token_count")
            recorded_truncated = record.get("context_truncated")
            accounting_matches = True
            if recorded_tokens is not None and int(recorded_tokens) != token_count:
                accounting_matches = False
                item_warnings.append(
                    "reconstructed context token count differs from the retrieval "
                    f"record ({token_count} vs {recorded_tokens})"
                )
            if (
                recorded_truncated is not None
                and bool(recorded_truncated) != truncated
            ):
                accounting_matches = False
                item_warnings.append(
                    "reconstructed context truncation flag differs from the retrieval record"
                )
            if accounting_matches and config.context.prefer_recorded_context:
                context_source = "verified_recorded_generator_context"

        context = JudgeContext(
            context_texts=texts,
            chunk_ids=ctx_chunk_ids,
            token_count=token_count,
            budget_tokens=config.context.context_budget_tokens,
            truncated=truncated,
            char_count=char_count,
            estimated_model_tokens=est_tokens,
            source=context_source,
            warnings=tuple(item_warnings),
        )

        generated = str(record.get("generated_answer_ar") or "").strip()
        if generated:
            any_generated = True

        items.append(
            JudgeItem(
                qa_id=qa_id,
                system=system.name,
                question_ar=str(qa.get("question_ar") or ""),
                question_type=str(
                    qa.get("question_type") or record.get("question_type") or ""
                ),
                difficulty=str(qa.get("difficulty") or record.get("difficulty") or ""),
                answer_mode=str(
                    qa.get("answer_mode") or record.get("answer_mode") or ""
                ),
                required_diagram=bool(
                    qa.get("required_diagram")
                    if qa
                    else record.get("required_diagram")
                ),
                required_formula=bool(
                    qa.get("required_formula")
                    if qa
                    else record.get("required_formula")
                ),
                reference_answer_ar=str(
                    qa.get("answer_reference_ar")
                    or record.get("answer_reference_ar")
                    or ""
                ),
                generated_answer_ar=generated,
                answer_available=bool(generated),
                context=context,
                retrieved_chunk_ids=tuple(retrieved_ids),
                retrieved_scores=tuple(retrieved_scores),
                retrieved_ranks=tuple(range(1, len(retrieved_ids) + 1)),
                answer_provider=str(record.get("answer_provider") or ""),
                answer_model=str(record.get("answer_model") or ""),
                gold_granularity=str(
                    record.get("gold_granularity") or system.gold_granularity
                ),
                mapping_type=system.mapping_type,
                mapping_method=mapping_methods.get(qa_id, ""),
                mapping_status=str(record.get("mapping_status") or ""),
                mapping_caveat=system.mapping_caveat,
                warnings=tuple(item_warnings),
            )
        )

    logger.info(
        "Loaded judge corpus for %s: items=%d chunks=%d generated_answers=%s",
        system.name, len(items), len(chunks_by_id), any_generated,
    )
    return JudgeCorpus(
        system=system, items=items, generated_answers_present=any_generated
    )
