"""
ragkit.retrieval.runner
═══════════════════════

The retrieval-evaluation orchestrator + CLI.

A single ``run(system_config, argv)`` drives all three systems, exactly as
``ragkit.pipeline.run`` drives all three ingestion experiments: the entry points
under ``experiments/`` build a config and call ``run``. Every branch it takes is
decided by configuration data, never by the system's name, so Proposed, B1, and
B2 share one control flow.

Per question the loop is:

    embed query (shared model) → retrieve top-k (no filter) → score against that
    system's gold mapping → assemble context under the shared budget → optionally
    generate an answer → emit one record.

Outputs land in ``retrieval_eval/`` (per-question JSONL + CSV, a compact summary
JSON, and a Markdown report), plus a cross-system comparison when more than one
system is evaluated in a run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm import tqdm

from .. import config as cfg
from . import answer as answer_mod
from . import corpus as corpus_mod
from . import engine as engine_mod
from . import metrics, report, schemas

logger = logging.getLogger("ragkit.retrieval")


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════


def build_arg_parser(system: Optional[cfg.RetrievalSystemConfig] = None) -> argparse.ArgumentParser:
    """Build the parser for a retrieval-evaluation run.

    When ``system`` is given (the per-system entry points) the ``--system`` flag is
    fixed to that system; the generic runner exposes it so ``all`` is available.
    """
    description = (
        f"ragkit retrieval evaluation — {system.label or system.name}"
        if system
        else "ragkit retrieval evaluation — Proposed / B1 / B2"
    )
    parser = argparse.ArgumentParser(description=description)

    if system is None:
        parser.add_argument(
            "--system",
            choices=(*cfg.RETRIEVAL_SYSTEM_ORDER, "all"),
            default="all",
            help="Which system to evaluate (default: all three).",
        )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML/JSON file overriding RetrievalExperimentConfig defaults.",
    )
    parser.add_argument(
        "--qa-dataset-dir", type=Path, default=None,
        help="Directory holding qa_dataset_v1.jsonl and the gold_mapping_* files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for retrieval artifacts (default: retrieval_eval).",
    )
    parser.add_argument(
        "--retriever",
        choices=(engine_mod.RETRIEVER_PINECONE, engine_mod.RETRIEVER_LOCAL),
        default=None,
        help=(
            "'pinecone' queries the real per-system index; 'local' scores the "
            "cached chunks offline (deterministic, no credentials) for smoke tests."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Chunks retrieved per question (held equal across systems; default 5).",
    )
    parser.add_argument(
        "--generate-answers", action="store_true",
        help="Also run the shared answer generator (same model/prompt for all systems).",
    )
    parser.add_argument(
        "--answer-provider", choices=(answer_mod.PROVIDER_MOCK, answer_mod.PROVIDER_VERTEX),
        default=None,
        help="Answer-generation provider ('mock' is offline/deterministic).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Evaluate only the first N QA items (debugging aid).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and log results without writing any files.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def _resolve_config(
    system: cfg.RetrievalSystemConfig,
    args: argparse.Namespace,
) -> cfg.RetrievalExperimentConfig:
    """Build the config from defaults, an optional file, then CLI overrides."""
    from dataclasses import replace

    config = schemas.load_retrieval_config(
        system,
        args.config,
        qa_dataset_dir=str(args.qa_dataset_dir) if args.qa_dataset_dir else None,
        output_dir=str(args.output_dir) if args.output_dir else None,
        retriever=args.retriever,
        generate_answers=True if args.generate_answers else None,
    )
    # Nested knobs need explicit replacement so the dataclasses stay frozen.
    if args.top_k is not None:
        if args.top_k < 1:
            print("--top-k must be >= 1", file=sys.stderr)
            sys.exit(2)
        hit_ks = tuple(k for k in config.retrieval.hit_at_ks if k <= args.top_k)
        # Always report Hit@top_k itself, even for an unusual --top-k.
        if args.top_k not in hit_ks:
            hit_ks = tuple(sorted({*hit_ks, args.top_k}))
        config = replace(
            config,
            retrieval=replace(config.retrieval, top_k=args.top_k, hit_at_ks=hit_ks),
        )
    if args.answer_provider is not None:
        config = replace(
            config, answer=replace(config.answer, provider=args.answer_provider)
        )
    return config


# ════════════════════════════════════════════
# PER-QUESTION RECORD
# ════════════════════════════════════════════


def _build_record(
    config: cfg.RetrievalExperimentConfig,
    qa: Dict[str, Any],
    retrieved: Sequence[engine_mod.RetrievedChunk],
    gold: Optional[corpus_mod.GoldTargets],
    context: answer_mod.AssembledContext,
    answer_result: Optional[answer_mod.AnswerResult],
) -> Dict[str, Any]:
    """Assemble one record in the shared output schema."""
    system = config.system
    scored = metrics.score_question(retrieved, gold, config)

    warnings: List[str] = []
    # The system-level caveat (B1 only) is carried on EVERY record so no row can
    # be read out of context.
    if system.proxy_disclaimer:
        warnings.append(system.proxy_disclaimer)
    if gold is None:
        warnings.append("no gold mapping found for this qa_id; scored as a miss")
    else:
        warnings.extend(gold.warnings)
    if scored.get("gold_unresolved_target_ids"):
        warnings.append(
            "gold mapping resolved no chunk for target(s): "
            + ", ".join(scored["gold_unresolved_target_ids"])
        )
    if context.truncated:
        warnings.append(
            f"context truncated at the shared budget of {context.budget_tokens} tokens"
        )

    record: Dict[str, Any] = {
        "qa_id": qa.get("qa_id"),
        "system": system.name,
        "question_type": qa.get("question_type"),
        "difficulty": qa.get("difficulty"),
        "answer_mode": qa.get("answer_mode"),
        "required_diagram": bool(qa.get("required_diagram")),
        "required_formula": bool(qa.get("required_formula")),
        "retrieved_chunk_ids": [hit.chunk_id for hit in retrieved],
        "retrieved_scores": [hit.score for hit in retrieved],
        "retrieved_ranks_relevant": scored["retrieved_ranks_relevant"],
        "first_gold_rank": scored["first_gold_rank"],
        "reciprocal_rank": scored["reciprocal_rank"],
        # Gold semantics travel with the row, not just the summary.
        "gold_granularity": system.gold_granularity,
        "gold_recall_kind": schemas.recall_kind_for(system),
        "gold_recall": scored["gold_recall"],
        "gold_targets_total": scored["gold_targets_total"],
        "gold_targets_covered": scored["gold_targets_covered"],
        "gold_target_ids": list(gold.target_ids) if gold else [],
        "gold_relevant_chunk_ids": list(gold.all_relevant_chunk_ids) if gold else [],
        "gold_missed_target_ids": scored["gold_missed_target_ids"],
        "mapping_status": gold.mapping_status if gold else "missing",
        "context_chunk_ids": list(context.chunk_ids),
        "context_token_count": context.token_count,
        "context_token_budget": context.budget_tokens,
        "context_truncated": context.truncated,
        "context_char_count": context.char_count,
        "context_estimated_model_tokens": context.estimated_model_tokens,
        "answer_reference_ar": qa.get("answer_reference_ar", ""),
        "generated_answer_ar": answer_result.answer_ar if answer_result else "",
        "answer_provider": answer_result.provider if answer_result else "",
        "answer_model": answer_result.model if answer_result else "",
        "answer_prompt_version": answer_result.prompt_version if answer_result else "",
        "answer_error": (answer_result.error or "") if answer_result else "",
        "warnings": warnings,
    }
    for k in config.retrieval.hit_at_ks:
        key = f"hit@{k}"
        record[key] = bool(scored[key])
    return record


# ════════════════════════════════════════════
# EVALUATION LOOP
# ════════════════════════════════════════════


def evaluate_system(
    config: cfg.RetrievalExperimentConfig,
    *,
    limit: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate one system end-to-end and return (records, summary)."""
    system = config.system
    corpus = corpus_mod.load_corpus(config)

    qa_items = corpus.qa_items[:limit] if limit else corpus.qa_items
    embedder = engine_mod.build_query_embedder(config)
    retriever = engine_mod.build_retriever(config, corpus.chunks_by_id)
    provider = answer_mod.build_answer_provider(config) if config.generate_answers else None

    # Embed every query up front so the embedding model is called in batches,
    # exactly as ingestion embedded documents.
    questions = [str(qa.get("question_ar") or "") for qa in qa_items]
    vectors = embedder.embed(questions)

    records: List[Dict[str, Any]] = []
    for qa, question, vector in tqdm(
        list(zip(qa_items, questions, vectors)),
        desc=f"Retrieval eval — {system.name}",
    ):
        retrieved = retriever.retrieve(question, vector)
        gold = corpus.gold(str(qa.get("qa_id")))
        context = answer_mod.assemble_context(retrieved, config.answer)
        result = provider.generate(question, context) if provider else None
        records.append(_build_record(config, qa, retrieved, gold, context, result))

    summary = metrics.build_summary(
        records,
        config,
        extra={
            "qa_dataset": str(Path(config.qa_dataset_dir) / config.qa_dataset_filename),
            "chunks_path": system.chunks_path,
            "corpus_chunk_count": len(corpus.chunks_by_id),
        },
    )
    return records, summary


def run_system(
    config: cfg.RetrievalExperimentConfig,
    layout: schemas.RetrievalOutputLayout,
    *,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Evaluate one system and write its artifacts."""
    records, summary = evaluate_system(config, limit=limit)
    report.log_summary(config, summary)
    if dry_run:
        logger.info("[dry-run] %s: computed %d records, wrote nothing", config.system.name, len(records))
        return summary
    report.write_system_outputs(config, layout, records, summary)
    return summary


# ════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════


def run(
    system: Optional[cfg.RetrievalSystemConfig] = None,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Parse args and evaluate the requested system(s). Returns an exit code."""
    load_dotenv()
    args = build_arg_parser(system).parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if system is not None:
        systems = [system]
    else:
        selected = getattr(args, "system", "all")
        systems = (
            [cfg.RETRIEVAL_SYSTEMS[name] for name in cfg.RETRIEVAL_SYSTEM_ORDER]
            if selected == "all"
            else [cfg.RETRIEVAL_SYSTEMS[selected]]
        )

    configs = [_resolve_config(s, args) for s in systems]
    layout = schemas.RetrievalOutputLayout(configs[0].output_dir)

    # Fail fast on a missing environment before doing any work.
    problem = engine_mod.check_environment(configs[0])
    if problem:
        engine_mod.fail_fast(problem)

    logger.info("══════════════════════════════════════════════")
    logger.info("RETRIEVAL EVALUATION")
    logger.info("  systems:        %s", ", ".join(c.system.name for c in configs))
    logger.info("  retriever:      %s", configs[0].retriever)
    logger.info("  top_k:          %d", configs[0].retrieval.top_k)
    logger.info("  metadata filter: %s", configs[0].retrieval.use_metadata_filter)
    logger.info("  qa dataset:     %s", configs[0].qa_dataset_dir)
    logger.info("  output dir:     %s", configs[0].output_dir)
    logger.info("  answers:        %s", configs[0].generate_answers)
    logger.info("  dry run:        %s", args.dry_run)
    logger.info("══════════════════════════════════════════════")

    summaries: List[Dict[str, Any]] = []
    try:
        for config in configs:
            logger.info("── system: %s ──", config.system.name)
            summaries.append(
                run_system(config, layout, limit=args.limit, dry_run=args.dry_run)
            )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    # A comparison only makes sense across more than one system.
    if len(summaries) > 1 and not args.dry_run:
        report.write_comparison(layout, summaries, configs[0])

    if not args.dry_run:
        print(f"\nDone. Retrieval artifacts in {configs[0].output_dir}/")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(None, argv)


if __name__ == "__main__":
    sys.exit(main())
