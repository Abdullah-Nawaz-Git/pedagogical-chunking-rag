"""
ragkit.judge.runner
═══════════════════

The LLM-as-judge orchestrator + CLI.

Mirrors ``ragkit.retrieval.runner``: a single ``run(system, argv)`` drives all
three systems, and the entry points under ``experiments/`` just build the CLI and
call it. Every decision is data-driven (dimension specs, mode, config) — never a
branch on the system name — so Proposed, B1, and B2 share one control flow.

Per system the loop is:

    load frozen corpus (QA + retrieval records + chunks) → for each item, for each
    eligible + not-yet-scored dimension, build the ISOLATED prompt → call the judge
    with bounded retries → parse → emit one atomic record.

All systems write into ONE resumable ledger (``judge_eval/judge_scores.jsonl``)
keyed by (qa_id, system, metric), so a rerun skips finished work and the paper's
cross-system paired comparison reads a single file. When more than one system is
scored, the runner also emits the combined summary + Markdown report and the
paired treatment-vs-control comparison.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

from .. import config as cfg
from . import client as client_mod
from . import corpus as corpus_mod
from . import metrics, parser as parser_mod, report, schemas, scoring

logger = logging.getLogger("ragkit.judge")


def _configure_llm_call_logger(path: Path) -> None:
    """Attach a dedicated JSONL file handler for detailed judge LLM call logs."""
    llm_logger = logging.getLogger("ragkit.judge.llm_calls")
    llm_logger.setLevel(logging.INFO)
    llm_logger.propagate = False

    resolved_target = path.resolve()
    for handler in llm_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", "")).resolve()
            if existing == resolved_target:
                return

    for handler in list(llm_logger.handlers):
        llm_logger.removeHandler(handler)
        handler.close()

    path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    llm_logger.addHandler(file_handler)


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════


def build_arg_parser(
    system: Optional[cfg.JudgeSystemConfig] = None,
) -> argparse.ArgumentParser:
    """Build the parser for a judge run.

    When ``system`` is given (the per-system entry points) the ``--system`` flag is
    fixed to it; the generic runner exposes it so ``all`` is available.
    """
    description = (
        f"ragkit LLM-as-judge evaluation — {system.label}"
        if system
        else "ragkit LLM-as-judge evaluation — Proposed / B1 / B2"
    )
    parser = argparse.ArgumentParser(description=description)

    if system is None:
        parser.add_argument(
            "--system",
            choices=(*cfg.JUDGE_SYSTEM_ORDER, "all"),
            default="all",
            help="Which system to judge (default: all three).",
        )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML/JSON file overriding JudgeExperimentConfig defaults.",
    )
    parser.add_argument(
        "--qa-dataset-dir", type=Path, default=None,
        help="Directory holding qa_dataset_v1.jsonl and the gold_mapping_* files.",
    )
    parser.add_argument(
        "--retrieval-eval-dir", type=Path, default=None,
        help="Directory holding the retrieval_records_<system>.jsonl artifacts.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for judge artifacts (default: judge_eval).",
    )
    parser.add_argument(
        "--provider",
        choices=(
            cfg.JUDGE_PROVIDER_MOCK,
            cfg.JUDGE_PROVIDER_VERTEX_ANTHROPIC,
            cfg.JUDGE_PROVIDER_VERTEX_GEMINI,
        ),
        default=None,
        help=(
            "Judge provider. 'mock' is offline/deterministic (smoke tests). "
            "'vertex_anthropic' (Claude) is the default real judge."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(
            cfg.JUDGE_MODE_AUTO,
            cfg.JUDGE_MODE_RETRIEVAL_ONLY,
            cfg.JUDGE_MODE_GENERATION,
        ),
        default=None,
        help=(
            "'retrieval_only' scores context recall/precision; 'generation' also "
            "scores faithfulness + answer relevancy; 'auto' picks per item."
        ),
    )
    parser.add_argument(
        "--dimensions", default=None,
        help="Comma-separated subset of judge dimensions (default: all).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Rescore every dimension even if a score already exists in the ledger.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Judge only the first N QA items per system (debugging aid).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute and log results without writing any files.",
    )
    parser.add_argument(
        "--recover-only", action="store_true",
        help=(
            "No judge calls. Load the existing ledger, recover the score from every "
            "truncated parse_error reply (reusing the saved raw_response), rewrite "
            "the ledger, and regenerate all summary/comparison artifacts."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging.")
    return parser


def _resolve_config(
    system: cfg.JudgeSystemConfig,
    args: argparse.Namespace,
) -> cfg.JudgeExperimentConfig:
    """Build the config from defaults, an optional file, then CLI overrides."""
    from dataclasses import replace

    config = schemas.load_judge_config(
        system,
        args.config,
        qa_dataset_dir=str(args.qa_dataset_dir) if args.qa_dataset_dir else None,
        retrieval_eval_dir=(
            str(args.retrieval_eval_dir) if args.retrieval_eval_dir else None
        ),
        output_dir=str(args.output_dir) if args.output_dir else None,
        mode=args.mode,
        resume=False if args.no_resume else None,
    )
    # The judge model is a nested knob; override it so the whole comparison uses
    # one provider (never a per-system confound).
    if args.provider is not None:
        config = replace(config, model=replace(config.model, provider=args.provider))
    if args.dimensions is not None:
        requested = tuple(
            d.strip() for d in args.dimensions.split(",") if d.strip()
        )
        unknown = [d for d in requested if d not in cfg.JUDGE_DIMENSIONS]
        if unknown:
            print(
                f"Unknown dimension(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(cfg.JUDGE_DIMENSIONS)}",
                file=sys.stderr,
            )
            sys.exit(2)
        config = replace(config, dimensions=requested)
    return config


# ════════════════════════════════════════════
# RESUMABLE LEDGER
# ════════════════════════════════════════════


def _load_ledger(layout: schemas.JudgeOutputLayout) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Read the existing merged ledger, keyed by (qa_id, system, metric).

    A previous score is only treated as resumable when it was a clean success or a
    legitimate skip; hard failures (parse/api/invalid) are dropped so a rerun
    retries them.
    """
    ledger: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if not layout.scores_jsonl.exists():
        return ledger
    for record in schemas.read_jsonl(layout.scores_jsonl):
        if str(record.get("judge_status")) in schemas.FAILURE_STATUSES:
            continue
        ledger[schemas.record_key(record)] = record
    return ledger


# ════════════════════════════════════════════
# PER-SYSTEM SCORING
# ════════════════════════════════════════════


def score_system(
    config: cfg.JudgeExperimentConfig,
    provider: client_mod.JudgeProvider,
    already_scored: Set[Tuple[str, str, str]],
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Score every eligible, not-yet-scored dimension for one system."""
    corpus = corpus_mod.load_judge_corpus(config)
    items = corpus.items[:limit] if limit else corpus.items

    if not corpus.generated_answers_present and config.mode == cfg.JUDGE_MODE_GENERATION:
        logger.warning(
            "%s: mode=generation but no generated answers were found in the "
            "retrieval records; faithfulness/answer_relevancy will all be recorded "
            "as skipped. Re-run retrieval with --generate-answers.",
            config.system.name,
        )

    new_records: List[Dict[str, Any]] = []
    for item in tqdm(items, desc=f"Judge — {config.system.name}"):
        new_records.extend(
            scoring.score_item(config, provider, item, already_scored)
        )
    return new_records


# ════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════


def run(
    system: Optional[cfg.JudgeSystemConfig] = None,
    argv: Optional[Sequence[str]] = None,
) -> int:
    """Parse args and judge the requested system(s). Returns an exit code."""
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
            [cfg.JUDGE_SYSTEMS[name] for name in cfg.JUDGE_SYSTEM_ORDER]
            if selected == "all"
            else [cfg.JUDGE_SYSTEMS[selected]]
        )

    configs = [_resolve_config(s, args) for s in systems]
    layout = schemas.JudgeOutputLayout(configs[0].output_dir)
    _configure_llm_call_logger(layout.llm_calls_jsonl)

    # No-LLM recovery/re-aggregation needs no provider and no environment.
    if args.recover_only:
        return _run_recover_only(
            configs,
            layout,
            tuple(cfg.JUDGE_DIMENSIONS),
            dry_run=args.dry_run,
        )

    # Fail fast on a missing environment before doing any work.
    problem = client_mod.check_environment(configs[0])
    if problem:
        client_mod.fail_fast(problem)

    dimensions = tuple(d for d in cfg.JUDGE_DIMENSIONS if d in configs[0].dimensions)

    logger.info("══════════════════════════════════════════════")
    logger.info("LLM-AS-JUDGE EVALUATION (inspired by RAGAS dimensions)")
    logger.info("  systems:        %s", ", ".join(c.system.name for c in configs))
    logger.info("  provider:       %s", configs[0].model.provider)
    logger.info("  judge model:    %s", client_mod.resolved_model_name(configs[0]))
    logger.info("  temperature:    %s", configs[0].model.temperature)
    logger.info("  mode:           %s", configs[0].mode)
    logger.info("  dimensions:     %s", ", ".join(dimensions))
    logger.info("  context budget: %d tokens", configs[0].context.context_budget_tokens)
    logger.info("  qa dataset:     %s", configs[0].qa_dataset_dir)
    logger.info("  retrieval dir:  %s", configs[0].retrieval_eval_dir)
    logger.info("  output dir:     %s", configs[0].output_dir)
    logger.info("  llm call log:   %s", layout.llm_calls_jsonl)
    logger.info("  resume:         %s", configs[0].resume)
    logger.info("  dry run:        %s", args.dry_run)
    logger.info("  NOTE: secondary evidence; retrieval-native metrics remain primary.")
    logger.info("══════════════════════════════════════════════")

    # One shared, resumable ledger across systems.
    ledger = _load_ledger(layout) if configs[0].resume else {}
    already_scored: Set[Tuple[str, str, str]] = set(ledger.keys())

    provider: Optional[client_mod.JudgeProvider] = None
    per_system_records: Dict[str, List[Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []

    try:
        provider = client_mod.build_judge_provider(configs[0])
        for config in configs:
            logger.info("── system: %s ──", config.system.name)
            new_records = score_system(
                config, provider, already_scored, limit=args.limit
            )
            # Merge fresh scores into the ledger (fresh always wins over a prior
            # non-failure record for the same key, e.g. a forced --no-resume rerun).
            for record in new_records:
                ledger[schemas.record_key(record)] = record
                already_scored.add(schemas.record_key(record))

            # This system's full record set (prior + new) for its summary.
            system_records = [
                r for r in ledger.values() if r.get("system") == config.system.name
            ]
            summary = metrics.build_summary(
                system_records, config, dimensions=dimensions
            )
            report.log_summary(config, summary, dimensions)
            summaries.append(summary)
            per_system_records[config.system.name] = system_records
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    if args.dry_run:
        total_new = sum(
            1 for k in already_scored if k in {schemas.record_key(r) for r in _flatten(per_system_records)}
        )
        logger.info(
            "[dry-run] computed scores for %d system(s); wrote nothing",
            len(summaries),
        )
        return 0

    _write_all(layout, configs, ledger, summaries, per_system_records, dimensions)
    print(f"\nDone. Judge artifacts in {configs[0].output_dir}/")
    return 0


def _run_recover_only(
    configs: Sequence[cfg.JudgeExperimentConfig],
    layout: schemas.JudgeOutputLayout,
    dimensions: Sequence[str],
    *,
    dry_run: bool,
) -> int:
    """Re-process an existing ledger without any judge calls.

    Loads the raw ``judge_scores.jsonl`` verbatim (including rows that were
    ``parse_error``), attempts to recover a numeric score from every truncated
    reply via ``parser.parse_judge_response`` (the raw response is preserved on
    the row), then rewrites the ledger, summaries, and comparisons.
    """
    ledger: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    recovered = 0
    remaining_errors = 0
    for record in schemas.read_jsonl(layout.scores_jsonl):
        if str(record.get("judge_status")) == schemas.STATUS_PARSE_ERROR:
            parsed = parser_mod.parse_judge_response(
                record.get("raw_response") or ""
            )
            if parsed.status == schemas.STATUS_RECOVERED and parsed.score is not None:
                record["score"] = parsed.score
                record["judge_status"] = parsed.status
                record["parse_ok"] = True
                record["warnings"] = list(record.get("warnings") or []) + [
                    parsed.error
                ]
                recovered += 1
            else:
                remaining_errors += 1
        ledger[schemas.record_key(record)] = record

    logger.info(
        "recover-only: recovered %d truncated score(s); %d unresolved error(s)",
        recovered,
        remaining_errors,
    )

    per_system_records: Dict[str, List[Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []
    for config in configs:
        system_records = [
            r for r in ledger.values() if r.get("system") == config.system.name
        ]
        summary = metrics.build_summary(
            system_records, config, dimensions=dimensions
        )
        report.log_summary(config, summary, dimensions)
        summaries.append(summary)
        per_system_records[config.system.name] = system_records

    if dry_run:
        logger.info("[dry-run] recover-only computed %d system(s); wrote nothing",
                    len(summaries))
        return 0

    _write_all(layout, configs, ledger, summaries, per_system_records, dimensions)
    print(f"\nDone. Recovered {recovered} score(s); artifacts in {configs[0].output_dir}/")
    return 0


def _flatten(mapping: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for records in mapping.values():
        out.extend(records)
    return out


def _write_all(
    layout: schemas.JudgeOutputLayout,
    configs: Sequence[cfg.JudgeExperimentConfig],
    ledger: Dict[Tuple[str, str, str], Dict[str, Any]],
    summaries: Sequence[Dict[str, Any]],
    per_system_records: Dict[str, List[Dict[str, Any]]],
    dimensions: Sequence[str],
) -> None:
    """Write the merged ledger, per-config snapshots, summary, and comparison."""
    layout.ensure()

    # The full merged ledger (JSONL + CSV), deterministically ordered.
    all_records = sorted(ledger.values(), key=schemas.sort_key)
    schemas.write_jsonl(layout.scores_jsonl, all_records)
    schemas.write_csv(layout.scores_csv, all_records, schemas.RECORD_COLUMNS)

    # A frozen copy of each system's resolved config, for provenance.
    for config in configs:
        schemas.write_json(
            layout.config_used(config.system.name), schemas.config_to_dict(config)
        )

    # Paired treatment-vs-control comparisons (the paper's key result). Computed
    # only when both members of a pair were actually scored in this ledger.
    comparisons: List[Dict[str, Any]] = []
    treatment_name, control_name = configs[0].aggregation.primary_pair
    if treatment_name in per_system_records and control_name in per_system_records:
        comparisons.append(
            metrics.compare_systems(
                per_system_records[treatment_name],
                per_system_records[control_name],
                dimensions,
                configs[0].aggregation,
                treatment_name=treatment_name,
                control_name=control_name,
            )
        )

    # Combined summary object (all systems + paired comparisons + caveats).
    comparison = report.build_comparison(summaries, comparisons, dimensions)
    schemas.write_json(layout.summary_json, comparison)
    report.write_comparison(layout, summaries, comparisons, dimensions)

    # A tiny run manifest for quick provenance lookup.
    schemas.write_json(
        layout.run_manifest,
        {
            "evaluation_kind": "llm_as_judge_inspired_by_ragas",
            "systems": [c.system.name for c in configs],
            "provider": configs[0].model.provider,
            "judge_model": client_mod.resolved_model_name(configs[0]),
            "mode": configs[0].mode,
            "dimensions": list(dimensions),
            "context_budget_tokens": configs[0].context.context_budget_tokens,
            "total_score_records": len(all_records),
            "note": (
                "Secondary, model-dependent evidence. Retrieval-native metrics in "
                "retrieval_eval/ remain primary."
            ),
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(None, argv)


if __name__ == "__main__":
    sys.exit(main())
