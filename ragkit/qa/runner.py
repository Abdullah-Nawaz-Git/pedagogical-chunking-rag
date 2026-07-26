"""
ragkit.qa.runner
════════════════

The QA-dataset orchestrator + CLI.


    select     Stage 1  — choose Proposed chunks → source_selection_plan.jsonl
    generate   Stages 2-3 — LLM candidate generation → qa_candidates.jsonl
    validate   Stage 4  — deterministic filtering → qa_validated.jsonl
    finalize   Stage 5  — freeze the 180-item dataset → qa_dataset_v1.*
    map-gold   Stage 6  — B1/B2/Proposed gold maps → gold_mapping_*.jsonl
    all        run select → generate → validate → finalize → map-gold

Common flags: ``--config`` (YAML/JSON overrides), ``--output-dir``, ``--seed``,
``--provider {mock,vertex}``, ``--force`` (regenerate), and ``--dry-run``
(compute + log, write nothing).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv

from .. import config as cfg
from . import (
    finalize,
    generation,
    gold_mapping,
    schemas,
    source_selection,
    validation,
)

logger = logging.getLogger("ragkit.qa")

# The pipeline stages, in execution order, for the ``all`` command.
_STAGE_ORDER = ("select", "generate", "validate", "finalize", "map-gold")


# ════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.qa_dataset",
        description=(
            "Prepare the frozen Arabic QA dataset from the Proposed pedagogical "
            "chunks, then map gold provenance onto the B1, B2, and Proposed corpora."
        ),
    )
    parser.add_argument(
        "command",
        choices=(*_STAGE_ORDER, "all"),
        help="Which pipeline stage to run (or 'all' for the full sequence).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON file overriding QAConfig defaults.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for all QA artifacts (defaults to QAConfig.output_dir).",
    )
    parser.add_argument(
        "--proposed-chunks",
        type=Path,
        default=None,
        help="Override path to the Proposed chunks.json (QA-generation source).",
    )
    parser.add_argument(
        "--b1-chunks", type=Path, default=None, help="Override path to the B1 chunks.json."
    )
    parser.add_argument(
        "--b2-chunks", type=Path, default=None, help="Override path to the B2 chunks.json."
    )
    parser.add_argument(
        "--provider",
        choices=("mock", "vertex"),
        default=None,
        help="Generation provider (default from config; 'mock' is offline/deterministic).",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides config)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For 'generate': ignore existing candidates and regenerate from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and log a stage without writing any files.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging."
    )
    return parser


def _resolve_config(args: argparse.Namespace) -> cfg.QAConfig:
    """Build the QAConfig from defaults, a file, and CLI overrides (CLI wins)."""
    from dataclasses import replace

    config = schemas.load_qa_config(args.config, seed=args.seed, provider=args.provider)
    overrides = {}
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    if args.proposed_chunks is not None:
        overrides["proposed_chunks_path"] = str(args.proposed_chunks)
    if args.b1_chunks is not None:
        overrides["b1_chunks_path"] = str(args.b1_chunks)
    if args.b2_chunks is not None:
        overrides["b2_chunks_path"] = str(args.b2_chunks)
    if overrides:
        config = replace(config, **overrides)
    return config


# ════════════════════════════════════════════
# STAGE DISPATCH
# ════════════════════════════════════════════


def _run_stage(
    stage: str,
    config: cfg.QAConfig,
    layout: schemas.QAOutputLayout,
    *,
    force: bool,
    dry_run: bool,
) -> None:
    """Run a single named stage."""
    if stage == "select":
        source_selection.run_source_selection(config, layout, dry_run=dry_run)
    elif stage == "generate":
        generation.run_generation(config, layout, force=force, dry_run=dry_run)
    elif stage == "validate":
        validation.run_validation(config, layout, dry_run=dry_run)
    elif stage == "finalize":
        finalize.run_finalize(config, layout, dry_run=dry_run)
    elif stage == "map-gold":
        gold_mapping.run_gold_mapping(config, layout, dry_run=dry_run)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"unknown stage: {stage}")


def run(argv: Optional[Sequence[str]] = None) -> int:
    """Parse args and execute the requested stage(s). Returns a process exit code."""
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = _resolve_config(args)
    layout = schemas.QAOutputLayout(config.output_dir)

    stages: List[str] = list(_STAGE_ORDER) if args.command == "all" else [args.command]

    logger.info("══════════════════════════════════════════════")
    logger.info("QA DATASET PIPELINE")
    logger.info("  command:        %s", args.command)
    logger.info("  output dir:     %s", config.output_dir)
    logger.info("  provider:       %s", config.provider)
    logger.info("  seed:           %d", config.random_seed)
    logger.info("  dry run:        %s", args.dry_run)
    logger.info("══════════════════════════════════════════════")

    try:
        for stage in stages:
            logger.info("── stage: %s ──", stage)
            _run_stage(stage, config, layout, force=args.force, dry_run=args.dry_run)
    except finalize.DeficitError as exc:
        logger.error("Finalize deficit: %s", exc)
        return 1
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if not args.dry_run:
        print(f"\nDone ({args.command}). Artifacts in {config.output_dir}/")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
