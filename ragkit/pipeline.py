"""
ragkit.pipeline
═══════════════

The orchestrator: render → extract → represent → chunk → embed → upsert.

A single ``run(config, argv)`` drives all four experiments. The branches it
takes are decided entirely by the ``ExperimentConfig`` passed in — which
extraction method, which chunker, which Pinecone index — so the experiment
files under ``experiments/`` are thin: build a config, call ``run``.

This keeps the variables under study explicit: every experiment shares the same
rendering, embedding, indexing, and (where applicable) extraction/representation
code, and differs only where the comparison intends it to.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

import cloudinary

from google import genai
from pinecone import Pinecone

from . import config as cfg
from .cache import CacheLayout


# ════════════════════════════════════════════
# CLI ARGUMENTS
# ════════════════════════════════════════════


@dataclass
class RunArgs:
    """Parsed command-line arguments for a single pipeline run."""

    pdf: Path
    semester: int
    cache_dir: Path
    start_page: int = 1
    end_page: Optional[int] = None
    # Only meaningful for experiments that use diagrams (proposed, B3).
    skip_bbox: bool = False
    chunk_only: bool = False
    embed_only: bool = False
    stop_after_texts: bool = False
    # Regenerate chunk/provenance artifacts only: reuse cached extractions
    # (no re-render/re-extract) and stop before embedding + Pinecone upsert.
    provenance_only: bool = False


def build_arg_parser(config: cfg.ExperimentConfig) -> argparse.ArgumentParser:
    """Build the argument parser for an experiment.

    The diagram-producing experiments (proposed, B2) additionally expose
    ``--skip-bbox`` and ``--chunk-only``; the OCR baselines do not.
    """
    parser = argparse.ArgumentParser(
        description=f"ragkit experiment '{config.name}' — "
                    f"{config.extraction} extraction + {config.chunking} chunking"
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Path to the PDF file")
    parser.add_argument("--semester", required=True, type=int, choices=[1, 2], help="Semester 1 or 2")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(config.default_cache_dir),
        help="Cache directory",
    )
    parser.add_argument("--start-page", type=int, default=1, help="First PDF page (1-based, inclusive)")
    parser.add_argument("--end-page", type=int, default=None, help="Last PDF page (1-based, inclusive)")
    parser.add_argument("--max-pages", type=int, default=None, help="Process at most this many pages from --start-page")
    parser.add_argument(
        "--stop-after-texts",
        action="store_true",
        help="Stop after writing embedding_texts.jsonl/chunks.json; skip Stage 3B embedding and Stage 3C Pinecone upsert",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help=(
            "Skip render/extract/chunk stages and load cached chunks.json + "
            "embedding_texts.jsonl, then run Stage 3B embedding and Stage 3C Pinecone upsert only"
        ),
    )
    parser.add_argument(
        "--provenance-only",
        action="store_true",
        help=(
            "Regenerate chunk + *_provenance.jsonl artifacts from CACHED extractions "
            "without re-running Gemini/OCR, and stop before embedding + Pinecone upsert. "
            "For Gemini experiments this implies --chunk-only; it always implies "
            "--stop-after-texts. Use it to (re)build retrieval-eval provenance cheaply."
        ),
    )

    if config.uses_diagrams:
        parser.add_argument("--skip-bbox", action="store_true", help="Skip cropping/upload stages")
        parser.add_argument(
            "--chunk-only", action="store_true",
            help="Skip stages 1-1.6 and start from cached extractions",
        )
    return parser


def parse_args(config: cfg.ExperimentConfig, argv: Optional[Sequence[str]] = None) -> RunArgs:
    """Parse and validate CLI args, exiting with code 2 on invalid input."""
    parser = build_arg_parser(config)
    args = parser.parse_args(argv)

    if args.start_page < 1:
        print("--start-page must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.max_pages is not None:
        if args.max_pages < 1:
            print("--max-pages must be >= 1", file=sys.stderr)
            sys.exit(2)
        args.end_page = args.start_page + args.max_pages - 1
    if args.end_page is not None and args.end_page < args.start_page:
        print("--end-page must be >= --start-page", file=sys.stderr)
        sys.exit(2)
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(2)

    provenance_only = bool(getattr(args, "provenance_only", False))
    chunk_only = bool(getattr(args, "chunk_only", False))
    embed_only = bool(getattr(args, "embed_only", False))
    stop_after_texts = bool(getattr(args, "stop_after_texts", False))

    if embed_only and (provenance_only or stop_after_texts):
        print("--embed-only cannot be combined with --provenance-only or --stop-after-texts", file=sys.stderr)
        sys.exit(2)

    # --provenance-only is a convenience switch: rebuild chunk + provenance
    # artifacts from cached extractions without re-running the expensive
    # extraction stages, and never touch embeddings or Pinecone. For Gemini
    # experiments that means skipping stages 1-1.6 (chunk_only); for every
    # experiment it means stopping after texts.
    if provenance_only:
        chunk_only = True
        stop_after_texts = True

    return RunArgs(
        pdf=args.pdf,
        semester=args.semester,
        cache_dir=args.cache_dir,
        start_page=args.start_page,
        end_page=args.end_page,
        skip_bbox=bool(getattr(args, "skip_bbox", False)),
        chunk_only=chunk_only,
        embed_only=embed_only,
        stop_after_texts=stop_after_texts,
        provenance_only=provenance_only,
    )


# ════════════════════════════════════════════
# ENVIRONMENT
# ════════════════════════════════════════════


@dataclass
class Environment:
    """Resolved environment variable values for a run."""

    gcp_project: str
    gcp_location: str
    gcp_credentials: Optional[str]
    pinecone_api_key: str
    pinecone_index_name: str
    cloudinary_cloud_name: Optional[str] = None
    cloudinary_api_key: Optional[str] = None
    cloudinary_api_secret: Optional[str] = None


def _resolve_environment(config: cfg.ExperimentConfig, args: RunArgs) -> Environment:
    """Read + validate the environment variables this experiment needs.

    Exits with code 2 when a required variable is missing.
    """
    env = config.env
    gcp_credentials     = os.environ.get(env.google_credentials)
    gcp_project         = os.environ.get(env.google_project)
    gcp_location        = os.environ.get(env.google_location, env.google_location_default)
    pinecone_api_key    = os.environ.get(env.pinecone_api_key)
    pinecone_index_name = os.environ.get(config.index_env_var, config.default_index_name)
    cloudinary_cloud_name = os.environ.get(env.cloudinary_cloud_name)
    cloudinary_api_key    = os.environ.get(env.cloudinary_api_key)
    cloudinary_api_secret = os.environ.get(env.cloudinary_api_secret)

    # GCP creds + project + Pinecone key are always required.
    missing = [
        name for name, val in [
            (env.google_credentials, gcp_credentials),
            (env.google_project, gcp_project),
            (env.pinecone_api_key, pinecone_api_key),
        ] if not val
    ]
    if missing:
        print(f"Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    credentials_path = Path(gcp_credentials)
    if not credentials_path.exists():
        print(
            f"{env.google_credentials} points to a missing file: {credentials_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    # Cloudinary creds are only needed when the crop/upload stage actually runs.
    if config.uses_diagrams and not args.skip_bbox and not args.chunk_only:
        cl_missing = [
            name for name, val in [
                (env.cloudinary_cloud_name, cloudinary_cloud_name),
                (env.cloudinary_api_key, cloudinary_api_key),
                (env.cloudinary_api_secret, cloudinary_api_secret),
            ] if not val
        ]
        if cl_missing:
            print(
                f"Missing Cloudinary env vars: {', '.join(cl_missing)} (or pass --skip-bbox)",
                file=sys.stderr,
            )
            sys.exit(2)

    return Environment(
        gcp_project=gcp_project,
        gcp_location=gcp_location,
        gcp_credentials=gcp_credentials,
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
        cloudinary_cloud_name=cloudinary_cloud_name,
        cloudinary_api_key=cloudinary_api_key,
        cloudinary_api_secret=cloudinary_api_secret,
    )


# ════════════════════════════════════════════
# EXTRACTION + CHUNKING BRANCHES
# ════════════════════════════════════════════


def _run_gemini_extraction(
    config: cfg.ExperimentConfig,
    args: RunArgs,
    layout: CacheLayout,
    vertex_client: genai.Client,
) -> Dict[str, Dict[str, int]]:
    """Stages 1 / 1.5 / 1.6 for the Gemini-extraction experiments (proposed, B3)."""
    from .render import render_pdf_pages
    from .extract import gemini as gemini_extract

    page_dimensions: Dict[str, Dict[str, int]] = {}
    if args.chunk_only:
        logging.info("Stages 1-1.6 skipped (--chunk-only)")
    else:
        logging.info("Stage 1: rendering PDF pages at %d DPI", config.render.dpi)
        page_dimensions = render_pdf_pages(
            args.pdf, layout.pages, config.render.dpi,
            start_page=args.start_page, end_page=args.end_page,
        )

        logging.info("Stage 1.5: structured extraction with %s", config.gemini.flash_model)
        gemini_extract.extract_pages(
            vertex_client, layout.pages, layout.extractions, layout.bboxes,
            layout.root, config.gemini,
            start_page=args.start_page, end_page=args.end_page,
        )
        gemini_extract.normalize_extractions_metadata(
            layout.extractions, start_page=args.start_page, end_page=args.end_page,
        )

        if args.skip_bbox:
            logging.info("Stage 1.6 skipped (--skip-bbox)")
        else:
            logging.info("Stage 1.6: cropping diagrams and uploading to Cloudinary")
            gemini_extract.crop_and_upload(
                layout.pages, layout.bboxes, layout.diagrams, layout.diagram_urls,
                page_dimensions, args.semester, layout.root,
                start_page=args.start_page, end_page=args.end_page,
            )

    # Re-run metadata normalisation (covers the --chunk-only path with stale data).
    updated_pages = gemini_extract.normalize_extractions_metadata(
        layout.extractions, start_page=args.start_page, end_page=args.end_page,
    )
    logging.info("Metadata normalisation: updated %d pages", updated_pages)
    return page_dimensions


def _chunk_pedagogical(
    config: cfg.ExperimentConfig, args: RunArgs, layout: CacheLayout,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Proposed chunker → embeddable chunks + their proposed-representation texts."""
    from .chunk import pedagogical
    from .represent import build_embedding_text, write_embedding_text_cache

    logging.info("Stage 2: building pedagogical chunks")
    chunks = pedagogical.build_chunks(
        layout.extractions, layout.diagram_urls, layout.root, args.semester,
        start_page=args.start_page, end_page=args.end_page,
    )

    # Chunks of type "other" (title pages, answer keys, etc.) are excluded from
    # embedding because they don't carry instructional content.
    embeddable = [c for c in chunks if c.get("content_type") != "other"]
    other_count = len(chunks) - len(embeddable)
    logging.info(
        "Chunks: total=%d  embeddable=%d  other(skipped)=%d",
        len(chunks), len(embeddable), other_count,
    )
    if not embeddable:
        return [], []

    logging.info("Stage 3A: building embedding text strings")
    texts = [build_embedding_text(c, layout.root, config.representation) for c in embeddable]
    cache_path = write_embedding_text_cache(layout.root, embeddable, texts)
    logging.info("Stage 3A: cached embedding texts to %s", cache_path)

    # Persist proposed-chunk provenance locally, keyed by chunk_id. A pedagogical
    # chunk consumes each contributing block in full (coverage 1.0).
    prov_path = _write_provenance_jsonl(
        layout.root,
        "proposed_provenance.jsonl",
        embeddable,
        ["source_block_ids", "source_page_numbers", "source_block_coverage"],
    )
    logging.info("Stage 3A: wrote proposed provenance to %s", prov_path)

    return embeddable, texts


def _chunk_fixed_stream(
    config: cfg.ExperimentConfig, args: RunArgs, layout: CacheLayout,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """B2 chunker — fixed windows over the proposed representation stream."""
    from .chunk import fixed
    from .represent import build_representation_stream

    logging.info("Stage B2.A: building proposed-style representation stream")
    stream, page_meta, source_block_total_units, block_meta = build_representation_stream(
        layout.extractions, layout.diagram_urls, layout.root, args.semester,
        config.representation, start_page=args.start_page, end_page=args.end_page,
    )
    logging.info(
        "Stage B2.A: representation stream has %d tokens across %d pages (%d source blocks)",
        len(stream), len(page_meta), len(source_block_total_units),
    )
    if not stream:
        return [], []

    logging.info(
        "Stage B2.B: fixed chunking (size=%d, overlap=%d tokens)",
        config.fixed_chunk.chunk_size_tokens, config.fixed_chunk.overlap_tokens,
    )
    raw_chunks = fixed.fixed_token_chunks(
        stream, config.fixed_chunk, source_block_total_units,
    )
    records = fixed.make_stream_records(
        raw_chunks, page_meta, block_meta, args.semester,
        chunk_id_prefix=config.chunk_id_prefix, content_type=config.chunk_content_type,
    )
    logging.info("Stage B2.B: produced %d fixed-window chunks", len(records))

    # Persist full B2 window provenance locally, keyed by chunk_id. Kept out of
    # Pinecone metadata to avoid per-vector metadata-size limits.
    prov_path = _write_provenance_jsonl(
        layout.root,
        "b2_provenance.jsonl",
        records,
        [
            "page_range",
            "whitespace_unit_count",
            "source_page_numbers",
            "source_block_ids",
            "source_block_unit_counts",
            "source_block_coverage",
        ],
    )
    logging.info("Stage B2.B: wrote B2 provenance to %s", prov_path)

    texts = [c["main_text_ar"] for c in records]
    return records, texts


def _chunk_fixed_ocr(
    config: cfg.ExperimentConfig, args: RunArgs, layout: CacheLayout,
    page_texts: Dict[int, str],
) -> tuple[List[Dict[str, Any]], List[str]]:
    """B1 chunker — fixed windows over the raw OCR token stream."""
    from .chunk import fixed

    logging.info(
        "Stage B1.2: fixed chunking (size=%d, overlap=%d tokens)",
        config.fixed_chunk.chunk_size_tokens, config.fixed_chunk.overlap_tokens,
    )
    stream = fixed.build_ocr_stream(page_texts)
    logging.info("Stage B1.2: OCR token stream has %d tokens", len(stream))
    raw_chunks = fixed.fixed_token_chunks(stream, config.fixed_chunk)
    records = fixed.make_ocr_records(
        raw_chunks, args.semester,
        chunk_id_prefix=config.chunk_id_prefix, content_type=config.chunk_content_type,
    )
    logging.info("Stage B1.2: produced %d fixed-window chunks", len(records))

    # Persist B1 provenance locally. Raw OCR has no pedagogical blocks, so this
    # is PAGE-level only — gold-unit matching for B1 must map page -> unit
    # externally using the same page metadata the other experiments share.
    prov_path = _write_provenance_jsonl(
        layout.root,
        "b1_provenance.jsonl",
        records,
        ["page_range", "whitespace_unit_count", "source_page_numbers", "source_page_unit_counts"],
    )
    logging.info("Stage B1.2: wrote B1 provenance to %s", prov_path)

    texts = [c["main_text_ar"] for c in records]
    return records, texts


def _write_provenance_jsonl(
    cache_root: Path,
    filename: str,
    chunks: List[Dict[str, Any]],
    fields: Sequence[str],
) -> Path:
    """
    Write per-chunk retrieval-evaluation provenance to ``<cache_root>/<filename>``.

    Each line is a JSON object keyed by ``chunk_id`` plus the requested
    provenance ``fields`` (only those present on the chunk are written). Living
    in the experiment's own cache directory keeps each run's provenance beside
    its ``chunks.json`` and prevents one experiment from overwriting another's.

    This file is the source of truth for retrieval-evaluation gold matching
    (Hit@k, MRR, Gold Unit Recall): it is kept separate from Pinecone metadata
    so the large per-chunk provenance dicts never risk the per-vector
    metadata-size limit.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    out_path = cache_root / filename
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            record: Dict[str, Any] = {"chunk_id": ch.get("chunk_id")}
            for fld in fields:
                if fld in ch:
                    record[fld] = ch[fld]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _write_chunk_and_text_artifacts(
    cache_root: Path,
    chunks: List[Dict[str, Any]],
    texts: List[str],
) -> None:
    """Write chunks + embedding text artifacts for offline debugging."""
    from .represent import write_embedding_text_cache

    chunks_path = cache_root / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logging.info("Debug artifact: wrote %d chunks to %s", len(chunks), chunks_path)

    text_cache_path = write_embedding_text_cache(cache_root, chunks, texts)
    logging.info(
        "Debug artifact: wrote %d embedding texts to %s",
        len(texts), text_cache_path,
    )


def _load_cached_chunk_and_text_artifacts(
    cache_root: Path,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Load chunks.json and embedding_texts.jsonl from a cache directory."""
    from .qa.schemas import load_chunks

    chunks_path = cache_root / "chunks.json"
    text_cache_path = cache_root / "embedding_texts.jsonl"

    if not chunks_path.exists():
        print(f"Missing cached chunks file: {chunks_path}", file=sys.stderr)
        sys.exit(2)
    if not text_cache_path.exists():
        print(f"Missing cached embedding texts file: {text_cache_path}", file=sys.stderr)
        sys.exit(2)

    chunks = load_chunks(chunks_path)
    text_records: List[Dict[str, Any]] = []
    with open(text_cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                print(
                    f"Expected JSON objects in {text_cache_path}, got {type(record).__name__}",
                    file=sys.stderr,
                )
                sys.exit(2)
            text_records.append(record)

    if len(chunks) != len(text_records):
        print(
            f"Cached chunk/text count mismatch: {len(chunks)} chunks vs {len(text_records)} text records",
            file=sys.stderr,
        )
        sys.exit(2)

    texts: List[str] = []
    for chunk, record in zip(chunks, text_records):
        if record.get("chunk_id") != chunk.get("chunk_id"):
            print(
                "Cached chunk/text order mismatch: "
                f"{record.get('chunk_id')} != {chunk.get('chunk_id')}",
                file=sys.stderr,
            )
            sys.exit(2)
        texts.append(str(record.get("text") or ""))

    return chunks, texts


# ════════════════════════════════════════════
# ORCHESTRATOR
# ════════════════════════════════════════════


def run(config: cfg.ExperimentConfig, argv: Optional[Sequence[str]] = None) -> int:
    """Run a complete experiment end-to-end. Returns a process exit code."""
    from .cache import setup_logging
    from .embed import embed_texts
    from .extract import tesseract as tesseract_extract
    from .index import ensure_pinecone_index, upsert_to_pinecone
    from .render import render_pdf_pages

    args = parse_args(config, argv)
    load_dotenv()
    env = _resolve_environment(config, args)

    layout = CacheLayout(root=args.cache_dir)
    setup_logging(layout.root)
    logging.info(
        "[%s] pipeline starting: pdf=%s semester=%d pages=[%d..%s] index=%s",
        config.name, args.pdf, args.semester, args.start_page,
        str(args.end_page) if args.end_page is not None else "end",
        env.pinecone_index_name,
    )

    # ── Client initialisation ────────────────────────────────────────────
    # One Vertex AI client serves both extraction (vision) and embedding.
    vertex_client = genai.Client(
        vertexai=True, project=env.gcp_project, location=env.gcp_location,
    )
    pc = Pinecone(api_key=env.pinecone_api_key)

    if config.uses_diagrams and not args.skip_bbox and not args.chunk_only and not args.embed_only:
        cloudinary.config(
            cloud_name=env.cloudinary_cloud_name,
            api_key=env.cloudinary_api_key,
            api_secret=env.cloudinary_api_secret,
            secure=True,
        )

    page_dimensions: Dict[str, Dict[str, int]] = {}

    # ── Extraction + chunking (the variables under study) ────────────────
    if args.embed_only:
        logging.info("Stage 3A skipped (--embed-only): loading cached chunks/texts")
        chunks, texts = _load_cached_chunk_and_text_artifacts(layout.root)
    elif config.extraction == cfg.EXTRACTION_GEMINI:
        page_dimensions = _run_gemini_extraction(config, args, layout, vertex_client)

        if config.chunking == cfg.CHUNKING_PEDAGOGICAL:
            chunks, texts = _chunk_pedagogical(config, args, layout)
        elif config.chunking == cfg.CHUNKING_FIXED_STREAM:
            chunks, texts = _chunk_fixed_stream(config, args, layout)
            # For B2 fixed-stream chunking, persist chunk/text artifacts too.
            _write_chunk_and_text_artifacts(layout.root, chunks, texts)
        else:
            raise ValueError(f"unsupported chunking for gemini extraction: {config.chunking}")

    elif config.extraction == cfg.EXTRACTION_TESSERACT:
        logging.info("Stage 1: rendering PDF pages at %d DPI", config.render.dpi)
        render_pdf_pages(
            args.pdf, layout.pages, config.render.dpi,
            start_page=args.start_page, end_page=args.end_page,
        )
        logging.info("Stage OCR: Tesseract OCR (lang=%s)", config.tesseract.ocr_lang)
        page_texts = tesseract_extract.ocr_pages(
            layout.pages, layout.ocr, layout.root, config.tesseract,
            start_page=args.start_page, end_page=args.end_page,
        )
        logging.info("OCR produced text for %d pages", len(page_texts))

        if config.chunking == cfg.CHUNKING_FIXED_OCR:
            chunks, texts = _chunk_fixed_ocr(config, args, layout, page_texts)
        else:
            raise ValueError(f"unsupported chunking for tesseract extraction: {config.chunking}")

        # For B1, persist chunk/text artifacts for inspection.
        _write_chunk_and_text_artifacts(layout.root, chunks, texts)
    else:
        raise ValueError(f"unknown extraction method: {config.extraction}")

    if not chunks:
        logging.warning("No embeddable chunks produced. Exiting.")
        return 0

    if args.stop_after_texts:
        logging.info("--stop-after-texts set: skipping Stage 3B embedding and Stage 3C Pinecone upsert")
        logging.info("══════════════════════════════════════════════")
        logging.info("PIPELINE SUMMARY [%s]", config.name)
        logging.info("  PDF:                    %s", args.pdf)
        logging.info("  Semester:               %d", args.semester)
        logging.info("  Extraction:             %s", config.extraction)
        logging.info("  Chunking:               %s", config.chunking)
        if page_dimensions:
            logging.info("  Pages rendered:         %d", len(page_dimensions))
        logging.info("  Chunks produced:        %d", len(chunks))
        logging.info("  Embedding texts:        %d", len(texts))
        logging.info("  Stage 3B/3C:            skipped (--stop-after-texts)")
        logging.info("  Cache dir:              %s", layout.root)
        logging.info("══════════════════════════════════════════════")
        print("\nDone (%s, texts-only). See" % config.name, layout.root / "pipeline.log", "for full log.")
        return 0

    # ── Stage 3B — embed texts with Gemini ───────────────────────────────
    logging.info(
        "Stage 3B: embedding %d texts with %s (dim=%d)",
        len(texts), config.embedding.model, config.embedding.dim,
    )
    vectors = embed_texts(vertex_client, texts, config.embedding)

    # Determine the actual vector dimension from the first non-empty vector.
    actual_dim = next((len(v) for v in vectors if v), config.embedding.dim)

    # ── Stage 3C — upsert to Pinecone ────────────────────────────────────
    logging.info(
        "Stage 3C: ensuring Pinecone index '%s' (dim=%d)",
        env.pinecone_index_name, actual_dim,
    )
    index = ensure_pinecone_index(pc, env.pinecone_index_name, actual_dim, config.index)
    upserted = upsert_to_pinecone(index, chunks, vectors, config.index, texts=texts)

    # ── Summary ──────────────────────────────────────────────────────────
    logging.info("══════════════════════════════════════════════")
    logging.info("PIPELINE SUMMARY [%s]", config.name)
    logging.info("  PDF:                    %s", args.pdf)
    logging.info("  Semester:               %d", args.semester)
    logging.info("  Extraction:             %s", config.extraction)
    logging.info("  Chunking:               %s", config.chunking)
    logging.info("  Embedding model:        %s", config.embedding.model)
    if page_dimensions:
        logging.info("  Pages rendered:         %d", len(page_dimensions))
    logging.info("  Chunks produced:        %d", len(chunks))
    logging.info("  Embedded chunks:        %d", sum(1 for v in vectors if v))
    logging.info("  Pinecone index:         %s", env.pinecone_index_name)
    logging.info("  Pinecone upserted:      %d", upserted)
    logging.info("  Cache dir:              %s", layout.root)
    logging.info("══════════════════════════════════════════════")

    print("\nDone (%s). See" % config.name, layout.root / "pipeline.log", "for full log.")
    return 0
