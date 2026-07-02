"""
ragkit.config
═════════════

Every tunable "knob" in the pipeline lives here as a frozen dataclass.

The goal of centralising configuration is twofold:
    1. The three experiments (proposed, B1, B2) differ ONLY in a handful of
       knobs — extraction method, chunking strategy, target index. Expressing
       those differences as data (an ``ExperimentConfig``) instead of as forked
       copies of a monolithic script keeps the variables under study explicit.
    2. Pipeline functions accept these values as plain arguments, so nothing in
       the codebase reaches for a hidden module-level global.

None of the default values changed during the refactor — they are exactly the
constants that previously lived at the top of ``ingest.py`` and the baseline
scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ════════════════════════════════════════════
# STAGE-LEVEL CONFIG OBJECTS
# ════════════════════════════════════════════


@dataclass(frozen=True)
class RenderConfig:
    """PDF → PNG rendering knobs."""

    # 300 DPI gives sufficient detail for Gemini vision and Tesseract OCR.
    dpi: int = 300


@dataclass(frozen=True)
class GeminiExtractionConfig:
    """Structured extraction (proposed / B2) knobs."""

    # Gemini model used for page extraction (vision + structured JSON output).
    flash_model: str = "gemini-3.1-pro-preview"
    # Let the model decide its own thinking budget (-1).
    thinking_budget: int = -1
    # Sampling temperature for extraction.
    temperature: float = 1.0


@dataclass(frozen=True)
class TesseractExtractionConfig:
    """Tesseract OCR (B1) knobs."""

    # Tesseract language code for Arabic; requires the `ara` traineddata.
    ocr_lang: str = "ara"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Gemini embedding knobs."""

    # Gemini embedding model and the output vector dimension it produces.
    model: str = "gemini-embedding-001"
    dim: int = 3072
    # The embedding API accepts at most 64 inputs per request.
    batch_size: int = 64
    task_type: str = "RETRIEVAL_DOCUMENT"


@dataclass(frozen=True)
class RepresentationConfig:
    """build_embedding_text knobs (proposed / B2 representation)."""

    # Embedding model token budget, estimated at `chars_per_token` chars/token.
    max_tokens: int = 1800
    chars_per_token: int = 3


@dataclass(frozen=True)
class FixedChunkConfig:
    """Fixed-window (512-token + overlap) chunking knobs — proposed/B1/B2."""

    # Tokens are approximated by whitespace-separated words.
    chunk_size_tokens: int = 512
    overlap_tokens: int = 50


@dataclass(frozen=True)
class IndexConfig:
    """Pinecone index management + upsert knobs."""

    # Pinecone metadata values cannot exceed ~40 KB per vector; cap text fields.
    max_metadata_text_chars: int = 12000
    upsert_batch_size: int = 100
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


@dataclass(frozen=True)
class EnvVars:
    """Names of the environment variables the pipeline reads.

    Centralising the names (rather than the secret values) makes the
    documentation in ``.env.example`` and the README easy to keep in sync.
    """

    google_credentials: str = "GOOGLE_APPLICATION_CREDENTIALS"
    google_project: str = "GOOGLE_CLOUD_PROJECT"
    google_location: str = "GOOGLE_CLOUD_LOCATION"
    pinecone_api_key: str = "PINECONE_API_KEY"
    cloudinary_cloud_name: str = "CLOUDINARY_CLOUD_NAME"
    cloudinary_api_key: str = "CLOUDINARY_API_KEY"
    cloudinary_api_secret: str = "CLOUDINARY_API_SECRET"

    # Default GCP location when GOOGLE_CLOUD_LOCATION is unset.
    google_location_default: str = "us-central1"


# ════════════════════════════════════════════
# EXPERIMENT CONFIG
# ════════════════════════════════════════════

# The legal values for the two variables under study.
EXTRACTION_GEMINI = "gemini"
EXTRACTION_TESSERACT = "tesseract"

CHUNKING_PEDAGOGICAL = "pedagogical"   # proposed
CHUNKING_FIXED_OCR = "fixed_ocr"       # B1: fixed windows over raw OCR text
CHUNKING_FIXED_STREAM = "fixed_stream" # B2: fixed windows over proposed representation


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete description of one experiment.

    The three experiments differ ONLY in the first handful of fields; every
    stage-level config object below them is shared and unchanged so that the
    *only* things that vary between runs are the variables under study.
    """

    # ── Identity / variables under study ─────────────────────────────────
    name: str                       # "proposed" | "b1" | "b2"
    extraction: str                 # EXTRACTION_*
    chunking: str                   # CHUNKING_*

    # ── Pinecone target ──────────────────────────────────────────────────
    index_env_var: str              # e.g. "PINECONE_INDEX_NAME_B1"
    default_index_name: str         # used when the env var is unset

    # ── Filesystem ───────────────────────────────────────────────────────
    default_cache_dir: str          # e.g. "cache_b1"

    # ── Behaviour flags ──────────────────────────────────────────────────
    # Whether the run crops diagrams and uploads them to Cloudinary. Only the
    # Gemini-extraction experiments (proposed, B2) produce diagrams.
    uses_diagrams: bool = False
    # The content_type label written to records produced by fixed/semantic
    # chunkers (pedagogical chunks set their own per-block type).
    chunk_content_type: str = ""
    # The chunk_id prefix used by the baseline record builders (e.g. "b1").
    chunk_id_prefix: str = ""

    # ── Shared stage configs (defaults are identical across experiments) ──
    render: RenderConfig = field(default_factory=RenderConfig)
    gemini: GeminiExtractionConfig = field(default_factory=GeminiExtractionConfig)
    tesseract: TesseractExtractionConfig = field(default_factory=TesseractExtractionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    fixed_chunk: FixedChunkConfig = field(default_factory=FixedChunkConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    env: EnvVars = field(default_factory=EnvVars)
