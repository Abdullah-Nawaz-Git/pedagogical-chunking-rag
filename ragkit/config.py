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

# Two variables under study.
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


# ════════════════════════════════════════════
# QA-DATASET PREPARATION CONFIG
# ════════════════════════════════════════════
#
# The QA-dataset pipeline (``ragkit.qa``) prepares ONE Arabic QA dataset
# from the Proposed pedagogical chunks, then maps gold provenance onto the B1,
# B2, and Proposed retrieval corpora.

# The six question types produced by the pipeline, in canonical order.
QA_QUESTION_TYPES: tuple[str, ...] = (
    "definition_recall",
    "theorem_statement",
    "formula_retrieval",
    "diagram_dependent",
    "worked_example_reasoning",
    "cross_lesson_application",
)


@dataclass(frozen=True)
class QAQuotas:
    """Exact number of final QA items required per question type (sums to 180)."""

    definition_recall: int = 30
    theorem_statement: int = 30
    formula_retrieval: int = 30
    diagram_dependent: int = 30
    worked_example_reasoning: int = 30
    cross_lesson_application: int = 30

    def as_dict(self) -> "dict[str, int]":
        return {t: getattr(self, t) for t in QA_QUESTION_TYPES}


@dataclass(frozen=True)
class QASelectionConfig:
    """Stage-1 source-selection knobs."""

    # A single Proposed chunk may seed at most this many generation tasks.
    max_questions_per_source_chunk: int = 2
    # Soft cap: try to keep any one lesson below this fraction of a type's tasks.
    max_lesson_fraction: float = 0.30
    # Worked-example chunks shorter than this are treated as incomplete prompts.
    min_example_characters: int = 250
    # content_type values eligible for worked_example_reasoning.
    allowed_example_types: tuple[str, ...] = ("example", "worked_example", "explore")


@dataclass(frozen=True)
class QAB2MappingConfig:
    """B2 gold-mapping source-block coverage thresholds."""

    ordinary_min_source_block_coverage: float = 0.80
    worked_example_min_source_block_coverage: float = 1.00


@dataclass(frozen=True)
class QAProposedMappingConfig:
    """Proposed gold-mapping source-block coverage threshold."""

    minimum_source_block_coverage: float = 0.99


@dataclass(frozen=True)
class QAValidationConfig:
    """Stage-4 deterministic validation limits + banned question phrases."""

    min_question_characters: int = 8
    min_answer_characters: int = 8
    max_question_characters: int = 500
    max_answer_characters: int = 1200
    # Phrases that leak retrieval-artifact context into the question. Compared
    # after Arabic normalisation, so store them normalised-friendly.
    banned_question_phrases: tuple[str, ...] = (
        "الصفحة",
        "في الصفحة",
        "النص أعلاه",
        "في النص",
        "الشكل أعلاه",
        "الرسم أعلاه",
        "القطعة أعلاه",
    )


@dataclass(frozen=True)
class QAConfig:
    """Complete description of a QA-dataset preparation run.

    Defaults mirror the project specification. An optional ``--config`` file may
    override any subset of these values (see ``ragkit.qa.schemas.load_qa_config``).
    """

    # ── Dataset identity / sizing ────────────────────────────────────────
    dataset_version: str = "v1.0"
    target_total: int = 180
    candidates_per_task: int = 2
    random_seed: int = 42

    # ── Nested knob groups ───────────────────────────────────────────────
    quotas: QAQuotas = field(default_factory=QAQuotas)
    selection: QASelectionConfig = field(default_factory=QASelectionConfig)
    b2_mapping: QAB2MappingConfig = field(default_factory=QAB2MappingConfig)
    proposed_mapping: QAProposedMappingConfig = field(default_factory=QAProposedMappingConfig)
    validation: QAValidationConfig = field(default_factory=QAValidationConfig)

    # ── Filesystem inputs / output ───────────────────────────────────────
    # Proposed pedagogical chunks — the ONLY QA-generation source.
    proposed_chunks_path: str = "cache/chunks.json"
    # B1 (OCR) and B2 (fixed) chunks — used only for gold mapping.
    b1_chunks_path: str = "cache_b1/chunks.json"
    b2_chunks_path: str = "cache_b2/chunks.json"
    # Directory that receives every QA artifact.
    output_dir: str = "qa_dataset"

    # ── Generation provenance (never a hard-coded model/key) ─────────────
    # Provider is chosen at the CLI ("mock" | "vertex"); the model comes from
    # the environment variable named below so no model id is baked into code.
    provider: str = "mock"
    generation_model_env_var: str = "QA_GENERATION_MODEL"
    generation_model_default: str = "gemini-3.1-pro-preview"
    generation_temperature: float = 1.0
    generation_prompt_version: str = "v1"
    # Recorded on every QA item so the dataset is traceable to its corpus.
    source_corpus_version: str = "v1.0"
