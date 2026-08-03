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

# The five question types produced by the pipeline, in canonical order.
QA_QUESTION_TYPES: tuple[str, ...] = (
    "definition_recall",
    "theorem_statement",
    "formula_retrieval",
    "diagram_dependent",
    "worked_example_reasoning",
)


@dataclass(frozen=True)
class QAQuotas:
    """Exact number of final QA items required per question type (sums to 100)."""

    definition_recall: int = 20
    theorem_statement: int = 20
    formula_retrieval: int = 20
    diagram_dependent: int = 20
    worked_example_reasoning: int = 20

    def as_dict(self) -> "dict[str, int]":
        return {t: getattr(self, t) for t in QA_QUESTION_TYPES}


@dataclass(frozen=True)
class QASelectionConfig:
    """Stage-1 source-selection knobs."""

    # A single Proposed chunk may seed at most this many generation tasks.
    max_questions_per_source_chunk: int = 1
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
        "مصطلحات الوحدة",
    )


@dataclass(frozen=True)
class QAConfig:
    """Complete description of a QA-dataset preparation run.

    Defaults mirror the project specification. An optional ``--config`` file may
    override any subset of these values (see ``ragkit.qa.schemas.load_qa_config``).
    """

    # ── Dataset identity / sizing ────────────────────────────────────────
    dataset_version: str = "v1.0"
    target_total: int = 150
    candidates_per_task: int = 1
    random_seed: int = 7

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


# ════════════════════════════════════════════
# RETRIEVAL-EVALUATION CONFIG
# ════════════════════════════════════════════
#
# The retrieval experiments (``ragkit.retrieval``) read the FROZEN QA dataset and
# the gold mapping artifacts produced by ``ragkit.qa``, query each system's
# already-built Pinecone index, and score the results. Nothing here regenerates
# the QA dataset or touches ingestion.
#
# Everything that must be held constant for the comparison to be fair lives in
# ``RetrievalConfig`` / ``AnswerGenerationConfig`` and is shared verbatim by all
# three systems. The ONLY per-system data is in ``RetrievalSystemConfig``:
# identity, which index to query, which chunk cache to read, and which gold
# granularity that system's provenance can legitimately support.

# How a system's gold relevance is defined. This is a *semantic* distinction, not
# a tuning knob: it records what the system's provenance can actually support.
GOLD_UNIT_LEVEL = "source_block_unit"   # proposed, B2 — true instructional-unit gold
GOLD_PAGE_PROXY = "page_overlap_proxy"  # B1 only — page-overlap proxy, NOT unit gold

# The B1 caveat, carried onto every B1 record and printed in every report so a
# reader can never mistake B1's proxy for true Gold Unit Recall.
B1_PROXY_DISCLAIMER = (
    "B1 has no source-block provenance. Its gold relevance is a PAGE-OVERLAP "
    "PROXY, so the reported recall is Gold Page Recall (proxy), not Gold Unit "
    "Recall. B1 numbers are not unit-level and must not be read as measuring "
    "the same quantity as the unit-level recall of Proposed/B2."
)


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval protocol knobs — IDENTICAL for every system under comparison."""

    # Top-k chunks fetched from the vector index per question.
    top_k: int = 5
    # Ranks at which Hit@k is reported (each must be <= top_k).
    hit_at_ks: tuple[int, ...] = (1, 5)
    # Query-side embedding task type. The MODEL and DIMENSION come from the
    # shared ``EmbeddingConfig``, so queries use the same embedding model family
    # as the documents; only the task type differs (query vs document).
    query_task_type: str = "RETRIEVAL_QUERY"
    # Metadata filtering is deliberately DISABLED: a filter would hand each
    # system a different effective candidate set and invalidate the comparison.
    use_metadata_filter: bool = False
    # Pinecone returns metadata (which carries content_text_ar) so generator
    # context is available even if a chunk is missing from the local cache.
    include_metadata: bool = True
    include_values: bool = False


@dataclass(frozen=True)
class AnswerGenerationConfig:
    """Answer-generation knobs — IDENTICAL for every system under comparison.

    Answer generation is OPTIONAL (``--generate-answers``); retrieval metrics are
    computed either way. When it runs, all systems share one model, one prompt
    version, and one context budget so the generator is never a confound.
    """

    # Provider is chosen at the CLI ("mock" | "vertex"); the model id is read
    # from the environment variable below so no model id is baked into code.
    provider: str = "mock"
    model_env_var: str = "RETRIEVAL_ANSWER_MODEL"
    model_default: str = "gemini-3.1-pro-preview"
    # Deterministic decoding: the comparison must not drift between reruns.
    temperature: float = 0.0
    prompt_version: str = "v1"
    # Shared context budget in whitespace tokens (the repo's token convention,
    # matching FixedChunkConfig). Retrieved chunks are appended in rank order and
    # the first chunk that would exceed the budget is truncated, so no system
    # gains an advantage purely from having larger chunks.
    context_budget_tokens: int = 2560
    # Used only to report an *estimated* model-token count alongside the exact
    # whitespace-token count (the same estimate RepresentationConfig uses).
    chars_per_token: int = 3


@dataclass(frozen=True)
class RetrievalSystemConfig:
    """The per-system half of a retrieval experiment — configuration only.

    Adding a system means adding one of these; it must never mean adding a
    branch to the shared engine, metrics, or reporting code.
    """

    # Identity — matches the ingestion experiment name and the gold-mapping
    # ``system`` field ("proposed" | "b1" | "b2").
    name: str
    # Which Pinecone index to query (already built by the ingestion run).
    index_env_var: str
    default_index_name: str
    # Chunk cache used to resolve retrieved chunk_ids back to text/provenance.
    chunks_path: str
    # Gold-mapping artifact for this system, under the QA dataset directory.
    gold_mapping_filename: str
    # What this system's provenance can legitimately support (GOLD_* constant).
    gold_granularity: str = GOLD_UNIT_LEVEL
    # A non-empty caveat is copied onto every record and every report for this
    # system. Only B1 sets one.
    proxy_disclaimer: str = ""
    # Human-readable label used in report tables.
    label: str = ""
    # The experimental role this system plays in the paper's comparison. Recorded
    # verbatim in the judge system registry / reports so a reader always knows why
    # a system is present (treatment, control, or baseline).
    experimental_role: str = ""

    @property
    def is_unit_level(self) -> bool:
        """True when the system supports true instructional-unit gold."""
        return self.gold_granularity == GOLD_UNIT_LEVEL

    @property
    def recall_metric_name(self) -> str:
        """The name under which this system's recall may honestly be reported."""
        return "gold_unit_recall" if self.is_unit_level else "gold_page_recall_proxy"


@dataclass(frozen=True)
class RetrievalExperimentConfig:
    """A complete description of one retrieval-evaluation run.

    Mirrors ``ExperimentConfig``: the per-system variable sits at the top and
    every shared stage config below it is identical across systems, so the only
    thing that varies between runs is the system being measured.
    """

    # ── The system under evaluation ───────────────────────────────────────
    system: RetrievalSystemConfig

    # ── Frozen inputs (never regenerated by this pipeline) ────────────────
    qa_dataset_dir: str = "qa_dataset"
    qa_dataset_filename: str = "qa_dataset_v1.jsonl"
    # Directory that receives every retrieval artifact.
    output_dir: str = "retrieval_eval"

    # ── Shared stage configs (identical across systems) ───────────────────
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    answer: AnswerGenerationConfig = field(default_factory=AnswerGenerationConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    env: EnvVars = field(default_factory=EnvVars)

    # ── Behaviour ─────────────────────────────────────────────────────────
    # "pinecone" queries the real index; "local" scores the cached chunks with a
    # deterministic offline retriever (no credentials) for smoke-testing.
    retriever: str = "pinecone"
    # Whether to run the shared answer generator.
    generate_answers: bool = False
    # Ties break on chunk_id rather than on RNG, but recording the seed keeps
    # run provenance explicit in config_used.json.
    random_seed: int = 7


# The three systems under study. Index names / env vars mirror the ingestion
# experiments so retrieval reads exactly the indexes ingestion wrote.
RETRIEVAL_SYSTEM_PROPOSED = RetrievalSystemConfig(
    name="proposed",
    index_env_var="PINECONE_INDEX_NAME",
    default_index_name="curriculum-highschool",
    chunks_path="cache/chunks.json",
    gold_mapping_filename="gold_mapping_proposed.jsonl",
    gold_granularity=GOLD_UNIT_LEVEL,
    label="Proposed (VLM + pedagogical chunks)",
    experimental_role="treatment (pedagogical chunk boundaries)",
)

RETRIEVAL_SYSTEM_B2 = RetrievalSystemConfig(
    name="b2",
    index_env_var="PINECONE_INDEX_NAME_B2",
    default_index_name="curriculum-highschool-b2",
    chunks_path="cache_b2/chunks.json",
    gold_mapping_filename="gold_mapping_b2.jsonl",
    gold_granularity=GOLD_UNIT_LEVEL,
    label="B2 (VLM + fixed 512-token windows)",
    experimental_role="structured fixed-window control (isolates chunk boundaries)",
)

RETRIEVAL_SYSTEM_B1 = RetrievalSystemConfig(
    name="b1",
    index_env_var="PINECONE_INDEX_NAME_B1",
    default_index_name="curriculum-highschool-b1",
    chunks_path="cache_b1/chunks.json",
    gold_mapping_filename="gold_mapping_b1.jsonl",
    # B1 keeps no source-block provenance, so unit-level gold is impossible.
    gold_granularity=GOLD_PAGE_PROXY,
    proxy_disclaimer=B1_PROXY_DISCLAIMER,
    label="B1 (Tesseract OCR + fixed 512-token windows)",
    experimental_role="OCR fixed-window baseline (page-overlap proxy gold)",
)

# Canonical system order for cross-system reports.
RETRIEVAL_SYSTEM_ORDER: tuple[str, ...] = ("proposed", "b2", "b1")

RETRIEVAL_SYSTEMS = {
    system.name: system
    for system in (RETRIEVAL_SYSTEM_PROPOSED, RETRIEVAL_SYSTEM_B2, RETRIEVAL_SYSTEM_B1)
}


# ════════════════════════════════════════════
# LLM-AS-JUDGE CONFIG (inspired by RAGAS dimensions)
# ════════════════════════════════════════════
#
# The judge pipeline (``ragkit.judge``) is a SECONDARY, end-to-end evaluation that
# runs ALONGSIDE — never replacing — the deterministic retrieval metrics
# (Hit@k, MRR, Gold-Unit / Gold-Page recall) implemented in ``ragkit.retrieval``.
#
# It reads the FROZEN retrieval-result artifacts (``retrieval_eval/*``), the frozen
# QA dataset, and the cached chunks, then asks a configurable LLM judge to score
# four dimensions inspired by (but NOT identical to) the RAGAS framework. The
# scores are deliberately called "LLM-as-judge scores inspired by RAGAS
# dimensions", never official "RAGAS scores": retrieval-native metrics stay the
# primary evidence and the judge is corroborating, model-dependent evidence.
#
# Nothing here regenerates QA data, gold mappings, chunk ids, or provenance, and
# NO gold-derived relevance signal (gold ids, Hit@k, mapping status) is ever shown
# to the judge — those remain for deterministic evaluation only.

# The four judge dimensions, in canonical report order. Each has its OWN prompt
# and its OWN structured response; they are scored in isolation so one dimension
# can never leak another dimension's inputs into the model.
JUDGE_CONTEXT_RECALL = "context_recall"
JUDGE_CONTEXT_PRECISION = "context_precision"
JUDGE_FAITHFULNESS = "faithfulness"
JUDGE_ANSWER_RELEVANCY = "answer_relevancy"

JUDGE_DIMENSIONS: tuple[str, ...] = (
    JUDGE_CONTEXT_RECALL,
    JUDGE_CONTEXT_PRECISION,
    JUDGE_FAITHFULNESS,
    JUDGE_ANSWER_RELEVANCY,
)


@dataclass(frozen=True)
class JudgeDimensionSpec:
    """What one judge dimension measures and which inputs its prompt receives.

    The ``needs_*`` flags are the SINGLE source of truth for input isolation: the
    scoring runner assembles a dimension's judge input strictly from the fields
    flagged here, so e.g. Answer Relevancy never sees the retrieved contexts and
    NO dimension ever sees gold ids, ranks, or mapping status.
    """

    name: str
    display: str
    question: str
    # Inputs the dimension's prompt is allowed to receive.
    needs_question: bool = True
    needs_reference_answer: bool = False
    needs_contexts: bool = False
    needs_generated_answer: bool = False

    @property
    def requires_generation(self) -> bool:
        """Dimensions that need a generated answer only run in generation mode."""
        return self.needs_generated_answer


# The isolated input contract per dimension (matches the project plan exactly).
JUDGE_DIMENSION_SPECS: "dict[str, JudgeDimensionSpec]" = {
    JUDGE_CONTEXT_RECALL: JudgeDimensionSpec(
        name=JUDGE_CONTEXT_RECALL,
        display="Context Recall",
        question=(
            "Did the retrieved contexts contain the information needed to answer "
            "the question according to the reference answer?"
        ),
        needs_reference_answer=True,
        needs_contexts=True,
    ),
    JUDGE_CONTEXT_PRECISION: JudgeDimensionSpec(
        name=JUDGE_CONTEXT_PRECISION,
        display="Context Precision",
        question=(
            "Are useful retrieved contexts ranked above irrelevant or less useful "
            "contexts?"
        ),
        needs_reference_answer=True,
        needs_contexts=True,
    ),
    JUDGE_FAITHFULNESS: JudgeDimensionSpec(
        name=JUDGE_FAITHFULNESS,
        display="Faithfulness",
        question="Is the generated answer supported by the retrieved contexts?",
        needs_contexts=True,
        needs_generated_answer=True,
    ),
    JUDGE_ANSWER_RELEVANCY: JudgeDimensionSpec(
        name=JUDGE_ANSWER_RELEVANCY,
        display="Answer Relevancy",
        question="Does the generated answer directly answer the question?",
        needs_generated_answer=True,
    ),
}

# Run modes. ``retrieval_only`` scores the two context dimensions (no generated
# answer needed); ``generation`` additionally scores faithfulness + answer
# relevancy; ``auto`` picks per item based on whether a generated answer exists.
JUDGE_MODE_RETRIEVAL_ONLY = "retrieval_only"
JUDGE_MODE_GENERATION = "generation"
JUDGE_MODE_AUTO = "auto"

# Judge providers. ``mock`` is deterministic + offline (smoke tests, CI). The two
# real providers run through Vertex AI; Claude via Vertex is the default judge.
JUDGE_PROVIDER_MOCK = "mock"
JUDGE_PROVIDER_VERTEX_ANTHROPIC = "vertex_anthropic"  # Claude Sonnet via Vertex
JUDGE_PROVIDER_VERTEX_GEMINI = "vertex_gemini"        # configurable alternative


@dataclass(frozen=True)
class JudgeEnvVars:
    """Extra environment-variable names the judge reads (beyond ``EnvVars``).

    Only NAMES live here, never secrets, matching ``EnvVars``. The Vertex project
    and Gemini location come from the shared ``EnvVars``; Claude-on-Vertex often
    needs its own region, so it has a dedicated (optional) override.
    """

    judge_model: str = "JUDGE_MODEL"
    # Claude on Vertex is served from a subset of regions. This optional override
    # falls back to GOOGLE_CLOUD_LOCATION and then to ``anthropic_location_default``.
    anthropic_location: str = "JUDGE_VERTEX_LOCATION"
    anthropic_location_default: str = "us-east5"


@dataclass(frozen=True)
class JudgeModelConfig:
    """Judge-model knobs — held IDENTICAL across systems so the judge is never a
    confound in the Proposed-vs-control comparison.

    No model id is baked into code: the id is read from ``JUDGE_MODEL`` and only
    falls back to ``model_default`` when that env var is unset. The default names
    Gemini on Vertex; set ``JUDGE_MODEL`` to pin a specific snapshot.
    """

    # Provider is chosen at the CLI / config; default is Gemini via Vertex.
    provider: str = JUDGE_PROVIDER_VERTEX_GEMINI
    model_env_var: str = "JUDGE_MODEL"
    # Overridable default. Pin ``JUDGE_MODEL`` to a Vertex snapshot when a
    # reproducible vendor release identifier is required for a study.
    model_default: str = "gemini-3.1-pro-preview"
    # Low-variance decoding per the spec (temperature 0 / lowest supported).
    temperature: float = 0.0
    # Cap the judge's JSON response; a score + short rationale is small, but the
    # cap is generous so a long rationale never gets truncated mid-string (which
    # previously forced whole-score loss and required parser-side recovery).
    max_output_tokens: int = 2048
    # Structured JSON-only output is requested from every provider.
    response_json_only: bool = True


@dataclass(frozen=True)
class JudgeContextConfig:
    """Context-fairness knobs — IDENTICAL across systems.

    Fairness is the whole point: Proposed must not look stronger merely because it
    is handed a longer context. The judge reuses the SAME frozen context the answer
    generator saw whenever the retrieval record preserved it (its
    ``context_chunk_ids``); otherwise it re-assembles the context deterministically
    under this shared budget. Either way the token count and any truncation are
    logged on every record.
    """

    # Must match the retrieval run's AnswerGenerationConfig.context_budget_tokens
    # so the judge scores the same-sized window every system's generator saw.
    context_budget_tokens: int = 2560
    # Estimate-only model-token accounting (same convention as the rest of ragkit).
    chars_per_token: int = 3
    # Prefer reconstructing the exact frozen context from the record's
    # context_chunk_ids; fall back to re-assembling from retrieved chunks.
    prefer_recorded_context: bool = True


@dataclass(frozen=True)
class JudgeAggregationConfig:
    """Paired-comparison + bootstrap knobs for aggregate reporting."""

    # The key paper comparison: treatment vs the structured fixed-window control.
    primary_pair: tuple[str, str] = ("proposed", "b2")
    # Bootstrap CIs on paired per-item score differences (matches the retrieval
    # analysis style). Set n_resamples=0 to skip CIs entirely.
    bootstrap_resamples: int = 2000
    bootstrap_confidence: float = 0.95
    bootstrap_seed: int = 7


@dataclass(frozen=True)
class JudgeSystemConfig:
    """The per-system half of a judge run — configuration only.

    Composes the existing ``RetrievalSystemConfig`` (so cache paths, gold-mapping
    filenames, labels, granularity, and the B1 proxy disclaimer are reused verbatim
    and can never drift from the retrieval pipeline) and adds the judge-specific
    ``experimental_role`` / ``mapping_type`` that the reports surface. Adding a
    system means adding one of these — never branching the judge engine on a name.
    """

    retrieval: RetrievalSystemConfig

    @property
    def name(self) -> str:
        return self.retrieval.name

    @property
    def label(self) -> str:
        return self.retrieval.label or self.retrieval.name

    @property
    def experimental_role(self) -> str:
        return self.retrieval.experimental_role

    @property
    def chunks_path(self) -> str:
        return self.retrieval.chunks_path

    @property
    def gold_mapping_filename(self) -> str:
        return self.retrieval.gold_mapping_filename

    @property
    def gold_granularity(self) -> str:
        return self.retrieval.gold_granularity

    @property
    def is_unit_level(self) -> bool:
        return self.retrieval.is_unit_level

    @property
    def mapping_type(self) -> str:
        """Human/report label for the gold mapping this system's provenance yields.

        This is reporting metadata ONLY — it is never shown to the judge, which
        sees no gold-derived signal at all.
        """
        return (
            "true_source_block_unit"
            if self.retrieval.is_unit_level
            else "page_overlap_proxy"
        )

    @property
    def mapping_caveat(self) -> str:
        """Non-empty only for B1's page-overlap proxy (reused from retrieval)."""
        return self.retrieval.proxy_disclaimer


@dataclass(frozen=True)
class JudgeExperimentConfig:
    """A complete description of one judge-scoring run.

    Mirrors ``RetrievalExperimentConfig``: the per-system variable sits at the top
    and every shared config below it is identical across systems, so the only thing
    that varies between runs is the system being judged.
    """

    # ── The system under evaluation ───────────────────────────────────────
    system: JudgeSystemConfig

    # ── Frozen inputs (never regenerated by this pipeline) ────────────────
    qa_dataset_dir: str = "qa_dataset"
    qa_dataset_filename: str = "qa_dataset_v1.jsonl"
    # Where the retrieval pipeline wrote its per-system records the judge reads.
    retrieval_eval_dir: str = "retrieval_eval"
    # Directory that receives every judge artifact.
    output_dir: str = "judge_eval"

    # ── Shared stage configs (identical across systems) ───────────────────
    model: JudgeModelConfig = field(default_factory=JudgeModelConfig)
    context: JudgeContextConfig = field(default_factory=JudgeContextConfig)
    aggregation: JudgeAggregationConfig = field(default_factory=JudgeAggregationConfig)
    answer: AnswerGenerationConfig = field(default_factory=AnswerGenerationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    env: EnvVars = field(default_factory=EnvVars)
    judge_env: JudgeEnvVars = field(default_factory=JudgeEnvVars)

    # ── Behaviour ─────────────────────────────────────────────────────────
    mode: str = JUDGE_MODE_AUTO
    # Which dimensions to score (subset of JUDGE_DIMENSIONS); empty means "all
    # dimensions eligible under the current mode".
    dimensions: tuple[str, ...] = JUDGE_DIMENSIONS
    # Long runs are resumable: already-scored (qa_id, system, metric) triples are
    # skipped on restart when this is set.
    resume: bool = True
    # Bounded retries on transient API / parse failures before recording an error.
    max_retries: int = 2
    random_seed: int = 7
    prompt_version: str = "v1"


# The three systems under study, wrapping the retrieval registry so the judge and
# retrieval pipelines can never disagree about a system's paths or granularity.
JUDGE_SYSTEM_PROPOSED = JudgeSystemConfig(retrieval=RETRIEVAL_SYSTEM_PROPOSED)
JUDGE_SYSTEM_B2 = JudgeSystemConfig(retrieval=RETRIEVAL_SYSTEM_B2)
JUDGE_SYSTEM_B1 = JudgeSystemConfig(retrieval=RETRIEVAL_SYSTEM_B1)

# Canonical judge system order (treatment, structured control, OCR baseline).
JUDGE_SYSTEM_ORDER: tuple[str, ...] = ("proposed", "b2", "b1")

# The key paper comparison isolates chunk boundaries: Proposed pedagogical
# chunking vs the structured fixed-window control (B2), holding structured
# extraction + representation constant.
JUDGE_PRIMARY_COMPARISON: tuple[str, str] = ("proposed", "b2")

JUDGE_SYSTEMS = {
    system.name: system
    for system in (JUDGE_SYSTEM_PROPOSED, JUDGE_SYSTEM_B2, JUDGE_SYSTEM_B1)
}
