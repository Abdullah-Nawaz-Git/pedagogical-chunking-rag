"""ragkit — a small toolkit for building RAG ingestion pipelines.

The package factors the experimental ingestion scripts into reusable pieces:

- ``config``      Dataclass-based configuration (all tunable knobs).
- ``cache``       Logging + on-disk cache directory helpers.
- ``render``      PDF -> page image rendering and page-range parsing.
- ``extract``     Page text extraction backends (VLM, Tesseract OCR).
- ``represent``   Building the text representation used for embedding.
- ``chunk``       Chunking strategies (fixed, semantic, pedagogical).
- ``embed``       Embedding backends.
- ``index``       Vector index construction and persistence.
- ``pipeline``    The orchestrator that wires the stages together.

Experiment entry points (``experiments/``) compose these pieces into the
specific configurations used by the proposed pipeline and baselines B1-B2.
"""

from importlib import import_module

from ragkit.config import (
    CHUNKING_FIXED_OCR,
    CHUNKING_FIXED_STREAM,
    CHUNKING_PEDAGOGICAL,
    EXTRACTION_GEMINI,
    EXTRACTION_TESSERACT,
    EmbeddingConfig,
    EnvVars,
    ExperimentConfig,
    FixedChunkConfig,
    GeminiExtractionConfig,
    IndexConfig,
    RenderConfig,
    RepresentationConfig,
    TesseractExtractionConfig,
)

__all__ = [
    "CacheLayout",
    "EmbeddingConfig",
    "EnvVars",
    "ExperimentConfig",
    "FixedChunkConfig",
    "GeminiExtractionConfig",
    "IndexConfig",
    "RenderConfig",
    "RepresentationConfig",
    "TesseractExtractionConfig",
    "CHUNKING_FIXED_OCR",
    "CHUNKING_FIXED_STREAM",
    "CHUNKING_PEDAGOGICAL",
    "EXTRACTION_GEMINI",
    "EXTRACTION_TESSERACT",
]

_LAZY_EXPORTS = {
    "CacheLayout": ("ragkit.cache", "CacheLayout"),
    "append_log": ("ragkit.cache", "append_log"),
    "setup_logging": ("ragkit.cache", "setup_logging"),
    "build_ocr_stream": ("ragkit.chunk.fixed", "build_ocr_stream"),
    "fixed_token_chunks": ("ragkit.chunk.fixed", "fixed_token_chunks"),
    "make_ocr_records": ("ragkit.chunk.fixed", "make_ocr_records"),
    "make_stream_records": ("ragkit.chunk.fixed", "make_stream_records"),
    "build_chunks": ("ragkit.chunk.pedagogical", "build_chunks"),
    "new_chunk_from_block": ("ragkit.chunk.pedagogical", "new_chunk_from_block"),
    "embed_texts": ("ragkit.embed", "embed_texts"),
    "ensure_pinecone_index": ("ragkit.index", "ensure_pinecone_index"),
    "sanitize_metadata": ("ragkit.index", "sanitize_metadata"),
    "upsert_to_pinecone": ("ragkit.index", "upsert_to_pinecone"),
    "run": ("ragkit.pipeline", "run"),
    "build_embedding_text": ("ragkit.represent", "build_embedding_text"),
    "build_representation_stream": ("ragkit.represent", "build_representation_stream"),
    "write_embedding_text_cache": ("ragkit.represent", "write_embedding_text_cache"),
    "key_in_range": ("ragkit.render", "key_in_range"),
    "page_in_range": ("ragkit.render", "page_in_range"),
    "render_pdf_pages": ("ragkit.render", "render_pdf_pages"),
}


def __getattr__(name: str):
    """Import selected helpers lazily so importing ragkit stays lightweight."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.1.0"
