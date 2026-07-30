"""
ragkit.retrieval.engine
═══════════════════════

The ONE retrieval engine shared by Proposed, B1, and B2.

The retrieval protocol is fixed here so it cannot drift between systems:

    * queries are embedded with the SAME embedding model family as the documents
      (``EmbeddingConfig.model`` / ``dim``), using the query task type;
    * exactly ``RetrievalConfig.top_k`` chunks are requested;
    * NO metadata filter is ever passed;
    * the index configuration is untouched — this module only reads the indexes
      that ingestion already built, and never upserts;
    * ranking is made deterministic by breaking score ties on ``chunk_id``.

Two retrievers implement the same protocol:

    * ``PineconeRetriever`` — the real evaluation path.
    * ``LocalRetriever``    — a deterministic, offline retriever that scores the
      cached ``chunks.json`` with token-overlap similarity. It exists so the whole
      evaluation (retrieve → score → report) can be exercised without Pinecone or
      GCP credentials, mirroring the ``mock`` provider in ``ragkit.qa``.

Both return the same ``RetrievedChunk`` list, so metrics and reporting never know
which retriever ran.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .. import config as cfg

logger = logging.getLogger("ragkit.retrieval.engine")

RETRIEVER_PINECONE = "pinecone"
RETRIEVER_LOCAL = "local"


# ════════════════════════════════════════════
# RESULT TYPE
# ════════════════════════════════════════════


@dataclass(frozen=True)
class RetrievedChunk:
    """One ranked hit. ``rank`` is 1-based."""

    rank: int
    chunk_id: str
    score: float
    # Text used for generator context. Preferred source is the local chunk cache
    # (identical to what was embedded); Pinecone metadata is the fallback.
    text: str = ""
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:  # pragma: no cover - simple normalisation
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


def _rank_deterministically(
    scored: Sequence[tuple[str, float]],
    top_k: int,
) -> List[tuple[str, float]]:
    """Sort by descending score, breaking ties on ascending chunk_id.

    Deterministic tie-breaking matters because fixed-window corpora (B1/B2) can
    contain near-duplicate overlapping windows with identical scores; without a
    stable rule, Hit@1 could flip between reruns.
    """
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:top_k]


# ════════════════════════════════════════════
# QUERY EMBEDDING (shared by every system)
# ════════════════════════════════════════════


class QueryEmbedder(Protocol):
    """Embeds QA questions into the same vector space as the documents."""

    model: str

    def embed(self, texts: List[str]) -> List[List[float]]:
        ...


class GeminiQueryEmbedder:
    """Embeds queries with the project's Gemini embedding model.

    Reuses ``ragkit.embed.embed_texts`` (the exact function ingestion used for
    documents) with the task type switched to ``RETRIEVAL_QUERY`` — the model and
    output dimensionality are held identical so queries and documents share one
    vector space.
    """

    def __init__(self, config: cfg.RetrievalExperimentConfig) -> None:
        from dataclasses import replace

        from google import genai

        env = config.env
        project = os.environ.get(env.google_project)
        location = os.environ.get(env.google_location, env.google_location_default)
        if not project:
            raise RuntimeError(
                f"{env.google_project} must be set to embed queries with Gemini"
            )
        self._client = genai.Client(vertexai=True, project=project, location=location)
        # Same model + dim as the documents; only task_type differs.
        self._embedding_config = replace(
            config.embedding, task_type=config.retrieval.query_task_type
        )
        self.model = self._embedding_config.model

    def embed(self, texts: List[str]) -> List[List[float]]:
        from ..embed import embed_texts

        return embed_texts(self._client, texts, self._embedding_config)


class LocalQueryEmbedder:
    """Offline stand-in that returns no vectors (``LocalRetriever`` is lexical)."""

    model = "local-token-overlap"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[] for _ in texts]


# ════════════════════════════════════════════
# RETRIEVERS
# ════════════════════════════════════════════


class Retriever(Protocol):
    """Retrieves ``top_k`` chunks for one embedded query."""

    name: str

    def retrieve(self, query_text: str, query_vector: List[float]) -> List[RetrievedChunk]:
        ...


class PineconeRetriever:
    """Queries the already-built Pinecone index for one system.

    Read-only: it never creates, configures, or upserts to an index, so the
    vector-store configuration stays exactly as ingestion left it and is
    identical across the three systems.
    """

    name = RETRIEVER_PINECONE

    def __init__(
        self,
        config: cfg.RetrievalExperimentConfig,
        chunks_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        from pinecone import Pinecone

        env = config.env
        api_key = os.environ.get(env.pinecone_api_key)
        if not api_key:
            raise RuntimeError(f"{env.pinecone_api_key} must be set to query Pinecone")
        self.index_name = os.environ.get(
            config.system.index_env_var, config.system.default_index_name
        )
        client = Pinecone(api_key=api_key)
        # Attach to the existing index; deliberately NOT ensure_pinecone_index(),
        # which would create one if it were missing.
        self._index = client.Index(self.index_name)
        self._config = config
        self._chunks_by_id = chunks_by_id
        logger.info(
            "Pinecone retriever attached to index %r (top_k=%d, metadata_filter=%s)",
            self.index_name,
            config.retrieval.top_k,
            config.retrieval.use_metadata_filter,
        )

    def retrieve(self, query_text: str, query_vector: List[float]) -> List[RetrievedChunk]:
        retrieval = self._config.retrieval
        if not query_vector:
            raise ValueError("PineconeRetriever requires a non-empty query vector")

        # No `filter=` argument is ever supplied: metadata filtering is disabled
        # for every system so the candidate sets stay comparable.
        response = self._index.query(
            vector=query_vector,
            top_k=retrieval.top_k,
            include_metadata=retrieval.include_metadata,
            include_values=retrieval.include_values,
        )
        matches = _response_matches(response)

        scored = [
            (str(match.get("id")), float(match.get("score") or 0.0))
            for match in matches
            if match.get("id")
        ]
        metadata_by_id = {
            str(match.get("id")): dict(match.get("metadata") or {}) for match in matches
        }

        results: List[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(
            _rank_deterministically(scored, retrieval.top_k), start=1
        ):
            metadata = metadata_by_id.get(chunk_id, {})
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=score,
                    text=_context_text(chunk_id, self._chunks_by_id, metadata),
                    metadata=metadata,
                )
            )
        return results


def _response_matches(response: Any) -> List[Dict[str, Any]]:
    """Normalise a Pinecone query response into a list of plain dicts."""
    if isinstance(response, dict):
        matches = response.get("matches") or []
    else:
        matches = getattr(response, "matches", None) or []
    normalised: List[Dict[str, Any]] = []
    for match in matches:
        if isinstance(match, dict):
            normalised.append(match)
        else:  # pinecone client objects expose attributes
            normalised.append(
                {
                    "id": getattr(match, "id", None),
                    "score": getattr(match, "score", 0.0),
                    "metadata": getattr(match, "metadata", None),
                }
            )
    return normalised


class LocalRetriever:
    """Deterministic offline retriever over the cached chunks.

    Scores each chunk by cosine similarity of token-count vectors between the
    query and the chunk's embedding text. This is NOT a substitute for the real
    dense retrieval — it exists so the evaluation harness, metrics, and reports
    can be run and verified without network access or credentials.
    """

    name = RETRIEVER_LOCAL

    def __init__(
        self,
        config: cfg.RetrievalExperimentConfig,
        chunks_by_id: Dict[str, Dict[str, Any]],
    ) -> None:
        self._config = config
        self._chunks_by_id = chunks_by_id
        # Pre-tokenise every chunk once; keep insertion order for stable ties.
        self._chunk_tokens: Dict[str, Dict[str, int]] = {}
        self._chunk_norms: Dict[str, float] = {}
        for chunk_id, chunk in chunks_by_id.items():
            counts = _token_counts(_chunk_text(chunk))
            self._chunk_tokens[chunk_id] = counts
            self._chunk_norms[chunk_id] = math.sqrt(
                sum(value * value for value in counts.values())
            )
        logger.info(
            "Local retriever built over %d cached chunks (offline, deterministic)",
            len(self._chunk_tokens),
        )

    def retrieve(self, query_text: str, query_vector: List[float]) -> List[RetrievedChunk]:
        query_counts = _token_counts(query_text)
        query_norm = math.sqrt(sum(v * v for v in query_counts.values()))
        scored: List[tuple[str, float]] = []
        if query_norm > 0:
            for chunk_id, counts in self._chunk_tokens.items():
                norm = self._chunk_norms[chunk_id]
                if norm <= 0:
                    continue
                # Iterate the (short) query side for speed.
                dot = sum(
                    weight * counts.get(token, 0)
                    for token, weight in query_counts.items()
                )
                if dot:
                    scored.append((chunk_id, dot / (query_norm * norm)))

        results: List[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(
            _rank_deterministically(scored, self._config.retrieval.top_k), start=1
        ):
            chunk = self._chunks_by_id.get(chunk_id) or {}
            results.append(
                RetrievedChunk(
                    rank=rank,
                    chunk_id=chunk_id,
                    score=round(score, 6),
                    text=_chunk_text(chunk),
                    metadata={
                        "content_type": chunk.get("content_type") or "",
                        "page_range_start": _page_range(chunk)[0],
                        "page_range_end": _page_range(chunk)[1],
                    },
                )
            )
        return results


# ════════════════════════════════════════════
# TEXT HELPERS
# ════════════════════════════════════════════

# Arabic + Latin word characters and digits; punctuation is dropped.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _token_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in _TOKEN_RE.findall((text or "").lower()):
        counts[token] = counts.get(token, 0) + 1
    return counts


def _chunk_text(chunk: Dict[str, Any]) -> str:
    """The chunk's own text, exactly as stored in the cache."""
    return str(chunk.get("main_text_ar") or "")


def _full_chunk_text(chunk: Dict[str, Any]) -> str:
    """Full representation text including math expressions and diagram descriptions.

    Matches the content that ``build_embedding_text()`` produces (``represent.py``)
    but without the embedding-model token-limit truncation — the downstream context
    budget handles that. For Proposed pedagogical chunks this includes unit/lesson
    titles, math expressions, and diagram descriptions that ``main_text_ar`` alone
    omits. For B2/B1 fixed-window chunks the extra fields are empty so the result
    is identical to ``_chunk_text``.
    """
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


def _page_range(chunk: Dict[str, Any]) -> tuple[int, int]:
    page_range = chunk.get("page_range")
    if isinstance(page_range, list) and len(page_range) == 2:
        try:
            return int(page_range[0]), int(page_range[1])
        except (TypeError, ValueError):
            pass
    page = chunk.get("page_number")
    if isinstance(page, int):
        return page, page
    return -1, -1


def _context_text(
    chunk_id: str,
    chunks_by_id: Dict[str, Dict[str, Any]],
    metadata: Dict[str, Any],
) -> str:
    """Prefer the local cache text; fall back to Pinecone's stored text.

    When the chunk carries structured enrichment fields (math expressions, diagram
    descriptions, or unit/lesson titles) beyond ``main_text_ar``, the full text is
    reconstructed so the answer generator sees the same content that was embedded.
    """
    chunk = chunks_by_id.get(chunk_id)
    if chunk:
        text = _chunk_text(chunk)
        if not text:
            return str(metadata.get("content_text_ar") or "")
        # Reconstruct the full representation when the chunk has fields that
        # ``main_text_ar`` alone would drop (math, diagrams, titles).
        if any([chunk.get("math_expressions"), chunk.get("diagrams"),
                chunk.get("unit_title_ar"), chunk.get("lesson_title_ar"),
                chunk.get("heading_ar")]):
            return _full_chunk_text(chunk)
        return text
    return str(metadata.get("content_text_ar") or "")


# ════════════════════════════════════════════
# FACTORIES
# ════════════════════════════════════════════


def build_query_embedder(config: cfg.RetrievalExperimentConfig) -> QueryEmbedder:
    """Instantiate the embedder implied by ``config.retriever``."""
    if config.retriever == RETRIEVER_LOCAL:
        return LocalQueryEmbedder()
    return GeminiQueryEmbedder(config)


def build_retriever(
    config: cfg.RetrievalExperimentConfig,
    chunks_by_id: Dict[str, Dict[str, Any]],
) -> Retriever:
    """Instantiate the retriever named by ``config.retriever``."""
    if config.retriever == RETRIEVER_LOCAL:
        return LocalRetriever(config, chunks_by_id)
    if config.retriever == RETRIEVER_PINECONE:
        return PineconeRetriever(config, chunks_by_id)
    raise ValueError(
        f"Unknown retriever {config.retriever!r} (expected "
        f"{RETRIEVER_PINECONE!r} or {RETRIEVER_LOCAL!r})"
    )


def check_environment(config: cfg.RetrievalExperimentConfig) -> Optional[str]:
    """Return an error message when the run's required env vars are missing."""
    if config.retriever == RETRIEVER_LOCAL:
        return None
    env = config.env
    missing = [
        name
        for name in (env.google_project, env.pinecone_api_key)
        if not os.environ.get(name)
    ]
    if missing:
        return (
            f"Missing required environment variables: {', '.join(missing)}. "
            f"Use --retriever {RETRIEVER_LOCAL} for an offline smoke test."
        )
    return None


def fail_fast(message: str) -> None:
    """Print an environment error and exit 2, matching ragkit.pipeline."""
    print(message, file=sys.stderr)
    sys.exit(2)
