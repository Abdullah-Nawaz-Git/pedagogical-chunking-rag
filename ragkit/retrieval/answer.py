"""
ragkit.retrieval.answer
═══════════════════════

Shared context assembly + answer generation for every system.

Three things are deliberately held constant here so the generator can never
become a confound in the comparison:

    * ONE prompt (``SHARED_SYSTEM_PROMPT`` + ``build_user_prompt``),
    * ONE model (from ``AnswerGenerationConfig``, read via an env var),
    * ONE context budget (``context_budget_tokens``).

Context budget + token accounting
---------------------------------
Retrieved chunks are concatenated in rank order until the budget is reached; the
chunk that would overflow it is truncated at the token level rather than dropped,
so every system receives the same-sized context window regardless of its chunk
granularity. This is what stops B1/B2's 512-token windows from buying an
advantage (or a penalty) purely through chunk size.

Tokens are counted as whitespace-separated units — the same convention
``FixedChunkConfig`` uses for chunking — and the ACTUAL count passed to the
generator is recorded on every record (``context_token_count``), alongside a
character count and an estimated model-token count.

Providers mirror ``ragkit.qa.llm_provider``: ``vertex`` for real runs and a
deterministic offline ``mock`` so the pipeline is runnable without credentials.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .. import config as cfg
from .engine import RetrievedChunk

logger = logging.getLogger("ragkit.retrieval.answer")

PROMPT_VERSION = "v1"

PROVIDER_MOCK = "mock"
PROVIDER_VERTEX = "vertex"


# ════════════════════════════════════════════
# SHARED PROMPT (identical for all systems)
# ════════════════════════════════════════════

SHARED_SYSTEM_PROMPT = """You are answering a question about a Grade 10 Arabic mathematics textbook.

Use ONLY the supplied retrieved context.
Do not use external mathematical knowledge and do not invent values, formulas, labels, or steps.
If the retrieved context does not contain the answer, reply exactly: لا يوجد في السياق المُقدَّم.
Answer in Modern Standard Arabic, as concisely as the question allows.
Preserve mathematical notation exactly as it appears in the context."""


def build_user_prompt(question_ar: str, context: str) -> str:
    """Assemble the shared user prompt from the question + retrieved context."""
    return (
        "--- Retrieved context ---\n"
        f"{context}\n\n"
        "--- Question ---\n"
        f"{question_ar}\n\n"
        "Answer in Arabic using only the retrieved context."
    )


# ════════════════════════════════════════════
# CONTEXT ASSEMBLY + TOKEN ACCOUNTING
# ════════════════════════════════════════════


@dataclass(frozen=True)
class AssembledContext:
    """The exact context handed to the generator, plus its measured size."""

    text: str
    chunk_ids: tuple[str, ...]
    token_count: int
    budget_tokens: int
    truncated: bool
    char_count: int
    estimated_model_tokens: int


def assemble_context(
    retrieved: Sequence[RetrievedChunk],
    config: cfg.AnswerGenerationConfig,
) -> AssembledContext:
    """Concatenate retrieved chunks in rank order under the shared token budget.

    Each chunk is prefixed with a rank marker so the generator can attribute
    content, and the marker's tokens are counted toward the budget too — the
    recorded ``token_count`` is exactly what the prompt carried.
    """
    budget = config.context_budget_tokens
    used_tokens = 0
    pieces: List[str] = []
    chunk_ids: List[str] = []
    truncated = False

    for hit in retrieved:
        if used_tokens >= budget:
            truncated = True
            break

        header_tokens = f"[{hit.rank}] {hit.chunk_id}".split()
        body_tokens = (hit.text or "").split()
        remaining = budget - used_tokens

        if len(header_tokens) >= remaining:
            # Not even the marker fits; stop cleanly.
            truncated = True
            break

        allowance = remaining - len(header_tokens)
        if len(body_tokens) > allowance:
            body_tokens = body_tokens[:allowance]
            truncated = True

        pieces.append(" ".join(header_tokens + body_tokens))
        used_tokens += len(header_tokens) + len(body_tokens)
        chunk_ids.append(hit.chunk_id)

    text = "\n\n".join(pieces)
    chars = len(text)
    return AssembledContext(
        text=text,
        chunk_ids=tuple(chunk_ids),
        # Recount from the final string so the number is measured, not predicted.
        token_count=len(text.split()),
        budget_tokens=budget,
        truncated=truncated,
        char_count=chars,
        estimated_model_tokens=(
            (chars + config.chars_per_token - 1) // config.chars_per_token
        ),
    )


# ════════════════════════════════════════════
# PROVIDERS
# ════════════════════════════════════════════


@dataclass
class AnswerResult:
    """One generated answer plus its provenance."""

    answer_ar: str
    provider: str
    model: str
    prompt_version: str
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class AnswerProvider(Protocol):
    """Generates one Arabic answer from a question + retrieved context."""

    name: str
    model: str

    def generate(self, question_ar: str, context: AssembledContext) -> AnswerResult:
        ...


class VertexAnswerProvider:
    """Answer generation via Vertex AI, shared verbatim by all three systems."""

    name = PROVIDER_VERTEX

    def __init__(self, config: cfg.RetrievalExperimentConfig) -> None:
        answer_config = config.answer
        # Model id comes from the environment — never hard-coded.
        self.model = os.environ.get(
            answer_config.model_env_var, answer_config.model_default
        )
        env = config.env
        project = os.environ.get(env.google_project)
        location = os.environ.get(env.google_location, env.google_location_default)
        if not project:
            raise RuntimeError(
                f"{env.google_project} must be set to use the vertex answer provider"
            )
        from google import genai

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._temperature = answer_config.temperature

    def generate(self, question_ar: str, context: AssembledContext) -> AnswerResult:
        from google.genai import types

        prompt = build_user_prompt(question_ar, context.text)
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_text(text=prompt)],
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    system_instruction=SHARED_SYSTEM_PROMPT,
                ),
            )
            text = (response.text or "").strip()
            return AnswerResult(
                answer_ar=text,
                provider=self.name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                error=None if text else "empty response",
            )
        except Exception as exc:  # noqa: BLE001 - record, never crash the run
            logger.warning("vertex answer generation error: %s", exc)
            return AnswerResult(
                answer_ar="",
                provider=self.name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                error=f"request error: {exc}",
            )


class MockAnswerProvider:
    """Deterministic offline provider: echoes the head of the retrieved context.

    Exists so ``--generate-answers`` can be exercised (including the context
    budget and token accounting) with no credentials. It performs no reasoning and
    its output must not be read as a quality signal.
    """

    name = PROVIDER_MOCK
    model = "mock-deterministic-1"

    def __init__(self, _config: cfg.RetrievalExperimentConfig) -> None:
        pass

    def generate(self, question_ar: str, context: AssembledContext) -> AnswerResult:
        if not context.text.strip():
            return AnswerResult(
                answer_ar="لا يوجد في السياق المُقدَّم.",
                provider=self.name,
                model=self.model,
                prompt_version=PROMPT_VERSION,
                error=None,
            )
        # First 40 whitespace tokens of the highest-ranked context, verbatim.
        head = " ".join(context.text.split()[:40])
        return AnswerResult(
            answer_ar=head,
            provider=self.name,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            error=None,
        )


def build_answer_provider(config: cfg.RetrievalExperimentConfig) -> AnswerProvider:
    """Instantiate the provider named by ``config.answer.provider``."""
    provider = (config.answer.provider or PROVIDER_MOCK).lower()
    if provider == PROVIDER_MOCK:
        return MockAnswerProvider(config)
    if provider == PROVIDER_VERTEX:
        return VertexAnswerProvider(config)
    raise ValueError(
        f"Unknown answer provider {config.answer.provider!r} "
        f"(expected {PROVIDER_MOCK!r} or {PROVIDER_VERTEX!r})"
    )
