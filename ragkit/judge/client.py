"""
ragkit.judge.client
═══════════════════

Pluggable judge providers behind one interface, mirroring
``ragkit.qa.llm_provider`` / ``ragkit.retrieval.answer``.

    * ``vertex_anthropic`` — Claude (Sonnet) via Vertex AI, the DEFAULT judge.
      Uses ``anthropic.AnthropicVertex``; the model id comes from ``JUDGE_MODEL``
      (never hard-coded) and credentials come from the ambient GCP environment.
    * ``vertex_gemini``    — a configurable alternative that runs a Gemini model
      through the same ``google-genai`` client the rest of ragkit uses, with
      JSON-only output requested.
    * ``mock``             — deterministic, offline. Fabricates a strictly-valid
      JSON score from a stable hash of the prompt so the whole pipeline
      (score → parse → aggregate → report) runs with no credentials. Its scores
      are NOT a quality signal.

Every provider takes the shared system prompt + a per-dimension user prompt and
returns a :class:`JudgeRawResult` carrying the raw text (for the shared parser)
plus any transport error. Retries and parsing live in ``ragkit.judge.scoring`` so
each provider stays a thin, single-call adapter. Decoding uses the configured
low-variance temperature (0 by default) so reruns are stable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional, Protocol

from .. import config as cfg

logger = logging.getLogger("ragkit.judge.client")


@dataclass(frozen=True)
class JudgeRawResult:
    """One raw judge reply plus transport provenance (pre-parse)."""

    raw_response: str
    provider: str
    model: str
    error: Optional[str] = None


class JudgeProvider(Protocol):
    """Produces one raw judge reply for a (system, user) prompt pair."""

    name: str
    model: str

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeRawResult:
        ...


# ════════════════════════════════════════════
# CLAUDE VIA VERTEX (default)
# ════════════════════════════════════════════


class VertexAnthropicJudge:
    """Claude (Sonnet) judge via Vertex AI (``anthropic.AnthropicVertex``)."""

    name = cfg.JUDGE_PROVIDER_VERTEX_ANTHROPIC

    def __init__(self, config: cfg.JudgeExperimentConfig) -> None:
        model_cfg = config.model
        judge_env = config.judge_env
        env = config.env

        # Model id from the environment; falls back to the configured default.
        self.model = os.environ.get(model_cfg.model_env_var, model_cfg.model_default)
        self._temperature = model_cfg.temperature
        self._max_tokens = model_cfg.max_output_tokens

        project = os.environ.get(env.google_project)
        if not project:
            raise RuntimeError(
                f"{env.google_project} must be set to use the {self.name} judge"
            )
        # Claude on Vertex is served from a subset of regions, so it has its own
        # optional override before falling back to the shared GCP location.
        region = (
            os.environ.get(judge_env.anthropic_location)
            or os.environ.get(env.google_location)
            or judge_env.anthropic_location_default
        )
        try:
            from anthropic import AnthropicVertex  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'anthropic' package is required for the vertex_anthropic judge. "
                "Install it (pip install 'anthropic[vertex]') or use --provider mock "
                "for an offline run."
            ) from exc

        self._client = AnthropicVertex(project_id=project, region=region)
        logger.info(
            "Anthropic-Vertex judge ready (model=%r, region=%r, temperature=%s)",
            self.model, region, self._temperature,
        )

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeRawResult:
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = _anthropic_text(message)
            return JudgeRawResult(
                raw_response=text,
                provider=self.name,
                model=self.model,
                error=None if text.strip() else "empty response",
            )
        except Exception as exc:  # noqa: BLE001 - record, never crash the run
            logger.warning("anthropic-vertex judge error: %s", exc)
            return JudgeRawResult(
                raw_response="",
                provider=self.name,
                model=self.model,
                error=f"request error: {exc}",
            )


def _anthropic_text(message: object) -> str:
    """Concatenate the text blocks of an Anthropic message response."""
    content = getattr(message, "content", None) or []
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
        elif isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    return "".join(parts).strip()


# ════════════════════════════════════════════
# GEMINI VIA VERTEX (configurable alternative)
# ════════════════════════════════════════════


class VertexGeminiJudge:
    """Gemini judge via Vertex AI (google-genai), with JSON-only output."""

    name = cfg.JUDGE_PROVIDER_VERTEX_GEMINI

    def __init__(self, config: cfg.JudgeExperimentConfig) -> None:
        model_cfg = config.model
        env = config.env

        self.model = os.environ.get(model_cfg.model_env_var, model_cfg.model_default)
        self._temperature = model_cfg.temperature
        self._max_tokens = model_cfg.max_output_tokens
        self._json_only = model_cfg.response_json_only

        project = os.environ.get(env.google_project)
        location = os.environ.get(env.google_location, env.google_location_default)
        if not project:
            raise RuntimeError(
                f"{env.google_project} must be set to use the {self.name} judge"
            )
        from google import genai  # type: ignore

        self._genai = genai
        self._client = genai.Client(vertexai=True, project=project, location=location)
        logger.info(
            "Gemini-Vertex judge ready (model=%r, temperature=%s)",
            self.model, self._temperature,
        )

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeRawResult:
        from google.genai import types  # type: ignore

        try:
            gen_config = types.GenerateContentConfig(
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                system_instruction=system_prompt,
                response_mime_type="application/json" if self._json_only else None,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_text(text=user_prompt)],
                config=gen_config,
            )
            text = (response.text or "").strip()
            return JudgeRawResult(
                raw_response=text,
                provider=self.name,
                model=self.model,
                error=None if text else "empty response",
            )
        except Exception as exc:  # noqa: BLE001 - record, never crash the run
            logger.warning("gemini-vertex judge error: %s", exc)
            return JudgeRawResult(
                raw_response="",
                provider=self.name,
                model=self.model,
                error=f"request error: {exc}",
            )


# ════════════════════════════════════════════
# MOCK (deterministic, offline)
# ════════════════════════════════════════════


class MockJudge:
    """Deterministic offline judge: a stable hash of the prompt → a valid score.

    Exists so the entire pipeline can be exercised (including parsing, resume,
    aggregation, and the paired comparison) with no credentials. The scores are
    reproducible but meaningless; they must never be read as a quality signal.
    """

    name = cfg.JUDGE_PROVIDER_MOCK
    model = "mock-judge-deterministic-1"

    def __init__(self, _config: cfg.JudgeExperimentConfig) -> None:
        pass

    def judge(self, system_prompt: str, user_prompt: str) -> JudgeRawResult:
        digest = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()
        score = round((int(digest[:8], 16) % 1001) / 1000.0, 3)
        confidence = round((int(digest[8:12], 16) % 1001) / 1000.0, 3)
        payload = {
            "score": score,
            "rationale": "deterministic mock judge score derived from the prompt hash; not a quality signal",
            "confidence": confidence,
        }
        return JudgeRawResult(
            raw_response=json.dumps(payload, ensure_ascii=False),
            provider=self.name,
            model=self.model,
            error=None,
        )


# ════════════════════════════════════════════
# FACTORY + ENVIRONMENT CHECK
# ════════════════════════════════════════════


def build_judge_provider(config: cfg.JudgeExperimentConfig) -> JudgeProvider:
    """Instantiate the provider named by ``config.model.provider``."""
    provider = (config.model.provider or cfg.JUDGE_PROVIDER_MOCK).lower()
    if provider == cfg.JUDGE_PROVIDER_MOCK:
        return MockJudge(config)
    if provider == cfg.JUDGE_PROVIDER_VERTEX_ANTHROPIC:
        return VertexAnthropicJudge(config)
    if provider == cfg.JUDGE_PROVIDER_VERTEX_GEMINI:
        return VertexGeminiJudge(config)
    raise ValueError(
        f"Unknown judge provider {config.model.provider!r} (expected "
        f"{cfg.JUDGE_PROVIDER_MOCK!r}, {cfg.JUDGE_PROVIDER_VERTEX_ANTHROPIC!r}, "
        f"or {cfg.JUDGE_PROVIDER_VERTEX_GEMINI!r})"
    )


def resolved_model_name(config: cfg.JudgeExperimentConfig) -> str:
    """The model id that will actually be used (for manifests/reports)."""
    if config.model.provider == cfg.JUDGE_PROVIDER_MOCK:
        return MockJudge.model
    return os.environ.get(config.model.model_env_var, config.model.model_default)


def check_environment(config: cfg.JudgeExperimentConfig) -> Optional[str]:
    """Return an error message when the run's required env vars are missing."""
    if config.model.provider == cfg.JUDGE_PROVIDER_MOCK:
        return None
    env = config.env
    if not os.environ.get(env.google_project):
        return (
            f"Missing required environment variable: {env.google_project}. "
            f"Use --provider {cfg.JUDGE_PROVIDER_MOCK} for an offline smoke test."
        )
    return None


def fail_fast(message: str) -> None:
    """Print an environment error and exit 2, matching the retrieval pipeline."""
    print(message, file=sys.stderr)
    sys.exit(2)
