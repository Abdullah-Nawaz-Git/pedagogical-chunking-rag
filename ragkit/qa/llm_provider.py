"""
ragkit.qa.llm_provider
══════════════════════

Stage 2 — pluggable QA-generation providers.

Two providers are supported:

    * ``vertex`` — Vertex AI via the same ``google-genai`` client the extraction
      stage uses. The model id is read from an environment variable (never
      hard-coded) and the API key/credentials come from the ambient GCP
      environment, so no secret is baked into the code.
    * ``mock`` — a deterministic, offline provider that fabricates strictly-valid
      candidate JSON from the task's own source evidence. It exists so the whole
      pipeline (generate → validate → finalize → map-gold) can be exercised and
      tested without any network access or credentials.

Every call returns a list of :class:`ProviderResult`, each recording the raw
response text plus full generation provenance (provider, model, temperature,
timestamp). Parsing and provenance-copying happen in ``ragkit.qa.generation`` —
providers never set gold ids.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol

from .. import config as cfg
from . import prompts

logger = logging.getLogger("ragkit.qa.provider")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProviderResult:
    """One raw generation result plus its provenance."""

    raw_response: str
    provider: str
    model: str
    temperature: float
    prompt_version: str
    timestamp_utc: str
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class QAProvider(Protocol):
    """A QA-generation provider."""

    name: str
    model: str

    def generate(self, task: Dict[str, Any], n: int, temperature: float) -> List[ProviderResult]:
        ...


# ════════════════════════════════════════════
# VERTEX AI PROVIDER
# ════════════════════════════════════════════


class VertexProvider:
    """QA generation via Vertex AI (google-genai)."""

    name = "vertex"

    def __init__(self, config: cfg.QAConfig) -> None:
        # Model id comes from the environment (falls back to the configured
        # default) — never a hard-coded model string.
        self.model = os.environ.get(
            config.generation_model_env_var, config.generation_model_default
        )
        env = cfg.EnvVars()
        project = os.environ.get(env.google_project)
        location = os.environ.get(env.google_location, env.google_location_default)
        if not project:
            raise RuntimeError(
                f"{env.google_project} must be set to use the vertex provider"
            )
        # Imported lazily so the mock provider works without the SDK installed.
        from google import genai  # type: ignore

        self._genai = genai
        self._client = genai.Client(vertexai=True, project=project, location=location)

    def generate(self, task: Dict[str, Any], n: int, temperature: float) -> List[ProviderResult]:
        from google.genai import types  # type: ignore

        user_prompt = prompts.build_user_prompt(task)
        results: List[ProviderResult] = []
        for _ in range(n):
            timestamp = _utc_now()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=[types.Part.from_text(text=user_prompt)],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=temperature,
                        system_instruction=prompts.SHARED_SYSTEM_PROMPT,
                    ),
                )
                text = (response.text or "").strip()
                results.append(
                    ProviderResult(
                        raw_response=text,
                        provider=self.name,
                        model=self.model,
                        temperature=temperature,
                        prompt_version=prompts.PROMPT_VERSION,
                        timestamp_utc=timestamp,
                        error=None if text else "empty response",
                    )
                )
            except Exception as exc:  # noqa: BLE001 - record, never crash the run
                logger.warning("vertex generation error for %s: %s", task.get("task_id"), exc)
                results.append(
                    ProviderResult(
                        raw_response="",
                        provider=self.name,
                        model=self.model,
                        temperature=temperature,
                        prompt_version=prompts.PROMPT_VERSION,
                        timestamp_utc=timestamp,
                        error=f"request error: {exc}",
                    )
                )
        return results


# ════════════════════════════════════════════
# MOCK PROVIDER (deterministic, offline)
# ════════════════════════════════════════════


class MockProvider:
    """Fabricates strictly-valid candidate JSON from a task's own evidence.

    Deterministic and offline. Questions embed a short reference tag derived from
    the source block ids so that questions are globally unique (avoiding spurious
    duplicate rejections) — this is a test/demo aid, not a real generator.
    """

    name = "mock"
    model = "mock-deterministic-1"

    def generate(self, task: Dict[str, Any], n: int, temperature: float) -> List[ProviderResult]:
        results: List[ProviderResult] = []
        for i in range(n):
            payload_obj = self._build(task, variant=i)
            results.append(
                ProviderResult(
                    raw_response=json.dumps(payload_obj, ensure_ascii=False),
                    provider=self.name,
                    model=self.model,
                    temperature=temperature,
                    prompt_version=prompts.PROMPT_VERSION,
                    timestamp_utc=_utc_now(),
                    error=None,
                )
            )
        return results

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _first_payload(task: Dict[str, Any]) -> Dict[str, Any]:
        payloads = task.get("source_payloads") or []
        if payloads:
            return payloads[0]
        return task.get("source_payload") or {}

    @staticmethod
    def _ref_tag(task: Dict[str, Any]) -> str:
        blocks = task.get("source_block_ids") or []
        return blocks[0] if blocks else (task.get("task_id") or "")

    @staticmethod
    def _clip(text: str, limit: int = 1000) -> str:
        text = (text or "").strip()
        return text[:limit] if len(text) > limit else text

    def _build(self, task: Dict[str, Any], variant: int) -> Dict[str, Any]:
        qtype = task["question_type"]
        payload = self._first_payload(task)
        tag = self._ref_tag(task)
        lesson = payload.get("lesson_number") or ""
        prefix = "أ) " if variant else ""  # keep the two candidates distinct
        base: Dict[str, Any] = {
            "question_en": None,
            "difficulty": "medium" if variant else "easy",
            "answer_mode": "extractive",
            "required_diagram": False,
            "required_formula": False,
        }

        if qtype == "definition_recall":
            named = payload.get("named_elements") or {}
            terms = named.get("definitions") or named.get("vocabulary") or []
            term = terms[0] if terms else (payload.get("heading_ar") or "المفهوم")
            if isinstance(term, dict):
                term = term.get("term_ar") or "المفهوم"
            base["question_ar"] = f"{prefix}ما تعريف {term} في الدرس {lesson}؟ (مرجع {tag})"
            base["answer_reference_ar"] = self._clip(payload.get("main_text_ar", ""))

        elif qtype == "theorem_statement":
            base["question_ar"] = f"{prefix}اذكر نص النظرية الواردة في الدرس {lesson}. (مرجع {tag})"
            base["answer_reference_ar"] = self._clip(payload.get("main_text_ar", ""))

        elif qtype == "formula_retrieval":
            expressions = payload.get("math_expressions") or []
            formula = expressions[0] if expressions else ""
            base["required_formula"] = True
            base["question_ar"] = f"{prefix}اكتب الصيغة الرياضية الواردة في الدرس {lesson}. (مرجع {tag})"
            base["answer_reference_ar"] = self._clip(f"الصيغة: {formula}. " + payload.get("main_text_ar", ""))

        elif qtype == "diagram_dependent":
            diagrams = payload.get("diagrams") or []
            desc = ""
            labels: List[Any] = []
            for diagram in diagrams:
                if (diagram.get("description") or "").strip():
                    desc = diagram["description"].strip()
                    labels = diagram.get("labels") or []
                    break
            base["required_diagram"] = True
            label_hint = ("، ".join(str(x) for x in labels[:3])) if labels else ""
            base["question_ar"] = (
                f"{prefix}بالاعتماد على الشكل الموضح ({label_hint}) في الدرس {lesson}، "
                f"صف العلاقة الهندسية المطلوبة. (مرجع {tag})"
            )
            base["answer_reference_ar"] = self._clip(desc or payload.get("main_text_ar", ""))

        elif qtype == "worked_example_reasoning":
            base["answer_mode"] = "reasoning"
            base["question_ar"] = f"{prefix}اشرح خطوات الحل المتّبعة في المثال في الدرس {lesson}. (مرجع {tag})"
            base["answer_reference_ar"] = self._clip(payload.get("main_text_ar", ""))

        else:  # pragma: no cover - defensive
            base["question_ar"] = f"سؤال حول الدرس {lesson}. (مرجع {tag})"
            base["answer_reference_ar"] = self._clip(payload.get("main_text_ar", ""))

        return base


# ════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════


def build_provider(config: cfg.QAConfig) -> QAProvider:
    """Instantiate the provider named by ``config.provider``."""
    provider_name = (config.provider or "mock").lower()
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "vertex":
        return VertexProvider(config)
    raise ValueError(f"Unknown QA provider: {config.provider!r} (expected 'mock' or 'vertex')")
