"""
ragkit.judge.parser
═══════════════════

The ONE parser for judge replies, shared by every dimension and provider.

Judges are asked for STRICT JSON (``score`` / ``rationale`` / ``confidence``).
Real models occasionally wrap that JSON in prose or code fences, so this module
extracts the first balanced JSON object, then validates the score range. It never
silently coerces or hides a failure: an out-of-range score is reported as
``invalid_score`` (not clamped), an unparseable reply as ``parse_error``, and the
raw text is preserved on the record either way so every verdict is auditable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

from . import schemas

# Matches an opening ```json / ``` fence and a trailing fence, so we can strip
# them before locating the JSON object.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

# Recovers the ``score`` field from a truncated JSON reply (e.g. the rationale
# string was cut off mid-token by the provider) so the value is never lost.
_SCORE_FIELD_RE = re.compile(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)')


@dataclass(frozen=True)
class ParsedScore:
    """The outcome of parsing one judge reply."""

    status: str
    score: Optional[float]
    rationale: str
    confidence: Optional[float]
    parse_ok: bool
    error: Optional[str] = None
    # Whatever value the model placed in "score", even when it was invalid, kept
    # for auditing without letting it contaminate the aggregates.
    raw_score: Any = None


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block in ``text``, or ``None``.

    Balancing braces (rather than a greedy regex) tolerates nested objects such
    as a structured ``confidence`` payload without over-matching trailing prose.
    """
    cleaned = _FENCE_RE.sub("", text or "").strip()
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return None


def _coerce_optional_float(value: Any) -> Optional[float]:
    """Best-effort float for the optional ``confidence`` field only."""
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:  # NaN
        return None
    return result


def parse_judge_response(raw: str) -> ParsedScore:
    """Parse a judge reply into a validated :class:`ParsedScore`.

    Range validation is strict: a numeric score outside ``[0.0, 1.0]`` yields
    ``invalid_score`` with ``score=None`` (never clamped), so it is excluded from
    the aggregates rather than quietly distorting them.

    When the reply is truncated (unbalanced braces / unterminated string) but a
    valid numeric ``score`` is still present, the score is recovered as
    ``recovered`` rather than dropped as ``parse_error`` — the raw text is kept
    on the record so the verdict remains auditable.
    """
    if not (raw or "").strip():
        return ParsedScore(
            status=schemas.STATUS_PARSE_ERROR,
            score=None,
            rationale="",
            confidence=None,
            parse_ok=False,
            error="empty judge response",
        )

    payload = _extract_json_object(raw)
    if payload is None:
        # Truncated reply: try to recover just the score field.
        match = _SCORE_FIELD_RE.search(raw)
        if match is not None:
            try:
                recovered = float(match.group(1))
            except (TypeError, ValueError):
                recovered = float("nan")
            if recovered == recovered and 0.0 <= recovered <= 1.0:
                return ParsedScore(
                    status=schemas.STATUS_RECOVERED,
                    score=recovered,
                    rationale="",
                    confidence=None,
                    parse_ok=True,
                    error="truncated judge response; score recovered",
                    raw_score=recovered,
                )
        return ParsedScore(
            status=schemas.STATUS_PARSE_ERROR,
            score=None,
            rationale="",
            confidence=None,
            parse_ok=False,
            error="no JSON object found in judge response",
        )

    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        return ParsedScore(
            status=schemas.STATUS_PARSE_ERROR,
            score=None,
            rationale="",
            confidence=None,
            parse_ok=False,
            error=f"json decode error: {exc}",
        )

    if not isinstance(obj, dict):
        return ParsedScore(
            status=schemas.STATUS_PARSE_ERROR,
            score=None,
            rationale="",
            confidence=None,
            parse_ok=False,
            error="judge response JSON was not an object",
        )

    raw_score = obj.get("score")
    rationale = str(obj.get("rationale") or "").strip()
    confidence = _coerce_optional_float(obj.get("confidence"))

    if "score" not in obj:
        return ParsedScore(
            status=schemas.STATUS_INVALID_SCORE,
            score=None,
            rationale=rationale,
            confidence=confidence,
            parse_ok=True,
            error="judge response JSON had no 'score' key",
            raw_score=raw_score,
        )

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return ParsedScore(
            status=schemas.STATUS_INVALID_SCORE,
            score=None,
            rationale=rationale,
            confidence=confidence,
            parse_ok=True,
            error=f"non-numeric score: {raw_score!r}",
            raw_score=raw_score,
        )

    if score != score or not (0.0 <= score <= 1.0):  # NaN or out of range
        return ParsedScore(
            status=schemas.STATUS_INVALID_SCORE,
            score=None,
            rationale=rationale,
            confidence=confidence,
            parse_ok=True,
            error=f"score out of range [0,1]: {score}",
            raw_score=raw_score,
        )

    return ParsedScore(
        status=schemas.STATUS_SCORED,
        score=score,
        rationale=rationale,
        confidence=confidence,
        parse_ok=True,
        error=None,
        raw_score=raw_score,
    )
