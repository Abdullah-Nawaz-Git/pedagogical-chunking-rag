"""
ragkit.qa.normalize
════════════════════

Arabic text normalisation used by banned-phrase detection and duplicate
question comparison.

The normalisation is intentionally conservative so it never mangles the
mathematical notation embedded in questions/answers (LaTeX commands, digits,
operators). It only:

    * collapses runs of whitespace to a single space,
    * removes the tatweel/kashida elongation character (ـ),
    * unifies the Alef variants (أ إ آ ٱ) to bare Alef (ا).

A second, slightly heavier form (:func:`normalize_for_compare`) additionally
folds trailing punctuation and case for robust duplicate detection, while still
leaving backslashes, digits, and math operators untouched.
"""

from __future__ import annotations

import re

# Unicode code points involved in normalisation.
_TATWEEL = "\u0640"  # ARABIC TATWEEL (ـ)

# Alef variants → bare Alef.
_ALEF_VARIANTS = {
    "\u0623": "\u0627",  # أ  ALEF WITH HAMZA ABOVE
    "\u0625": "\u0627",  # إ  ALEF WITH HAMZA BELOW
    "\u0622": "\u0627",  # آ  ALEF WITH MADDA ABOVE
    "\u0671": "\u0627",  # ٱ  ALEF WASLA
}
_ALEF_TABLE = {ord(k): v for k, v in _ALEF_VARIANTS.items()}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Apply the core, notation-preserving Arabic normalisation.

    Whitespace is collapsed, tatweel is stripped, and Alef variants are unified.
    Mathematical notation (LaTeX, digits, symbols) is deliberately preserved.
    """
    if not text:
        return ""
    text = text.replace(_TATWEEL, "")
    text = text.translate(_ALEF_TABLE)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# Punctuation folded only for duplicate comparison (Arabic + ASCII sentence
# punctuation). Math operators (=, +, -, ^, /, \, {}, digits) are NOT included.
_COMPARE_PUNCT_RE = re.compile(r"[\u060C\u061B\u061F\.,;:!?\u066A\u066B\u066C\"'`()]+")


def normalize_for_compare(text: str) -> str:
    """Normalise for duplicate detection: core normalisation + punctuation fold.

    Builds on :func:`normalize_arabic`, then removes sentence punctuation and
    lowercases Latin characters so that questions differing only in trailing
    punctuation or letter case are recognised as duplicates. Math content is
    still preserved.
    """
    normalized = normalize_arabic(text)
    normalized = _COMPARE_PUNCT_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip().lower()


def contains_banned_phrase(question: str, banned_phrases: "list[str] | tuple[str, ...]") -> "str | None":
    """Return the first banned phrase present in ``question`` (normalised), else None."""
    normalized_q = normalize_arabic(question)
    for phrase in banned_phrases:
        if normalize_arabic(phrase) in normalized_q:
            return phrase
    return None
