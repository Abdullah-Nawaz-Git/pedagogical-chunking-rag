"""Page-level text extraction backends.

Two backends are provided, matching the two extraction strategies under study:

- ``gemini``    Structured vision extraction (proposed pipeline and B3).
- ``tesseract`` Plain OCR text extraction (baselines B1 and B2).
"""

from importlib import import_module

__all__ = ["gemini", "tesseract"]


def __getattr__(name: str):
	if name in __all__:
		module = import_module(f"{__name__}.{name}")
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
