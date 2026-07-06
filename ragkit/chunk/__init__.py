"""Chunking strategies.

Each module turns extracted content into the list of records that get embedded
and upserted:

- ``pedagogical`` Block-aware chunking over Gemini structured output (proposed).
- ``fixed``       Fixed-size token windows with overlap (B1 over OCR, B2 over
                  the proposed representation stream).
"""

from importlib import import_module

__all__ = ["fixed", "pedagogical"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
