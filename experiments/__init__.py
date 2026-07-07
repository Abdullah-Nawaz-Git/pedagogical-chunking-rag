"""Experiment entry points.

Each module composes ``ragkit`` into one concrete experiment:

- ``proposed`` Gemini structured extraction + pedagogical chunking (main).
- ``b1``       Tesseract OCR + fixed-window chunking.
- ``b2``       Gemini structured extraction + fixed-window chunking.

The three modules differ only in their ``ExperimentConfig``; all behaviour
lives in ``ragkit``. Run any of them with ``python -m experiments.<name>``.
"""
