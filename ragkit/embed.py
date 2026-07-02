"""
ragkit.embed
════════════

Stage 3B — embed text strings using the Gemini embedding model.

Shared by every experiment unchanged so the embedding model is held constant
across the proposed system and all baselines. (B2 calls this twice: once to
embed individual sentences when locating semantic breakpoints, and again to
embed the final chunk texts that are stored in Pinecone.)
"""

from __future__ import annotations

from typing import List

from google import genai
from google.genai import types

from .config import EmbeddingConfig


def embed_texts(
    client: genai.Client,
    texts: List[str],
    config: EmbeddingConfig,
) -> List[List[float]]:
    """
    Embed a list of text strings using the Gemini embedding model.

    Texts are sent in batches of ``config.batch_size`` (the API maximum) to
    avoid request-size limits. Each batch produces a list of ``config.dim``-
    dimensional float vectors.

    Returns a list of embedding vectors in the same order as the input texts.
    """
    if not texts:
        return []

    batch_size = config.batch_size
    all_vectors: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        resp = client.models.embed_content(
            model=config.model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=config.task_type,
                output_dimensionality=config.dim,
            ),
        )

        embeddings = resp.embeddings or []

        # The API must return exactly one vector per input text
        if len(embeddings) != len(batch):
            raise ValueError(
                f"embedding count mismatch: got {len(embeddings)}, "
                f"expected {len(batch)}"
            )

        all_vectors.extend([list(e.values) for e in embeddings])

    return all_vectors
