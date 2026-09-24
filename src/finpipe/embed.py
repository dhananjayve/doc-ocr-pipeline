"""Batched sentence embeddings (normalized, so cosine == dot product)."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from finpipe.config import Settings


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, settings: Settings):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embed_model, device=settings.embed_device)
        self.batch_size = settings.embed_batch_size
        self.query_prefix = settings.query_prefix

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            self.query_prefix + text, normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
