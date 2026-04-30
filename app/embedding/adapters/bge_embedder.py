"""
adapters/bge_embedder.py — sentence-transformers implementation of IEmbedder.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.embedding.ports.embedder import IEmbedder


class BGEEmbedder(IEmbedder):
    def __init__(self, model_name: str, device: Optional[str] = None, batch_size: int = 64) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._dims = self._model.get_embedding_dimension()

    def embed(self, text: str) -> tuple[np.ndarray, bool]:
        return self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32), True

    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        return self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self._batch_size,
            convert_to_numpy=True,
        ).astype(np.float32), True

    @property
    def dims(self) -> int:
        return self._dims
