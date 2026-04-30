"""
ports/embedder.py — IEmbedder interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class IEmbedder(ABC):

    @abstractmethod
    def embed(self, text: str) -> tuple[np.ndarray, bool]:
        """Embed a single text, returning a float32 vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        """Embed multiple texts, returning shape (N, dims) float32 array."""
        ...

    @property
    @abstractmethod
    def dims(self) -> int:
        """Embedding dimension."""
        ...
