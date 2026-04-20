"""
ports/chunk_repository.py — IChunkRepository interface.
The application layer depends on this abstraction, not on asyncpg directly.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from crawlerapp.domain.chunk import Chunk


class IChunkRepository(ABC):

    @abstractmethod
    async def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Fetch a single chunk by UUID. Returns None if not found."""
        ...

    @abstractmethod
    async def save_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Persist the embedding vector for a chunk."""
        ...
