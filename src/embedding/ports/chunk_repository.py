"""
ports/chunk_repository.py — IChunkRepository interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
 
from embedding.domain.chunk import Chunk
 
 
class IChunkRepository(ABC):
 
    @abstractmethod
    async def get_unembedded_by_page(self, page_id: str) -> list[Chunk]:
        """Fetch all chunks for a page that have no embedding yet."""
        ...
 
    @abstractmethod
    async def save_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Persist the embedding vector for a single chunk."""
        ...