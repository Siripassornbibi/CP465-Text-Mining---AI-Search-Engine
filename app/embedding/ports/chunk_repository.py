"""
ports/chunk_repository.py — IChunkRepository interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
 
from app.embedding.domain.chunk import Chunk
 
CURRENT_EMBEDDING_VERSION = 2
 
class IChunkRepository(ABC):
 
    @abstractmethod
    async def get_unembedded_by_page(self, page_id: str) -> list[Chunk]:
        """Fetch all chunks for a page that have no embedding yet."""
        ...
 
    @abstractmethod
    async def save_embedding(
        self,
        chunk_id: str,
        embedding: np.ndarray,
        version: int = CURRENT_EMBEDDING_VERSION,
    ) -> None:
        """Persist the embedding vector and version for a single chunk."""
        ...
 