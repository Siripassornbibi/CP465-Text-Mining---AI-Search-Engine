"""
ports/search_repository.py — ISearchRepository interface.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class SearchResult:
    section_heading: str
    content: str
    url: str
    title: str
    score: float


class ISearchRepository(ABC):

    @abstractmethod
    async def similarity_search(self, vector: np.ndarray, top_k: int) -> list[SearchResult]:
        """Return top_k chunks ordered by cosine similarity to vector."""
        ...
