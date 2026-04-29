"""
application/search.py — SearchUseCase.
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from app.embedding.ports.embedder import IEmbedder
from app.embedding.ports.search_repository import ISearchRepository, SearchResult

logger = logging.getLogger(__name__)

QUERY_PREFIX = "Represent this query for searching: "

@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]


class SearchUseCase:
    def __init__(
        self,
        embedder: IEmbedder,
        search_repo: ISearchRepository,
        executor: ThreadPoolExecutor,
        top_k: int = 8,
    ) -> None:
        self._embedder = embedder
        self._search_repo = search_repo
        self._executor = executor
        self._top_k = top_k

    async def execute(self, query: str, top_k: int | None = None) -> SearchResponse:
        k = top_k or self._top_k
        loop = asyncio.get_event_loop()

        vector = await loop.run_in_executor(
            self._executor,
            lambda: self._embedder.embed(f"{QUERY_PREFIX}{query}"),
        )

        results = await self._search_repo.similarity_search(vector, k)
        return SearchResponse(query=query, results=results)
