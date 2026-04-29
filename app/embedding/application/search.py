"""
application/search.py — SearchUseCase with multi-query retrieval.
Embeds all queries, merges results by chunk id, re-ranks by best score.
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from embedding.domain.chunk import QUERY_PREFIX
from embedding.ports.embedder import IEmbedder
from embedding.ports.search_repository import ISearchRepository, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class SearchResponse:
    query: str
    expanded_queries: list[str]
    results: list[SearchResult]


class SearchUseCase:
    def __init__(
        self,
        embedder: IEmbedder,
        search_repo: ISearchRepository,
        executor: ThreadPoolExecutor,
        top_k: int = 8,
    ) -> None:
        self._embedder    = embedder
        self._search_repo = search_repo
        self._executor    = executor
        self._top_k       = top_k

    async def execute(
        self,
        query: str,
        top_k: int | None = None,
        expanded_queries: list[str] | None = None,
    ) -> SearchResponse:
        """
        Embeds all queries (original + expanded), retrieves top_k per query,
        merges by chunk URL deduplication, and re-ranks by best score.
        """
        k = top_k or self._top_k
        all_queries = expanded_queries if expanded_queries else [query]
        loop = asyncio.get_event_loop()

        # embed all queries concurrently in the thread pool
        async def embed_query(q: str):
            return await loop.run_in_executor(
                self._executor,
                lambda: self._embedder.embed(f"{QUERY_PREFIX}{q}"),
            )

        vectors = await asyncio.gather(*[embed_query(q) for q in all_queries])

        # retrieve top_k results per query concurrently
        retrievals = await asyncio.gather(*[
            self._search_repo.similarity_search(vec, k)
            for vec in vectors
        ])

        # merge — keep best score per unique (url, section_heading) pair
        best: dict[str, SearchResult] = {}
        for results in retrievals:
            for r in results:
                key = f"{r.url}::{r.section_heading}"
                if key not in best or r.score > best[key].score:
                    best[key] = r

        # sort by score descending, cap at top_k
        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)[:k]

        logger.info(
            "Multi-query retrieval: %d queries → %d unique chunks (top %d)",
            len(all_queries), len(best), len(merged),
        )

        return SearchResponse(
            query=query,
            expanded_queries=all_queries[1:],  # excludes original
            results=merged,
        )