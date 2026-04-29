"""
adapters/postgres_search_repo.py — asyncpg implementation of ISearchRepository.
"""
from __future__ import annotations

import asyncpg
import numpy as np

from app.embedding.ports.search_repository import ISearchRepository, SearchResult


class PostgresSearchRepository(ISearchRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def similarity_search(self, vector: np.ndarray, top_k: int) -> list[SearchResult]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.section_heading,
                    c.content,
                    p.url,
                    pm.title,
                    1 - (c.embedding <=> $1) AS score
                FROM   chunks c
                JOIN   pages p          ON p.uuid = c.page_uuid
                JOIN   page_metadata pm ON pm.page_uuid = p.uuid
                WHERE  c.embedding IS NOT NULL
                  AND  p.indexable = true
                ORDER  BY c.embedding <=> $1
                LIMIT  $2
                """,
                vector,
                top_k,
            )
        return [
            SearchResult(
                section_heading=r["section_heading"] or "",
                content=r["content"],
                url=r["url"],
                title=r["title"],
                score=float(r["score"]),
            )
            for r in rows
        ]
