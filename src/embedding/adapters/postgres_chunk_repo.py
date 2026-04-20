"""
adapters/postgres_chunk_repo.py — asyncpg implementation of IChunkRepository.
"""
from __future__ import annotations

import asyncpg
import numpy as np

from embedding.domain.chunk import Chunk
from embedding.ports.chunk_repository import IChunkRepository


class PostgresChunkRepository(IChunkRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_unembedded_by_page(self, page_id: str) -> list[Chunk]:
        """Return all chunks belonging to page_id that have no embedding."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT uuid, section_heading, content
                FROM   chunks
                WHERE  page_uuid   = $1
                  AND  embedding IS NULL
                ORDER  BY chunk_index
                """,
                page_id,
            )
        return [
            Chunk(
                id=str(row["uuid"]),
                section_heading=row["section_heading"] or "",
                content=row["content"],
            )
            for row in rows
        ]

    async def save_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chunks SET embedding = $1 WHERE uuid = $2",
                embedding,
                chunk_id,
            )