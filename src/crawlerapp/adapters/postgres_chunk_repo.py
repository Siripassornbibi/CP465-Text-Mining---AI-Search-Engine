"""
adapters/postgres_chunk_repo.py — asyncpg implementation of IChunkRepository.
"""
from __future__ import annotations
from typing import Optional

import asyncpg
import numpy as np

from crawlerapp.domain.chunk import Chunk
from crawlerapp.ports.chunk_repository import IChunkRepository


class PostgresChunkRepository(IChunkRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_by_id(self, chunk_id: str) -> Optional[Chunk]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, section_heading, content,
                       embedding IS NOT NULL AS has_embedding
                FROM   chunks
                WHERE  id = $1
                """,
                chunk_id,
            )
        if not row:
            return None
        return Chunk(
            id=str(row["id"]),
            section_heading=row["section_heading"] or "",
            content=row["content"],
            embedding=np.zeros(1) if row["has_embedding"] else None,
        )

    async def save_embedding(self, chunk_id: str, embedding: np.ndarray) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chunks SET embedding = $1 WHERE id = $2",
                embedding,
                chunk_id,
            )
