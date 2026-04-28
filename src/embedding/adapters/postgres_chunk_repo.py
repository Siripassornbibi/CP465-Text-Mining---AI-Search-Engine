"""
adapters/postgres_chunk_repo.py — asyncpg implementation of IChunkRepository.
"""
from __future__ import annotations

import asyncpg
import numpy as np

from embedding.domain.chunk import Chunk
from embedding.ports.chunk_repository import CURRENT_EMBEDDING_VERSION, IChunkRepository


class PostgresChunkRepository(IChunkRepository):
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_unembedded_by_page(self, page_id: str, version: int = CURRENT_EMBEDDING_VERSION) -> list[Chunk]:
        """
        Return all unembedded chunks for a page, joining page_metadata
        to get the title and description for richer embedding context.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.uuid,
                    c.section_heading,
                    c.content,
                    COALESCE(pm.title, '')          AS page_title,
                    COALESCE(pm.description, '') 	AS page_description
                FROM   chunks c
                JOIN   page_metadata pm ON pm.page_uuid = c.page_uuid
                WHERE  c.page_uuid   = $1
                  AND  (c.embedding IS NULL OR c.embedding_version < $2)
                ORDER  BY c.chunk_index
                """,
                page_id,
				version,
            )
        return [
            Chunk(
                id=str(row["uuid"]),
                section_heading=row["section_heading"] or "",
                content=row["content"],
                page_title=row["page_title"],
                page_description=row["page_description"],
            )
            for row in rows
        ]

    async def save_embedding(self, chunk_id: str, embedding: np.ndarray, version: int = CURRENT_EMBEDDING_VERSION) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE chunks SET embedding = $1, embedding_version = $2 WHERE uuid = $3",
                embedding,
				version,
                chunk_id,
            )