"""
application/embed_chunk.py — EmbedChunkUseCase.
Orchestrates domain + ports. No infrastructure knowledge.
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from crawlerapp.domain.events import EmbeddingRequestedEvent, EMBEDDING_REQUESTED
from crawlerapp.ports.chunk_repository import IChunkRepository
from crawlerapp.ports.embedder import IEmbedder

logger = logging.getLogger(__name__)


class EmbedChunkUseCase:
    def __init__(
        self,
        repo: IChunkRepository,
        embedder: IEmbedder,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._executor = executor

    async def execute(self, event: EmbeddingRequestedEvent) -> None:
        if not event.is_embedding_request():
            logger.warning("Ignoring unknown event_type '%s'", event.event_type)
            return

        chunk = await self._repo.get_by_id(event.chunk_uuid)
        if chunk is None:
            logger.warning("Chunk %s not found, skipping.", event.chunk_uuid)
            return

        if chunk.is_embedded():
            logger.info("Chunk %s already embedded, skipping.", event.chunk_uuid)
            return

        logger.info("Embedding chunk %s ...", chunk.id)

        loop = asyncio.get_event_loop()
        text = chunk.prepared_text()
        embedding = await loop.run_in_executor(
            self._executor,
            lambda: self._embedder.embed(text),
        )

        await self._repo.save_embedding(chunk.id, embedding)
        logger.info("Chunk %s embedded and saved.", chunk.id)
