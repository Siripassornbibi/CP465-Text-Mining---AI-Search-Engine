"""
application/embed_chunk.py — EmbedChunkUseCase.
Receives a page_uuid, fetches all unembedded chunks for that page,
embeds each one, and saves back to the database.
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from embedding.domain.events import EmbeddingRequestedEvent
from embedding.ports.chunk_repository import IChunkRepository
from embedding.ports.embedder import IEmbedder

logger = logging.getLogger(__name__)


class EmbedChunkUseCase:
    def __init__(
        self,
        repo: IChunkRepository,
        embedder: IEmbedder,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._repo     = repo
        self._embedder = embedder
        self._executor = executor

    async def execute(self, event: EmbeddingRequestedEvent) -> None:
        if not event.is_embedding_request():
            logger.warning("Ignoring unknown event_type '%s'", event.event_type)
            return

        chunks = await self._repo.get_unembedded_by_page(event.page_uuid)

        if not chunks:
            logger.info("No unembedded chunks for page %s, skipping.", event.page_uuid)
            return

        logger.info("Embedding %d chunk(s) for page %s ...", len(chunks), event.page_uuid)

        loop = asyncio.get_event_loop()

        for chunk in chunks:
            text = chunk.prepared_text()
            embedding = await loop.run_in_executor(
                self._executor,
                lambda t=text: self._embedder.embed(t),
            )
            await self._repo.save_embedding(chunk.id, embedding)
            logger.info("Saved embedding for chunk %s", chunk.id)

        logger.info("Done — %d chunk(s) embedded for page %s.", len(chunks), event.page_uuid)