"""
worker.py — RabbitMQ consumer for on-demand chunk embedding

Expected message format:
{
    "event_type": "prawler.embedding.id",
    "chunk_uuid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import asyncpg
import aio_pika
import aio_pika.abc
import numpy as np
from pgvector.asyncpg import register_vector

from config import AppConfig, RabbitMQConfig
from embedder import Chunk, Embedder

logger = logging.getLogger(__name__)

EXPECTED_EVENT = "prawler.embedding.id"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

async def fetch_chunk(conn: asyncpg.Connection, chunk_uuid: str) -> Optional[Chunk]:
    """Fetch a single chunk row by UUID. Returns None if not found."""
    row = await conn.fetchrow(
        """
        SELECT id, section_heading, content
        FROM   chunks
        WHERE  id = $1
        """,
        chunk_uuid,
    )
    if not row:
        return None
    return Chunk(
        id=str(row["id"]),
        section_heading=row["section_heading"] or "",
        content=row["content"],
    )


async def save_embedding(conn: asyncpg.Connection, chunk_uuid: str, vector: np.ndarray) -> None:
    """Write the embedding vector back to the chunks table."""
    await conn.execute(
        "UPDATE chunks SET embedding = $1 WHERE id = $2",
        vector,
        chunk_uuid,
    )


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

async def handle_message(
    message: aio_pika.abc.AbstractIncomingMessage,
    pool: asyncpg.Pool,
    embedder: Embedder,
) -> None:
    """
    Process a single RabbitMQ message:
    1. Parse JSON body
    2. Validate event_type
    3. Fetch chunk from DB
    4. Embed
    5. Save vector back to DB
    6. Ack (or nack on failure)
    """
    async with message.process(requeue=False):  # auto-ack on success, nack on exception
        try:
            body = json.loads(message.body.decode())
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in message: %s", e)
            return  # discard malformed messages — don't requeue

        event_type = body.get("event_type")
        chunk_uuid = body.get("chunk_uuid")

        # --- validate ---
        if event_type != EXPECTED_EVENT:
            logger.warning("Unexpected event_type '%s', skipping.", event_type)
            return

        if not chunk_uuid:
            logger.error("Message missing chunk_uuid: %s", body)
            return

        logger.info("Processing chunk %s", chunk_uuid)

        async with pool.acquire() as conn:
            # --- fetch chunk ---
            chunk = await fetch_chunk(conn, chunk_uuid)
            if chunk is None:
                logger.warning("Chunk %s not found in DB, skipping.", chunk_uuid)
                return

            # check if already embedded — avoid redundant work
            already = await conn.fetchval(
                "SELECT embedding IS NOT NULL FROM chunks WHERE id = $1", chunk_uuid
            )
            if already:
                logger.info("Chunk %s already embedded, skipping.", chunk_uuid)
                return

            # --- embed (CPU-bound, run in thread pool) ---
            loop = asyncio.get_event_loop()
            text = Embedder.prepare_chunk_text(chunk.section_heading, chunk.content)
            vector: np.ndarray = await loop.run_in_executor(
                None,
                lambda: embedder.embed_texts([text])[0],
            )

            # --- save ---
            await save_embedding(conn, chunk_uuid, vector)
            logger.info("Embedded and saved chunk %s", chunk_uuid)


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class EmbeddingWorker:
    def __init__(self, config: AppConfig):
        self.cfg = config
        self.rmq_cfg: RabbitMQConfig = config.rabbitmq
        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._pool: Optional[asyncpg.Pool] = None
        self._embedder: Optional[Embedder] = None

    async def start(self) -> None:
        logger.info("Loading embedder model %s ...", self.cfg.embedder.model)
        self._embedder = Embedder(
            model_name=self.cfg.embedder.model,
            batch_size=self.cfg.embedder.batch_size,
            device=self.cfg.embedder.device,
        )

        logger.info("Connecting to Postgres ...")
        self._pool = await asyncpg.create_pool(
            self.cfg.database_url,
            min_size=2,
            max_size=10,
            init=register_vector,
        )

        logger.info("Connecting to RabbitMQ at %s ...", self.rmq_cfg.url)
        self._connection = await aio_pika.connect_robust(self.rmq_cfg.url)
        self._channel = await self._connection.channel()

        # limit in-flight messages per worker
        await self._channel.set_qos(prefetch_count=self.rmq_cfg.prefetch_count)

        # declare exchange
        exchange = await self._channel.declare_exchange(
            self.rmq_cfg.exchange,
            type=aio_pika.ExchangeType(self.rmq_cfg.exchange_type),
            durable=True,
        )

        # declare queue (durable so messages survive restarts)
        queue = await self._channel.declare_queue(
            self.rmq_cfg.queue,
            durable=True,
            arguments={"x-queue-type": "classic"},
        )

        # bind queue to exchange with routing key
        await queue.bind(exchange, routing_key=self.rmq_cfg.routing_key)

        logger.info(
            "Listening on exchange='%s' queue='%s' routing_key='%s'",
            self.rmq_cfg.exchange,
            self.rmq_cfg.queue,
            self.rmq_cfg.routing_key,
        )

        # start consuming
        await queue.consume(self._on_message)

    async def _on_message(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        await handle_message(message, self._pool, self._embedder)

    async def stop(self) -> None:
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()
        if self._pool:
            await self._pool.close()
        logger.info("Worker stopped.")
