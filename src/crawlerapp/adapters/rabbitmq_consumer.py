"""
adapters/rabbitmq_consumer.py — aio_pika consumer wired to EmbedChunkUseCase.
"""
from __future__ import annotations
import json
import logging
from typing import Optional

import aio_pika
import aio_pika.abc

from crawlerapp.application.embed_chunk import EmbedChunkUseCase
from crawlerapp.domain.events import EmbeddingRequestedEvent

logger = logging.getLogger(__name__)


class RabbitMQConsumer:
    def __init__(
        self,
        amqp_url: str,
        exchange: str,
        exchange_type: str,
        queue: str,
        routing_key: str,
        prefetch_count: int,
        use_case: EmbedChunkUseCase,
    ) -> None:
        self._amqp_url = amqp_url
        self._exchange = exchange
        self._exchange_type = exchange_type
        self._queue = queue
        self._routing_key = routing_key
        self._prefetch_count = prefetch_count
        self._use_case = use_case
        self._connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None

    async def start(self) -> None:
        logger.info("Connecting to RabbitMQ ...")
        self._connection = await aio_pika.connect_robust(self._amqp_url)
        self._channel = await self._connection.channel()
        await self._channel.set_qos(prefetch_count=self._prefetch_count)

        exchange = await self._channel.declare_exchange(
            self._exchange,
            type=aio_pika.ExchangeType(self._exchange_type),
            durable=True,
        )
        queue = await self._channel.declare_queue(
            self._queue,
            durable=True,
            arguments={"x-queue-type": "classic"},
        )
        await queue.bind(exchange, routing_key=self._routing_key)

        await queue.consume(self._handle)
        logger.info(
            "Listening — exchange='%s' queue='%s' routing_key='%s'",
            self._exchange, self._queue, self._routing_key,
        )

    async def stop(self) -> None:
        if self._channel:
            await self._channel.close()
        if self._connection:
            await self._connection.close()
        logger.info("RabbitMQ consumer stopped.")

    async def _handle(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process(requeue=False):
            try:
                body = json.loads(message.body.decode())
                event = EmbeddingRequestedEvent.from_dict(body)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Bad message, discarding: %s", e)
                return

            await self._use_case.execute(event)
