"""
infrastructure/container.py — Dependency container.
Owns all singletons. Wires ports → adapters → use cases.
This is the only place that knows about concrete implementations.
"""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from pgvector.asyncpg import register_vector

from crawlerapp.adapters.bge_embedder import BGEEmbedder
from crawlerapp.adapters.postgres_chunk_repo import PostgresChunkRepository
from crawlerapp.adapters.postgres_search_repo import PostgresSearchRepository
from crawlerapp.adapters.rabbitmq_consumer import RabbitMQConsumer
from crawlerapp.application.embed_chunk import EmbedChunkUseCase
from crawlerapp.application.search import SearchUseCase
from crawlerapp.config import AppConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful search assistant. Using ONLY the context \
passages below, write a concise summary paragraph that directly answers the \
user's query. Do not make up information.

After the summary, list sources as:
Sources:
- <title> — <url>

Context:
{context}"""


class Container:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        # shared thread pool for CPU-bound work (embedding)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")

        # ports → adapters (built eagerly; connections opened in start())
        self._pool: asyncpg.Pool | None = None
        self.embedder = BGEEmbedder(
            model_name=config.embedder.model,
            device=config.embedder.device,
            batch_size=config.embedder.batch_size,
        )

        # repositories (need pool — finalised in start())
        self._chunk_repo: PostgresChunkRepository | None = None
        self._search_repo: PostgresSearchRepository | None = None

        # use cases (finalised in start())
        self.embed_chunk_use_case: EmbedChunkUseCase | None = None
        self.search_use_case: SearchUseCase | None = None

        # LLM chain
        llm = ChatOllama(
            model=config.llm.model,
            base_url=config.llm.ollama_url,
            temperature=0.2,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{query}"),
        ])
        self.llm_chain = RunnablePassthrough() | prompt | llm | StrOutputParser()

        # RabbitMQ consumer (finalised in start())
        self._consumer: RabbitMQConsumer | None = None

    async def start(self) -> None:
        logger.info("Opening Postgres pool ...")
        self._pool = await asyncpg.create_pool(
            self.config.database_url,
            min_size=2,
            max_size=10,
            init=register_vector,
        )

        self._chunk_repo  = PostgresChunkRepository(self._pool)
        self._search_repo = PostgresSearchRepository(self._pool)

        self.embed_chunk_use_case = EmbedChunkUseCase(
            repo=self._chunk_repo,
            embedder=self.embedder,
            executor=self._executor,
        )
        self.search_use_case = SearchUseCase(
            embedder=self.embedder,
            search_repo=self._search_repo,
            executor=self._executor,
            top_k=self.config.api.top_k,
        )

        rmq = self.config.rabbitmq
        self._consumer = RabbitMQConsumer(
            amqp_url=rmq.url,
            exchange=rmq.exchange,
            exchange_type=rmq.exchange_type,
            queue=rmq.queue,
            routing_key=rmq.routing_key,
            prefetch_count=rmq.prefetch_count,
            use_case=self.embed_chunk_use_case,
        )
        await self._consumer.start()

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()
        if self._pool:
            await self._pool.close()
        self._executor.shutdown(wait=False)
        logger.info("Container shut down.")
