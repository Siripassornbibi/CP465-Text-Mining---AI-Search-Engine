"""
infrastructure/container.py — Split containers for API and Worker modes.

ApiContainer    — FastAPI + search use case + LLM chain
WorkerContainer — RabbitMQ consumer + embed use case
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import asyncpg
from pgvector.asyncpg import register_vector

from app.embedding.adapters.bge_embedder import BGEEmbedder
from app.embedding.adapters.ollama_embedder import OllamaEmbedder
from app.embedding.adapters.postgres_chunk_repo import PostgresChunkRepository
from app.embedding.adapters.postgres_search_repo import PostgresSearchRepository
from app.embedding.adapters.rabbitmq_consumer import RabbitMQConsumer
from app.embedding.application.embed_chunk import EmbedChunkUseCase
from app.embedding.application.search import SearchUseCase
from app.embedding.config import AppConfig
from app.embedding.ports.embedder import IEmbedder

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful search assistant. Using ONLY the context \
passages below, write a concise summary paragraph that directly answers the \
user's query. Do not make up information. Do NOT include any preamble, \
introduction, or phrases like "Here is a summary" — start the answer directly.

After the summary, list sources as:
Sources:
- <title> — <url>

Context:
{context}"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_embedder(config: AppConfig) -> IEmbedder:
    cfg = config.embedder
    if cfg.backend == "local":
        logger.info("Embedder: local BGE — model=%s device=%s", cfg.model, cfg.device)
        return BGEEmbedder(
            model_name=cfg.model,
            device=cfg.device,
            batch_size=cfg.batch_size,
        )
    if cfg.backend == "ollama":
        logger.info("Embedder: remote Ollama — url=%s model=%s", cfg.ollama_url, cfg.model)
        return OllamaEmbedder(
            base_url=cfg.ollama_url,
            model=cfg.model,
            timeout=cfg.timeout,
        )
    raise ValueError(f"Unknown embedder backend: '{cfg.backend}'. Use 'local' or 'ollama'.")


async def _open_pool(database_url: str) -> asyncpg.Pool:
    logger.info("Opening Postgres pool ...")
    return await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=10,
        init=register_vector,
    )


# ---------------------------------------------------------------------------
# ApiContainer — runs FastAPI + search
# ---------------------------------------------------------------------------

class ApiContainer:
    """Boots only what the API needs: DB pool, embedder, LLM chain, search use case."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")
        self.embedder: IEmbedder = _build_embedder(config)

        # LLM chain — import here so worker mode doesn't need langchain
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnablePassthrough
        from langchain_ollama import ChatOllama

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

        self._pool: asyncpg.Pool | None = None
        self.search_use_case: SearchUseCase | None = None

    async def start(self) -> None:
        self._pool = await _open_pool(self.config.database_url)
        search_repo = PostgresSearchRepository(self._pool)
        self.search_use_case = SearchUseCase(
            embedder=self.embedder,
            search_repo=search_repo,
            executor=self._executor,
            top_k=self.config.api.top_k,
        )
        logger.info("ApiContainer ready.")

    async def stop(self) -> None:
        if self._pool:
            await self._pool.close()
        if isinstance(self.embedder, OllamaEmbedder):
            self.embedder.close()
        self._executor.shutdown(wait=False)
        logger.info("ApiContainer shut down.")


# ---------------------------------------------------------------------------
# WorkerContainer — runs RabbitMQ consumer + embedding
# ---------------------------------------------------------------------------

class WorkerContainer:
    """Boots only what the worker needs: DB pool, embedder, RabbitMQ consumer."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")
        self.embedder: IEmbedder = _build_embedder(config)

        self._pool: asyncpg.Pool | None = None
        self._consumer: RabbitMQConsumer | None = None

    async def start(self) -> None:
        self._pool = await _open_pool(self.config.database_url)
        chunk_repo = PostgresChunkRepository(self._pool)

        embed_use_case = EmbedChunkUseCase(
            repo=chunk_repo,
            embedder=self.embedder,
            executor=self._executor,
        )

        rmq = self.config.rabbitmq
        self._consumer = RabbitMQConsumer(
            amqp_url=rmq.url,
            exchange=rmq.exchange,
            exchange_type=rmq.exchange_type,
            queue=rmq.queue,
            routing_key=rmq.routing_key,
            prefetch_count=rmq.prefetch_count,
            use_case=embed_use_case,
        )
        await self._consumer.start()
        logger.info("WorkerContainer ready — listening on queue '%s'.", rmq.queue)

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()
        if self._pool:
            # timeout=5 prevents hanging if a query is mid-flight
            try:
                await asyncio.wait_for(self._pool.close(), timeout=5)
                logger.info("Postgres pool closed.")
            except asyncio.TimeoutError:
                logger.warning("Postgres pool close timed out — forcing.")
                self._pool.terminate()
        if isinstance(self.embedder, OllamaEmbedder):
            self.embedder.close()
        self._executor.shutdown(wait=False)
        logger.info("WorkerContainer shut down.")