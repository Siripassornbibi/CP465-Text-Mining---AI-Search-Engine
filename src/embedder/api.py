"""
api.py — FastAPI search endpoint + RabbitMQ embedding worker (single process)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pgvector.asyncpg import register_vector
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from config import AppConfig
from worker import EmbeddingWorker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
cfg = AppConfig.load(CONFIG_PATH)

QUERY_PREFIX = "Represent this query for searching: "

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

class AppState:
    pool: asyncpg.Pool
    embed_model: SentenceTransformer
    llm: ChatOllama
    chain: any
    worker: EmbeddingWorker

state = AppState()

# ---------------------------------------------------------------------------
# Lifespan — start everything once at boot
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("Loading embedding model %s ...", cfg.embedder.model)
    state.embed_model = SentenceTransformer(cfg.embedder.model, device=cfg.embedder.device)

    logger.info("Connecting to Postgres ...")
    state.pool = await asyncpg.create_pool(
        cfg.database_url,
        min_size=2,
        max_size=10,
        init=register_vector,
    )

    logger.info("Initialising LLM (%s) ...", cfg.llm.model)
    state.llm = ChatOllama(
        model=cfg.llm.model,
        base_url=cfg.llm.ollama_url,
        temperature=0.2,
    )
    state.chain = build_chain(state.llm)

    # start RabbitMQ worker as a background task in the same event loop
    state.worker = EmbeddingWorker(cfg)
    await state.worker.start()
    logger.info("RabbitMQ worker started — listening on queue '%s'", cfg.rabbitmq.queue)

    logger.info("Ready.")
    yield

    # graceful shutdown
    await state.worker.stop()
    await state.pool.close()


app = FastAPI(title="Crawler Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# LangChain RAG chain
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful search assistant. Using ONLY the context \
passages below, write a concise summary paragraph that directly answers the \
user's query. Do not make up information not present in the context.

After the summary, list the sources you used on separate lines as:
Sources:
- <title> — <url>

Context:
{context}"""


def build_chain(llm: ChatOllama):
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{query}"),
    ])
    return RunnablePassthrough() | prompt | llm | StrOutputParser()


def format_context(results: list[dict]) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Section: {r['section_heading'] or 'Introduction'}\n"
            f"{r['content']}"
        )
    return "\n\n---\n\n".join(parts)


def dedupe_sources(results: list[dict]) -> list[dict]:
    seen = set()
    sources = []
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            sources.append({"title": r["title"], "url": r["url"], "score": float(r["score"])})
    return sources

# ---------------------------------------------------------------------------
# Vector search
# ---------------------------------------------------------------------------

async def vector_search(query: str, top_k: int) -> list[dict]:
    loop = asyncio.get_event_loop()
    prefixed = f"{QUERY_PREFIX}{query}"
    vector: np.ndarray = await loop.run_in_executor(
        None,
        lambda: state.embed_model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32),
    )

    async with state.pool.acquire() as conn:
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
    return [dict(r) for r in rows]

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=8, ge=1, le=20)


class SourceLink(BaseModel):
    title: str
    url: str
    score: float


class SearchResponse(BaseModel):
    query: str
    summary: str
    sources: list[SourceLink]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    results = await vector_search(req.query, req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant results found.")

    context = format_context(results)
    summary = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: state.chain.invoke({"context": context, "query": req.query}),
    )
    sources = dedupe_sources(results)
    return SearchResponse(
        query=req.query,
        summary=summary,
        sources=[SourceLink(**s) for s in sources],
    )


@app.post("/search/stream")
async def search_stream(req: SearchRequest):
    results = await vector_search(req.query, req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant results found.")

    context = format_context(results)
    sources = dedupe_sources(results)

    async def generate():
        async for token in state.chain.astream({"context": context, "query": req.query}):
            yield f"data: {token}\n\n"
        yield f"event: sources\ndata: {_json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embed_model": cfg.embedder.model,
        "llm": cfg.llm.model,
        "rabbitmq": {
            "exchange": cfg.rabbitmq.exchange,
            "queue": cfg.rabbitmq.queue,
            "routing_key": cfg.rabbitmq.routing_key,
        },
    }
