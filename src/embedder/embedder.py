"""
embedder.py — local embedding pipeline using sentence-transformers + pgvector
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import asyncpg
import numpy as np
from pgvector.asyncpg import register_vector
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Good local models to choose from:
#
#  Model                              Dims   Size    Notes
#  ---------------------------------  -----  ------  -------------------------
#  BAAI/bge-small-en-v1.5            384    130MB   fastest, good quality
#  BAAI/bge-base-en-v1.5             768    440MB   best quality/speed balance ✓
#  BAAI/bge-large-en-v1.5            1024   1.3GB   highest quality, slow
#  sentence-transformers/all-MiniLM-L6-v2  384  80MB  lightweight baseline
#  nomic-ai/nomic-embed-text-v1.5    768    550MB   long context (8192 tokens)
#
# BGE models expect a query/passage prefix for retrieval tasks:
#   - passages (chunks): prepend "Represent this sentence: "  (or nothing)
#   - queries at search time: prepend "Represent this query for searching: "

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_BATCH_SIZE = 64
PASSAGE_PREFIX = ""          # BGE base doesn't need a prefix for passages
QUERY_PREFIX = "Represent this query for searching: "


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str           # uuid from DB
    section_heading: str
    content: str


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: Optional[str] = None,   # "cpu", "cuda", "mps" — auto-detected if None
        normalize: bool = True,         # normalize for cosine similarity
    ):
        logger.info("Loading model %s ...", model_name)
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.dims = self.model.get_sentence_embedding_dimension()
        logger.info("Model loaded — dims=%d device=%s", self.dims, self.model.device)

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_chunk_text(heading: str, content: str, prefix: str = PASSAGE_PREFIX) -> str:
        """
        Prefix the section heading so the model embeds with full context.
        "Go generics — It was introduced in version 1.18" embeds much better
        than the bare content alone.
        """
        body = f"{heading} — {content}" if heading else content
        return f"{prefix}{body}" if prefix else body

    @staticmethod
    def prepare_query(query: str) -> str:
        """Wrap a search-time query with the BGE query prefix."""
        return f"{QUERY_PREFIX}{query}"

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts. Returns float32 array of shape (N, dims).
        Automatically batches internally via sentence-transformers.
        """
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > self.batch_size,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> list[tuple[str, np.ndarray]]:
        """
        Embed a list of Chunk objects.
        Returns [(chunk_id, vector), ...] in the same order.
        """
        texts = [
            self.prepare_chunk_text(c.section_heading, c.content)
            for c in chunks
        ]
        vectors = self.embed_texts(texts)
        return [(chunks[i].id, vectors[i]) for i in range(len(chunks))]

    # ------------------------------------------------------------------
    # Postgres storage
    # ------------------------------------------------------------------

    async def store(
        self,
        conn: asyncpg.Connection,
        chunk_vectors: list[tuple[str, np.ndarray]],
    ) -> None:
        """
        Bulk-update the embedding column for a list of (chunk_id, vector) pairs.
        Uses a single executemany call — much faster than individual UPDATEs.
        """
        await register_vector(conn)
        await conn.executemany(
            "UPDATE chunks SET embedding = $1 WHERE id = $2",
            [(vec, chunk_id) for chunk_id, vec in chunk_vectors],
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def embed_and_store(
        self,
        conn: asyncpg.Connection,
        chunks: list[Chunk],
    ) -> None:
        """
        Embed all chunks and write vectors to Postgres.
        Processes in batches to bound memory usage.
        """
        if not chunks:
            return

        total = len(chunks)
        stored = 0

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]

            # CPU-bound — run in thread pool so it doesn't block the event loop
            loop = asyncio.get_event_loop()
            chunk_vectors = await loop.run_in_executor(
                None, self.embed_chunks, batch
            )

            await self.store(conn, chunk_vectors)
            stored += len(batch)
            logger.info("Embedded %d / %d chunks", stored, total)


# ---------------------------------------------------------------------------
# Postgres helpers
# ---------------------------------------------------------------------------

async def get_conn(dsn: str) -> asyncpg.Connection:
    conn = await asyncpg.connect(dsn)
    await register_vector(conn)
    return conn


async def get_unembedded_chunks(conn: asyncpg.Connection, page_id: str) -> list[Chunk]:
    """Fetch chunks that haven't been embedded yet for a given page."""
    rows = await conn.fetch(
        """
        SELECT id, section_heading, content
        FROM chunks
        WHERE page_id = $1
          AND embedding IS NULL
        ORDER BY chunk_index
        """,
        page_id,
    )
    return [Chunk(id=str(r["id"]), section_heading=r["section_heading"] or "", content=r["content"]) for r in rows]


# ---------------------------------------------------------------------------
# Search helper (query time)
# ---------------------------------------------------------------------------

async def search(
    conn: asyncpg.Connection,
    embedder: Embedder,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Embed a search query and return the top_k most similar chunks.
    Uses cosine similarity via pgvector <=> operator.
    """
    await register_vector(conn)

    query_vec = embedder.embed_texts([embedder.prepare_query(query)])[0]

    rows = await conn.fetch(
        """
        SELECT
            c.id,
            c.section_heading,
            c.content,
            p.url,
            pm.title,
            1 - (c.embedding <=> $1) AS score
        FROM chunks c
        JOIN pages p ON p.id = c.page_id
        JOIN page_metadata pm ON pm.page_id = p.id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> $1
        LIMIT $2
        """,
        query_vec,
        top_k,
    )

    return [dict(r) for r in rows]
