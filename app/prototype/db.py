"""
db.py — PostgreSQL + pgvector setup
Vector dimension อ่านจาก embedder.py เพื่อรองรับทุก model อัตโนมัติ
"""
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

DSN = (
    f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}"
    f":{os.getenv('POSTGRES_PASSWORD', 'postgres')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}"
    f":{os.getenv('POSTGRES_PORT', '5432')}"
    f"/{os.getenv('POSTGRES_DB', 'ai_search')}"
)


async def get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(DSN, min_size=2, max_size=10)


async def setup_db(pool: asyncpg.Pool):
    """
    สร้าง table โดยใช้ dimension จาก embedder
    ถ้า model เปลี่ยน → ต้อง DROP TABLE แล้วรันใหม่
    """
    from embedding import get_embedding_dim
    dim = get_embedding_dim()
    print(f"  📐 Embedding dimension: {dim}")

    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS search_cache (
                id          BIGSERIAL PRIMARY KEY,
                query       TEXT NOT NULL,
                query_emb   vector({dim}),
                results     JSONB NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute(f"""
            CREATE TABLE IF NOT EXISTS web_pages (
                id          BIGSERIAL PRIMARY KEY,
                url         TEXT UNIQUE NOT NULL,
                title       TEXT,
                content     TEXT,
                summary     TEXT,
                content_emb vector({dim}),
                scraped_at  TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS search_cache_emb_idx
                ON search_cache USING hnsw (query_emb vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS web_pages_emb_idx
                ON web_pages USING hnsw (content_emb vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
        """)
    print("✅ Database setup complete")


async def cache_lookup(pool: asyncpg.Pool, query_emb: list[float], threshold: float = 0.75):
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT results, 1 - (query_emb <=> $1::vector) AS similarity
            FROM search_cache
            WHERE 1 - (query_emb <=> $1::vector) > $2
            ORDER BY query_emb <=> $1::vector
            LIMIT 1
            """,
            str(query_emb),
            threshold,
        )
    return row


async def cache_store(pool: asyncpg.Pool, query: str, query_emb: list[float], results: list[dict]):
    import json
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO search_cache (query, query_emb, results)
            VALUES ($1, $2::vector, $3::jsonb)
            ON CONFLICT DO NOTHING
            """,
            query, str(query_emb), json.dumps(results),
        )


async def store_webpage(
    pool: asyncpg.Pool,
    url: str, title: str, content: str, summary: str, content_emb: list[float],
):
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO web_pages (url, title, content, summary, content_emb)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (url) DO UPDATE
                SET title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    summary = EXCLUDED.summary,
                    content_emb = EXCLUDED.content_emb,
                    scraped_at = NOW()
            """,
            url, title, content, summary, str(content_emb),
        )