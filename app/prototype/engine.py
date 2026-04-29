"""
engine.py — orchestrator หลัก
เชื่อม embedding → cache lookup → web search → scrape → summarize → store
"""
import asyncio
import json
import os
from dataclasses import dataclass, asdict

import asyncpg
from dotenv import load_dotenv

from app.prototype.db import get_pool, setup_db, cache_lookup, cache_store, store_webpage
from embedding import embed_text, embed_texts
from app.prototype.searcher import search_web, scrape_all, WebResult
from app.prototype.summarizer import summarize_all

load_dotenv()

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", "10"))


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    summary: str
    from_cache: bool = False


class AISearchEngine:
    def __init__(self):
        self.pool: asyncpg.Pool | None = None

    async def init(self):
        """เริ่ม connection pool และ setup database"""
        self.pool = await get_pool()
        await setup_db(self.pool)
        print("🚀 AISearchEngine ready")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def search(self, query: str) -> list[SearchResult]:
        """
        ขั้นตอนหลัก:
        1. embed query
        2. ค้นหาใน cache (vector similarity)
        3. ถ้า cache miss → ค้นหาเว็บ, scrape, summarize, store
        4. คืน results
        """
        print(f"\n🔍 Query: {query}")

        # Step 1: embed query
        print("  ⚡ Embedding query...")
        query_emb = await embed_text(query)

        # Step 2: ค้นหา cache
        print("  🗄️  Checking cache...")
        cached = await cache_lookup(self.pool, query_emb, SIMILARITY_THRESHOLD)
        if cached:
            print(f"  ✅ Cache hit! (similarity={cached['similarity']:.3f})")
            results_data = json.loads(cached["results"]) if isinstance(cached["results"], str) else cached["results"]
            return [
                SearchResult(from_cache=True, **r)
                for r in results_data
            ]

        # Step 3: cache miss → ค้นหาเว็บ
        print("  🌐 Cache miss — searching web...")
        raw_results = search_web(query, MAX_RESULTS)
        if not raw_results:
            return []

        # Step 4: scrape เนื้อหา
        print(f"  📥 Scraping {len(raw_results)} pages...")
        pages = await scrape_all(raw_results)

        # Step 5: summarize ด้วย LLM
        print("  🤖 Summarizing with LLM...")
        pages = await summarize_all(query, pages, query_emb)

        # Step 6: embed เนื้อหาและ store
        print("  💾 Storing to database...")
        contents = [p.content or p.snippet for p in pages]
        embeddings = await embed_texts(contents)

        store_tasks = [
            store_webpage(self.pool, p.url, p.title, p.content, p.summary, emb)
            for p, emb in zip(pages, embeddings)
        ]
        await asyncio.gather(*store_tasks)

        # Step 7: store query cache
        results_for_cache = [
            {
                "url": p.url,
                "title": p.title,
                "snippet": p.snippet,
                "summary": p.summary,
            }
            for p in pages
        ]
        await cache_store(self.pool, query, query_emb, results_for_cache)

        return [
            SearchResult(
                url=p.url,
                title=p.title,
                snippet=p.snippet,
                summary=p.summary,
                from_cache=False,
            )
            for p in pages
        ]


# Singleton
_engine: AISearchEngine | None = None


async def get_engine() -> AISearchEngine:
    global _engine
    if _engine is None:
        _engine = AISearchEngine()
        await _engine.init()
    return _engine