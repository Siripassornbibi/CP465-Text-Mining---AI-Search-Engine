"""
api.py — FastAPI server
รัน: uvicorn api:app --reload
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.prototype.engine import get_engine, AISearchEngine, SearchResult


# ── Lifespan (startup / shutdown) ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = await get_engine()
    app.state.engine = engine
    yield
    await engine.close()


app = FastAPI(
    title="AI Search Engine",
    description="ค้นหาเว็บ + สรุปด้วย LLM + cache ด้วย pgvector",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ─────────────────────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    max_results: int = 10


class SearchResultItem(BaseModel):
    rank: int
    url: str
    title: str
    snippet: str
    summary: str
    from_cache: bool


class SearchResponse(BaseModel):
    query: str
    total: int
    elapsed_ms: float
    from_cache: bool
    results: list[SearchResultItem]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/search", response_model=SearchResponse)
async def search(q: str = Query(..., description="คำค้นหา", min_length=1)):
    """
    ค้นหาเว็บ + สรุปด้วย AI
    
    - **q**: คำค้นหา
    - คืนผลลัพธ์ 10 เว็บ พร้อมสรุป
    """
    engine: AISearchEngine = app.state.engine
    t0 = time.perf_counter()

    try:
        results = await engine.search(q)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.perf_counter() - t0) * 1000
    any_cache = any(r.from_cache for r in results)

    return SearchResponse(
        query=q,
        total=len(results),
        elapsed_ms=round(elapsed, 1),
        from_cache=any_cache,
        results=[
            SearchResultItem(rank=i + 1, **r.__dict__)
            for i, r in enumerate(results)
        ],
    )


@app.post("/search", response_model=SearchResponse)
async def search_post(body: SearchRequest):
    """POST version สำหรับ query ยาว"""
    return await search(q=body.query)


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── CLI runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
