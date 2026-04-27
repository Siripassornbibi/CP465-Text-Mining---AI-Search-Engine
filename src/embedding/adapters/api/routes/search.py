"""
adapters/api/routes/search.py
"""
from __future__ import annotations
import json
import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from embedding.adapters.api.dependencies import SearchUseCaseDep

router = APIRouter()


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

PREAMBLES = (
    "here is a concise summary",
    "here is a summary",
    "here's a summary",
    "to answer your query",
)

def _strip_preamble(text: str) -> str:
    lower = text.lower()
    for p in PREAMBLES:
        if lower.startswith(p):
            # cut to the first colon or newline after the preamble
            cut = text.find(":", len(p))
            if cut != -1:
                return text[cut + 1:].lstrip()
    return text

def _format_context(results) -> str:
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] {r.title}\nURL: {r.url}\n"
            f"Section: {r.section_heading or 'Introduction'}\n{r.content}"
        )
    return "\n\n---\n\n".join(parts)


def _dedupe_sources(results) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            out.append({"title": r.title, "url": r.url, "score": r.score})
    return out


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request, use_case: SearchUseCaseDep) -> SearchResponse:
    response = await use_case.execute(req.query, req.top_k)
    if not response.results:
        raise HTTPException(status_code=404, detail="No relevant results found.")

    context = _format_context(response.results)
    chain   = request.app.state.container.llm_chain

    loop    = asyncio.get_event_loop()
    summary = await loop.run_in_executor(
        None,
        lambda: chain.invoke({"context": context, "query": req.query}),
    )

    return SearchResponse(
        query=req.query,
        summary=_strip_preamble(summary),
        sources=[SourceLink(**s) for s in _dedupe_sources(response.results)],
    )


@router.post("/search/stream")
async def search_stream(req: SearchRequest, request: Request, use_case: SearchUseCaseDep):
    response = await use_case.execute(req.query, req.top_k)
    if not response.results:
        raise HTTPException(status_code=404, detail="No relevant results found.")
 
    context = _format_context(response.results)
    sources = _dedupe_sources(response.results)
    chain   = request.app.state.container.llm_chain
 
    async def generate():
        buffer = ""
        preamble_stripped = False
 
        async for token in chain.astream({"context": context, "query": req.query}):
            if not preamble_stripped:
                # accumulate until we have enough text to detect a preamble
                buffer += token
                if len(buffer) < 120:
                    continue
                buffer = _strip_preamble(buffer)
                preamble_stripped = True
                yield f"data: {buffer}\n\n"
                buffer = ""
            else:
                yield f"data: {token}\n\n"
 
        # flush any remaining buffer (short responses)
        if buffer:
            yield f"data: {_strip_preamble(buffer)}\n\n"
 
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield "data: [DONE]\n\n"
 
    return StreamingResponse(generate(), media_type="text/event-stream")