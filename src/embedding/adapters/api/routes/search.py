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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PREAMBLES = (
    "here is a concise summary paragraph that directly answers the user's query:",
    "here is a concise summary:",
    "here is a summary:",
    "here's a summary:",
    "here's a concise summary:",
    "to answer your query,",
    "to directly answer your query,",
)


def _fix_newlines(text: str) -> str:
    """Ollama sometimes streams literal '\\n' instead of real newline characters."""
    return text.replace("\\n", "\n")


def _strip_preamble(text: str) -> str:
    lower = text.lower()
    for p in _PREAMBLES:
        if lower.startswith(p):
            return text[len(p):].lstrip("\n ")
    return text


def _clean(text: str) -> str:
    return _strip_preamble(_fix_newlines(text))


def _sse_token(token: str) -> str:
    """
    Encode a token as a safe SSE data line.
    JSON-encodes the token so embedded newlines survive the SSE wire format.
    Frontend must JSON.parse() each data field.

    e.g. token = "hello\nworld" → data: "hello\\nworld"
    """
    return f"data: {json.dumps(token)}\n\n"


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

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
        summary=_clean(summary),
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
            token = _fix_newlines(token)

            if not preamble_stripped:
                buffer += token
                if len(buffer) < 120:
                    continue
                buffer = _strip_preamble(buffer)
                preamble_stripped = True
                yield _sse_token(buffer)
                buffer = ""
            else:
                yield _sse_token(token)

        # flush remaining buffer for short responses
        if buffer:
            yield _sse_token(_strip_preamble(buffer))

        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        yield f"data: {json.dumps('[DONE]')}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")