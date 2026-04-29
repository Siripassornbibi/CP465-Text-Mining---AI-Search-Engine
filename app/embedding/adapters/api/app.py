"""
adapters/api/app.py — FastAPI app factory.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, TYPE_CHECKING
import os
 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
 
from app.embedding.adapters.api.routes import search
from app.embedding.adapters.api.routes import health
 
if TYPE_CHECKING:
    from app.embedding.infrastructure.container import ApiContainer
 
STATIC_DIR = Path(os.getenv("STATIC_DIR", "search-ui/dist"))
 
 
def create_app(container: "ApiContainer") -> FastAPI:
 
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.container = container
        await container.start()
        yield
        await container.stop()
 
    app = FastAPI(title="Crawler Search API", lifespan=lifespan)
 
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
 
    app.include_router(health.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
 
    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
 
        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_frontend(full_path: str) -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")
    else:
        import logging
        logging.getLogger(__name__).warning(
            "Static dir '%s' not found — frontend not served.", STATIC_DIR
        )
 
    return app