"""
adapters/api/app.py — FastAPI app factory.
Receives a fully-built container so it has no infrastructure knowledge.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import AsyncIterator, TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from embedding.adapters.api.routes import health, search

if TYPE_CHECKING:
    from embedding.infrastructure.container import Container


def create_app(container: "Container") -> FastAPI:

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

    app.include_router(health.router)
    app.include_router(search.router)

    return app
