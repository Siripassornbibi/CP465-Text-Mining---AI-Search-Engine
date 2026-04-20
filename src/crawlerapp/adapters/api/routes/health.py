"""
adapters/api/routes/health.py
"""
from __future__ import annotations
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    cfg = request.app.state.container.config
    return {
        "status": "ok",
        "embed_model": cfg.embedder.model,
        "llm": cfg.llm.model,
        "rabbitmq": {
            "exchange": cfg.rabbitmq.exchange,
            "queue":    cfg.rabbitmq.queue,
        },
    }
