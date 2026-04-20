"""
config.py — loads and validates config.json
database_url is read from the DB_URL environment variable.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class RabbitMQConfig:
    url: str
    exchange: str
    exchange_type: str
    queue: str
    routing_key: str
    prefetch_count: int = 4


@dataclass
class EmbedderConfig:
    """
    backend: "local"  — load BGE model on this machine (sentence-transformers)
             "ollama" — call a remote Ollama server over HTTP
    """
    backend: Literal["local", "ollama"]
    model: str
    # local-only
    batch_size: int = 64
    device: Optional[str] = None
    # ollama-only
    ollama_url: Optional[str] = None
    timeout: float = 60.0


@dataclass
class LLMConfig:
    model: str
    ollama_url: str


@dataclass
class APIConfig:
    host: str
    port: int
    top_k: int


@dataclass
class AppConfig:
    database_url: str
    rabbitmq: RabbitMQConfig
    embedder: EmbedderConfig
    llm: LLMConfig
    api: APIConfig

    @classmethod
    def load(cls, path: str = "config.json") -> "AppConfig":
        raw = json.loads(Path(path).read_text())

        database_url = os.environ.get("DB_URL")
        if not database_url:
            raise RuntimeError("Environment variable DB_URL is not set.")

        embedder_raw = raw["embedder"]

        # validate ollama backend has a url
        if embedder_raw.get("backend") == "ollama" and not embedder_raw.get("ollama_url"):
            raise RuntimeError("embedder.ollama_url is required when backend is 'ollama'.")

        return cls(
            database_url=database_url,
            rabbitmq=RabbitMQConfig(**raw["rabbitmq"]),
            embedder=EmbedderConfig(**embedder_raw),
            llm=LLMConfig(**raw["llm"]),
            api=APIConfig(**raw["api"]),
        )
