"""
config.py — loads and validates config.json
database_url is read from the DB_URL environment variable.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


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
    model: str
    batch_size: int
    device: Optional[str]


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
        return cls(
            database_url=database_url,
            rabbitmq=RabbitMQConfig(**raw["rabbitmq"]),
            embedder=EmbedderConfig(**raw["embedder"]),
            llm=LLMConfig(**raw["llm"]),
            api=APIConfig(**raw["api"]),
        )
