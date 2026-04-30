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
	exchange: str
	exchange_type: str
	queue: str
	routing_key: str
	prefetch_count: int = 4
	url: str = "amqp://guest:guest@localhost:5672/"


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
	expand_queries: int = 2   # number of additional queries to generate (0 = disabled)


@dataclass
class AppConfig:
	database_url: str
	rabbitmq: RabbitMQConfig
	embedder: EmbedderConfig
	llm: LLMConfig
	evaluation_llm: LLMConfig
	api: APIConfig

	@classmethod
	def load(cls, path: str = "config.json") -> "AppConfig":
		raw = json.loads(Path(path).read_text())

		database_url = os.environ.get("DB_URL")
		if not database_url:
			raise RuntimeError("Environment variable DB_URL is not set.")

		amqp_url = os.environ.get("AMQP_URL")
		if not amqp_url:
			amqp_url = "amqp://guest:guest@localhost:5672/"

		embedder_raw = raw["embedder"]

		# validate ollama backend has a url
		if embedder_raw.get("backend") == "ollama" and not embedder_raw.get("ollama_url"):
			raise RuntimeError("embedder.ollama_url is required when backend is 'ollama'.")
		
		rabbitmq = RabbitMQConfig(**raw["rabbitmq"])
		rabbitmq.url = amqp_url

		return cls(
			database_url=database_url,
			rabbitmq=rabbitmq,
			embedder=EmbedderConfig(**embedder_raw),
			llm=LLMConfig(**raw["llm"]),
			evaluation_llm=LLMConfig(**raw["evaluation_llm"]),
			api=APIConfig(**raw["api"]),
		)