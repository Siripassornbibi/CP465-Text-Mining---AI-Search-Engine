"""
adapters/ollama_embedder.py — remote IEmbedder via Ollama's /api/embed endpoint.
Runs on the Mac; called over the network from Windows.
No local model is loaded — all computation happens on the remote machine.
"""
from __future__ import annotations
import json
import logging
import time

import httpx
import numpy as np

from app.embedding.ports.embedder import IEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(IEmbedder):
    """
    Calls POST {base_url}/api/embed to embed text remotely.
    Compatible with any model pulled on the Ollama server
    e.g. "bge-m3", "nomic-embed-text", "mxbai-embed-large".
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._dims: int | None = None

        # probe dims on init so misconfiguration fails fast
        probe, _ = self.embed("hello")
        self._dims = len(probe)
        logger.info(
            "OllamaEmbedder ready — url=%s model=%s dims=%d",
            self._base_url, self._model, self._dims,
        )

    # ------------------------------------------------------------------
    # IEmbedder implementation
    # ------------------------------------------------------------------

    def embed(self, text: str) -> tuple[np.ndarray, bool]:
        embeddings, ok = self._call([text])
        return embeddings[0], ok

    def embed_batch(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        # Ollama /api/embed accepts a list natively
        return self._call(texts)

    @property
    def dims(self) -> int:
        return self._dims

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        logger.info("OllamaEmbedder - Embedding — model=%s input=%s", self._model, texts)
        max_retries = 3
        backoff = 0.5 #seconds

        payload = {"model": self._model, "input": texts}

        for attempt in range(max_retries): 
            try:
                resp = self._client.post(
                    f"{self._base_url}/api/embed",
                    json=payload,
                )

                resp.raise_for_status()

                data = resp.json()

                # Ollama returns {"embeddings": [[...], [...]]}
                vectors = np.array(data["embeddings"], dtype=np.float32)

                # normalise for cosine similarity (matches BGEEmbedder behaviour)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return vectors / norms, True
            except Exception as e:
                print(f"[Retry {attempt+1}] Embedding failed:", str(e), " with payload: ", json.dumps(payload))
                if attempt < max_retries - 1:
                    sleep_time = backoff * (2 ** attempt)  # exponential backoff
                    time.sleep(sleep_time)
                else:
                    print("All retries failed. Returning fallback embeddings.")

        # 🔥 Fallback (always same shape)
        return np.zeros((len(texts), self._dims), dtype=np.float32), False

    def close(self) -> None:
        self._client.close()
