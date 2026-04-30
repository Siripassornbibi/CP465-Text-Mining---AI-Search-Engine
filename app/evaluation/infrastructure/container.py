"""
evalapp/infrastructure/container.py
Builds the eval use cases by borrowing the ApiContainer
from the embedding package (renamed from crawlerapp).
"""
from __future__ import annotations
import logging

from embedding.config import AppConfig
from embedding.infrastructure.container import ApiContainer

from evaluation.application.retrieval_eval import RetrievalEvaluator
from evaluation.application.rag_eval import RAGEvaluator

logger = logging.getLogger(__name__)


class EvalContainer:
    """
    Thin wrapper around the embedding ApiContainer.
    Adds eval-specific use cases without duplicating any infra.
    """

    def __init__(self, config: AppConfig, top_k: int = 5) -> None:
        self._api = ApiContainer(config)
        self._top_k = top_k
        self.config = config

        # wired after start()
        self.retrieval_evaluator: RetrievalEvaluator | None = None
        self.rag_evaluator: RAGEvaluator | None = None

    async def start(self) -> None:
        await self._api.start()

        self.retrieval_evaluator = RetrievalEvaluator(
            search_use_case=self._api.search_use_case,
            top_k=self._top_k,
        )

        self.rag_evaluator = RAGEvaluator(
            search_use_case=self._api.search_use_case,
            llm_chain=self._api.llm_chain,
            ollama_base_url=self.config.evaluation_llm.ollama_url,
            evaluation_llm_model=self.config.evaluation_llm.model,
            embed_model=self.config.embedder.model,
            executor=self._api._executor,
            top_k=self._top_k,
        )

        logger.info("EvalContainer ready.")

    async def stop(self) -> None:
        await self._api.stop()
        logger.info("EvalContainer shut down.")