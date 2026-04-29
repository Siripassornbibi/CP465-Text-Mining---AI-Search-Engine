"""
evalapp/application/retrieval_eval.py
Runs Recall@K and MRR against the embedding index.
Wired to the embedding package's SearchUseCase via the container.
"""
from __future__ import annotations
import logging

import numpy as np

from evaluation.domain import TestCase, RetrievalResult, RetrievalReport

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Uses the embedding package's SearchUseCase directly so evaluation
    runs against the exact same retrieval path as production.
    """

    def __init__(self, search_use_case, top_k: int = 5) -> None:
        self._search = search_use_case
        self._top_k = top_k

    async def run(self, cases: list[TestCase]) -> RetrievalReport:
        results: list[RetrievalResult] = []

        for case in cases:
            logger.info("Evaluating retrieval: %s", case.question)
            response = await self._search.execute(case.question, self._top_k)

            result = RetrievalResult(
                question=case.question,
                expected_urls=case.expected_urls,
                retrieved_urls=[r.url for r in response.results],
                retrieved_contents=[r.content for r in response.results],
                scores=[r.score for r in response.results],
            )
            results.append(result)

            hit = result.hit_at(self._top_k)
            rr  = result.reciprocal_rank()
            logger.info(
                "  hit@%d=%s  RR=%.2f  top_score=%.2f",
                self._top_k, hit,
                rr,
                result.scores[0] if result.scores else 0,
            )

        n = len(cases)
        recall_at_1 = sum(r.hit_at(1) for r in results) / n
        recall_at_k = sum(r.hit_at(self._top_k) for r in results) / n
        mrr         = float(np.mean([r.reciprocal_rank() for r in results]))

        return RetrievalReport(
            recall_at_1=recall_at_1,
            recall_at_k=recall_at_k,
            mrr=mrr,
            top_k=self._top_k,
            total=n,
            per_question=results,
        )