"""
evalapp/application/rag_eval.py
Runs RAGAS metrics (faithfulness, answer_relevancy,
context_precision, context_recall) end to end.
"""
from __future__ import annotations
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from evaluation.domain import TestCase, RAGResult, RAGReport

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Generates answers via the LLM chain and scores them with RAGAS.
    Accepts the embedding container's search_use_case and llm_chain
    so it runs against the exact same stack as production.
    """

    def __init__(
        self,
        search_use_case,
        llm_chain,
        ollama_base_url: str,
        llm_model: str,
        embed_model: str,
        executor: ThreadPoolExecutor,
        top_k: int = 5,
    ) -> None:
        self._search      = search_use_case
        self._chain       = llm_chain
        self._ollama_url  = ollama_base_url
        self._llm_model   = llm_model
        self._embed_model = embed_model
        self._executor    = executor
        self._top_k       = top_k

    async def _generate_answer(self, question: str, contexts: list[str]) -> str:
        context_text = "\n\n---\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(contexts)
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._chain.invoke({
                "context": context_text,
                "query": question,
            }),
        )

    async def run(self, cases: list[TestCase]) -> RAGReport:
        # import here so the retrieval-only eval path doesn't require ragas
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )
            from datasets import Dataset
            from langchain_ollama import ChatOllama, OllamaEmbeddings
        except ImportError:
            raise ImportError(
                "RAG evaluation requires extra dependencies. "
                "Run: pip install ragas datasets langchain-ollama"
            )

        rag_results: list[RAGResult] = []

        for case in cases:
            logger.info("Generating answer for: %s", case.question)

            response = await self._search.execute(case.question, self._top_k)
            contexts = [r.content for r in response.results]
            answer   = await self._generate_answer(case.question, contexts)

            rag_results.append(RAGResult(
                question=case.question,
                answer=answer,
                ground_truth=case.ground_truth,
                contexts=contexts,
            ))

        # build RAGAS dataset
        dataset = Dataset.from_dict({
            "question":    [r.question     for r in rag_results],
            "answer":      [r.answer       for r in rag_results],
            "contexts":    [r.contexts     for r in rag_results],
            "ground_truth":[r.ground_truth for r in rag_results],
        })

        logger.info("Running RAGAS scoring ...")

        llm        = ChatOllama(model=self._llm_model,   base_url=self._ollama_url)
        embeddings = OllamaEmbeddings(model=self._embed_model, base_url=self._ollama_url)

        ragas_result = ragas_evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )

        df = ragas_result.to_pandas()

        per_question = [
            {
                "question":          row["question"],
                "answer":            row["answer"],
                "faithfulness":      row.get("faithfulness", 0),
                "answer_relevancy":  row.get("answer_relevancy", 0),
                "context_precision": row.get("context_precision", 0),
                "context_recall":    row.get("context_recall", 0),
            }
            for _, row in df.iterrows()
        ]

        return RAGReport(
            faithfulness      =float(df["faithfulness"].mean()),
            answer_relevancy  =float(df["answer_relevancy"].mean()),
            context_precision =float(df["context_precision"].mean()),
            context_recall    =float(df["context_recall"].mean()),
            total             =len(cases),
            per_question      =per_question,
        )