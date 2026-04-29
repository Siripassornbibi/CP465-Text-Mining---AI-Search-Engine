"""
evalapp/domain.py — pure data models for evaluation, no dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TestCase:
    question: str
    expected_url: str
    ground_truth: str


@dataclass
class RetrievalResult:
    question: str
    expected_url: str
    retrieved_urls: list[str]
    retrieved_contents: list[str]
    scores: list[float]

    def hit_at(self, k: int) -> bool:
        """True if expected_url appears in the top-k results."""
        return any(
            self.expected_url in url
            for url in self.retrieved_urls[:k]
        )

    def reciprocal_rank(self) -> float:
        for i, url in enumerate(self.retrieved_urls):
            if self.expected_url in url:
                return 1.0 / (i + 1)
        return 0.0


@dataclass
class RAGResult:
    question: str
    answer: str
    ground_truth: str
    contexts: list[str]


@dataclass
class RetrievalReport:
    recall_at_1: float
    recall_at_k: float
    mrr: float
    top_k: int
    total: int
    per_question: list[RetrievalResult] = field(default_factory=list)


@dataclass
class RAGReport:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    total: int
    per_question: list[dict] = field(default_factory=list)