"""
evalapp/adapters/reporter.py — prints evaluation reports to stdout.
"""
from __future__ import annotations
from evaluation.domain import RetrievalReport, RAGReport


def print_retrieval_report(report: RetrievalReport) -> None:
    print("\n" + "═" * 52)
    print("  RETRIEVAL EVALUATION")
    print("═" * 52)
    print(f"  Total questions : {report.total}")
    print(f"  Top-K           : {report.top_k}")
    print(f"  Recall@1        : {report.recall_at_1:.2%}")
    print(f"  Recall@{report.top_k}        : {report.recall_at_k:.2%}")
    print(f"  MRR             : {report.mrr:.4f}")
    print("─" * 52)

    for r in report.per_question:
        hit  = "✓" if r.hit_at(report.top_k) else "✗"
        rr   = r.reciprocal_rank()
        top  = r.scores[0] if r.scores else 0.0
        print(f"  {hit} [{rr:.2f}] (top={top:.2f}) {r.question[:55]}")

        if not r.hit_at(report.top_k):
            print(f"    Expected : {r.expected_url}")
            if r.retrieved_urls:
                print(f"    Got      : {r.retrieved_urls[0]}")

    print("═" * 52 + "\n")


def print_rag_report(report: RAGReport) -> None:
    print("\n" + "═" * 52)
    print("  RAG EVALUATION (RAGAS)")
    print("═" * 52)
    print(f"  Total questions    : {report.total}")
    print(f"  Faithfulness       : {report.faithfulness:.4f}")
    print(f"  Answer Relevancy   : {report.answer_relevancy:.4f}")
    print(f"  Context Precision  : {report.context_precision:.4f}")
    print(f"  Context Recall     : {report.context_recall:.4f}")
    print("─" * 52)

    for r in report.per_question:
        print(f"\n  Q: {r['question'][:60]}")
        print(f"     faithfulness={r['faithfulness']:.2f}  "
              f"relevancy={r['answer_relevancy']:.2f}  "
              f"precision={r['context_precision']:.2f}  "
              f"recall={r['context_recall']:.2f}")

    print("═" * 52 + "\n")