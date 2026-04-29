"""
evalapp/adapters/reporter.py — prints evaluation reports to stdout and writes to a file.
"""
from __future__ import annotations
import sys
from typing import TextIO
from evaluation.domain import RetrievalReport, RAGReport


def _write(line: str, *outputs: TextIO) -> None:
    """Write a line to all provided outputs."""
    for out in outputs:
        out.write(line + "\n")


def print_retrieval_report(report: RetrievalReport, filepath: str = "./retrival_report.txt") -> None:
    outputs = [sys.stdout]
    file_handle = None

    if filepath:
        file_handle = open(filepath, "w", encoding="utf-8")
        outputs.append(file_handle)

    try:
        _write("\n" + "═" * 52, *outputs)
        _write("  RETRIEVAL EVALUATION", *outputs)
        _write("═" * 52, *outputs)
        _write(f"  Total questions : {report.total}", *outputs)
        _write(f"  Top-K           : {report.top_k}", *outputs)
        _write(f"  Recall@1        : {report.recall_at_1:.2%}", *outputs)
        _write(f"  Recall@{report.top_k}        : {report.recall_at_k:.2%}", *outputs)
        _write(f"  MRR             : {report.mrr:.4f}", *outputs)
        _write("─" * 52, *outputs)

        for r in report.per_question:
            hit = "✓" if r.hit_at(report.top_k) else "✗"
            rr = r.reciprocal_rank()
            top = r.scores[0] if r.scores else 0.0
            _write(f"  {hit} [{rr:.2f}] (top={top:.2f}) {r.question[:55]}", *outputs)

            if not r.hit_at(report.top_k):
                _write(f"    Expected : {', '.join(r.expected_urls)}", *outputs)
                if r.retrieved_urls:
                    _write(f"    Got      : {', '.join(r.retrieved_urls)}", *outputs)

        _write("═" * 52 + "\n", *outputs)
    finally:
        if file_handle:
            file_handle.close()


def print_rag_report(report: RAGReport, filepath: str = "./rag_report.txt") -> None:
    outputs = [sys.stdout]
    file_handle = None

    if filepath:
        file_handle = open(filepath, "w", encoding="utf-8")
        outputs.append(file_handle)

    try:
        _write("\n" + "═" * 52, *outputs)
        _write("  RAG EVALUATION (RAGAS)", *outputs)
        _write("═" * 52, *outputs)
        _write(f"  Total questions    : {report.total}", *outputs)
        _write(f"  Faithfulness       : {report.faithfulness:.4f}", *outputs)
        _write(f"  Answer Relevancy   : {report.answer_relevancy:.4f}", *outputs)
        _write(f"  Context Precision  : {report.context_precision:.4f}", *outputs)
        _write(f"  Context Recall     : {report.context_recall:.4f}", *outputs)
        _write("─" * 52, *outputs)

        for r in report.per_question:
            _write(f"\n  Q: {r['question'][:60]}", *outputs)
            _write(f"     faithfulness={r['faithfulness']:.2f}  "
                   f"relevancy={r['answer_relevancy']:.2f}  "
                   f"precision={r['context_precision']:.2f}  "
                   f"recall={r['context_recall']:.2f}", *outputs)

        _write("═" * 52 + "\n", *outputs)
    finally:
        if file_handle:
            file_handle.close()