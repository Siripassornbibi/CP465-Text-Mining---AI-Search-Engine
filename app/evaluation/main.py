"""
main.py — eval entry point.

Usage:
    # retrieval only (fast, no LLM)
    python -m evalapp

    # full RAG eval with RAGAS (slow, requires LLM)
    python -m evalapp --rag

    # custom test cases file
    python -m evalapp --cases path/to/test_cases.json

    # custom top-k
    python -m evalapp --top-k 10
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import os
import sys

from embedding.config import AppConfig

from evaluation.adapters.test_case_loader import load_test_cases
from evaluation.adapters.reporter import print_retrieval_report, print_rag_report
from evaluation.infrastructure.container import EvalContainer

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    p.add_argument(
        "--rag",
        action="store_true",
        help="Also run full RAGAS end-to-end evaluation (requires ragas + datasets).",
    )
    p.add_argument(
        "--cases",
        default=os.getenv("TEST_CASES_PATH", "test_cases.json"),
        help="Path to test_cases.json (default: test_cases.json).",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("EVAL_TOP_K", "5")),
        help="Number of chunks to retrieve per query (default: 5).",
    )
    return p.parse_args()


async def run(args: argparse.Namespace) -> None:
    if not os.environ.get("DB_URL"):
        logger.error("DB_URL environment variable is not set.")
        sys.exit(1)

    # load config from the embedding package
    config_path = os.getenv("CONFIG_PATH", "config.json")
    cfg = AppConfig.load(config_path)

    # load test cases
    logger.info("Loading test cases from %s ...", args.cases)
    cases = load_test_cases(args.cases)
    logger.info("Loaded %d test case(s).", len(cases))

    container = EvalContainer(cfg, top_k=args.top_k)
    try:
        await container.start()

        # ── retrieval evaluation ──
        retrieval_report = await container.retrieval_evaluator.run(cases)
        print_retrieval_report(retrieval_report)

        # ── RAG evaluation (optional) ──
        if args.rag:
            rag_report = await container.rag_evaluator.run(cases)
            print_rag_report(rag_report)

    finally:
        await container.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        logger.info("Cancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()