"""
application/expand_query.py — generates related queries from the user query
using the LLM to improve retrieval recall (query expansion).
"""
from __future__ import annotations
import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """\
Given the search query below, generate {count} alternative phrasings that capture \
the same intent from different angles. These will be used to retrieve documents \
so make them diverse and specific.

Rules:
- Output ONLY the queries, one per line, no numbering, no explanation
- Do not repeat the original query
- Keep each query under 20 words

Query: {query}

Alternative queries:"""


class ExpandQueryUseCase:
    def __init__(
        self,
        llm_chain,
        executor: ThreadPoolExecutor,
        expand_count: int = 2,
    ) -> None:
        self._chain        = llm_chain
        self._executor     = executor
        self._expand_count = max(0, expand_count)

    async def execute(self, query: str) -> list[str]:
        """
        Returns [original_query, expanded_1, ..., expanded_n].
        Always includes the original as the first query.
        Falls back to [original_query] if expansion is disabled or LLM fails.
        """
        if self._expand_count == 0:
            logger.debug("Query expansion disabled (expand_queries=0).")
            return [query]

        loop = asyncio.get_event_loop()

        try:
            prompt = EXPANSION_PROMPT.format(
                count=self._expand_count,
                query=query,
            )
            raw: str = await loop.run_in_executor(
                self._executor,
                lambda: self._chain.invoke({
                    "context": "",
                    "query": prompt,
                }),
            )

            expanded = _parse_queries(raw, limit=self._expand_count)
            all_queries = [query] + expanded
            logger.info(
                "Query expansion: '%s' -> %d queries: %s",
                query, len(all_queries), all_queries,
            )
            return all_queries

        except Exception as e:
            logger.warning("Query expansion failed, using original only: %s", e)
            return [query]


def _parse_queries(raw: str, limit: int = 2) -> list[str]:
    """Extract non-empty lines, strip leading bullets/numbers/quotes."""
    lines = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^[\s\-\*\d\.\"\']+", "", line).strip()
        line = re.sub(r"[\"\' ]+$", "", line).strip()
        if line and len(line) > 5:
            lines.append(line)
    return lines[:limit]