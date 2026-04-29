"""
evalapp/adapters/test_case_loader.py
Reads test_cases.json — question-to-value format.
"""
from __future__ import annotations
import json
from pathlib import Path

from evaluation.domain import TestCase


def load_test_cases(path: str = "test_cases.json") -> list[TestCase]:
    """
    Reads a JSON file in question-to-value format:

    {
      "How does Go handle concurrency?": {
        "expected_urls": [
            "https://..."
        ],
        "ground_truth": "Go uses goroutines..."
      },
      ...
    }
    """
    raw: dict = json.loads(Path(path).read_text(encoding="utf-8"))

    cases = []
    for question, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected a dict for question '{question}', got {type(value).__name__}. "
                f"Format must be: {{ 'expected_urls': ['...]', 'ground_truth': '...' }}"
            )
        missing = [k for k in ("expected_urls", "ground_truth") if k not in value]
        if missing:
            raise ValueError(
                f"Question '{question}' is missing keys: {missing}"
            )
		
        expected = value["expected_urls"]

        if isinstance(expected, str):
            expected = [expected]
        elif not isinstance(expected, list):
            raise ValueError(
				f"'expected_urls' for question '{question}' must be string or list"
			)

        cases.append(TestCase(
            question=question,
            expected_urls=expected,
            ground_truth=value["ground_truth"],
        ))

    return cases