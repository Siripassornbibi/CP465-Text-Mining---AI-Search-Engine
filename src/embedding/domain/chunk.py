"""
domain/chunk.py — Chunk entity
Pure dataclass, zero external dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    id: str
    section_heading: str
    content: str
    embedding: Optional[np.ndarray] = None

    def is_embedded(self) -> bool:
        return self.embedding is not None

    def prepared_text(self, prefix: str = "") -> str:
        """Return the text that should be embedded — heading + content."""
        body = f"{self.section_heading} — {self.content}" if self.section_heading else self.content
        return f"{prefix}{body}" if prefix else body
