"""
domain/chunk.py — Chunk entity
Pure dataclass, zero external dependencies.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

# BGE-M3 instruction prefixes (asymmetric retrieval)
PASSAGE_PREFIX = "Represent this sentence: "

@dataclass
class Chunk:
    id: str
    section_heading: str
    content: str
    page_title: str = ""         # from <title> or og:title
    page_description: str = ""   # from <meta name="description">
    embedding: Optional[np.ndarray] = None

    def is_embedded(self) -> bool:
        return self.embedding is not None

    def prepared_text(self) -> str:
        """
        Build the passage text for embedding.

        Format:
            <prefix><title> — <heading> — <description> | <content>

        - title gives the model page-level topic context
        - heading gives section-level context
        - description is the human-written summary (high signal)
        - content is the actual chunk body

        Parts are only included when non-empty so missing metadata
        doesn't add noise ("  —  — | text" would hurt, not help).
        """
        parts: list[str] = []

        if self.page_title:
            parts.append(self.page_title)
        if self.section_heading:
            parts.append(self.section_heading)

        header = " — ".join(parts)

        if self.page_description:
            body = f"{self.page_description} | {self.content}"
        else:
            body = self.content

        text = f"{header} — {body}" if header else body
        return f"{PASSAGE_PREFIX}{text}"