"""
domain/events.py — Domain events.
"""
from __future__ import annotations
from dataclasses import dataclass

EMBEDDING_REQUESTED = "prawler.embedding.id"

@dataclass(frozen=True)
class EmbeddingRequestedEvent:
    event_type: str
    page_uuid: str          # changed from chunk_uuid

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingRequestedEvent":
        event_type = data.get("event_type", "")
        page_uuid  = data.get("page_uuid", "")
        if not event_type:
            raise ValueError("Missing event_type")
        if not page_uuid:
            raise ValueError("Missing page_uuid")
        return cls(event_type=event_type, page_uuid=page_uuid)

    def is_embedding_request(self) -> bool:
        return self.event_type == EMBEDDING_REQUESTED