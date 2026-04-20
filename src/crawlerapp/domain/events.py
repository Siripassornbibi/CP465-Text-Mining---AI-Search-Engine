"""
domain/events.py — Domain events.
"""
from __future__ import annotations
from dataclasses import dataclass

EMBEDDING_REQUESTED = "prawler.embedding.id"


@dataclass(frozen=True)
class EmbeddingRequestedEvent:
    event_type: str
    chunk_uuid: str

    @classmethod
    def from_dict(cls, data: dict) -> "EmbeddingRequestedEvent":
        event_type = data.get("event_type", "")
        chunk_uuid = data.get("chunk_uuid", "")
        if not event_type:
            raise ValueError("Missing event_type")
        if not chunk_uuid:
            raise ValueError("Missing chunk_uuid")
        return cls(event_type=event_type, chunk_uuid=chunk_uuid)

    def is_embedding_request(self) -> bool:
        return self.event_type == EMBEDDING_REQUESTED
