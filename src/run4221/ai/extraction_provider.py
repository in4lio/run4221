from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from run4221.ingestion.page_snapshot import PageSnapshot


@dataclass(frozen=True)
class EventExtraction:
    name: str
    public_id: str
    city: str
    country: str
    timezone: str
    event_date: str | None
    distances: tuple[str, ...]
    regions: tuple[str, ...]
    official_url: str
    registration_url: str | None
    confidence: float
    evidence_snippets: tuple[str, ...]
    provider_name: str = "unknown"


class ExtractorProvider(Protocol):
    provider_name: str

    async def extract(self, snapshot: PageSnapshot) -> EventExtraction:
        """Return structured event fields extracted from a stored page snapshot."""


class ExtractorProviderError(RuntimeError):
    pass
