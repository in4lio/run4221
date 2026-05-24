from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel, Field, SecretStr

from run4221.ai.extraction_provider import (
    EventExtraction,
    ExtractorProviderError,
)
from run4221.events import DISTANCE_KEY_TO_CODE, normalize_event_id
from run4221.ingestion.page_snapshot import PageSnapshot

MAX_SNAPSHOT_TEXT_CHARS = 25_000
MAX_LINKS = 80
MAX_CANDIDATE_LINKS = 20

REGISTRATION_LINK_TERMS = (
    "register",
    "registration",
    "entry",
    "entries",
    "sign up",
    "signup",
    "anmeldung",
    "anmelden",
    "teilnahme",
    "startplatz",
    "buchung",
)

EVENT_DETAIL_LINK_TERMS = (
    "wettbewerbe",
    "race",
    "races",
    "event",
    "events",
    "marathon",
    "half",
    "halb",
    "21",
    "42",
)

class OpenAIEventExtraction(BaseModel):
    name: str = Field(description="Official public event name.")
    public_id: str = Field(description="Lowercase public event ID such as berlin.42.")
    city: str
    country: str
    timezone: str
    event_date: str | None
    distances: list[str]
    regions: list[str]
    official_url: str
    registration_url: str | None
    confidence: float
    evidence_snippets: list[str]


class OpenAIExtractorProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: SecretStr | str,
        model: str,
        instructions: str,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.instructions = instructions.strip()
        if not self.instructions:
            raise ValueError("OpenAI extractor instructions cannot be empty.")
        self.client = client or AsyncOpenAI(api_key=secret_value(api_key))

    async def extract(self, snapshot: PageSnapshot) -> EventExtraction:
        try:
            response = await self.client.responses.parse(
                model=self.model,
                instructions=self.instructions,
                input=build_snapshot_prompt(snapshot),
                text_format=OpenAIEventExtraction,
                max_output_tokens=1_500,
            )
        except OpenAIError as error:
            raise ExtractorProviderError(format_openai_error(error)) from error

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise ExtractorProviderError("OpenAI extraction returned no parsed output.")

        return parsed_to_event_extraction(parsed, snapshot)


def secret_value(value: SecretStr | str) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()

    return value


def build_snapshot_prompt(snapshot: PageSnapshot) -> str:
    payload = {
        "source_url": snapshot.source_url,
        "final_url": snapshot.final_url,
        "status_code": snapshot.status_code,
        "content_type": snapshot.content_type,
        "title": snapshot.title,
        "text_hash": snapshot.text_hash,
        "links": [
            {"url": link.url, "text": link.text}
            for link in snapshot.links[:MAX_LINKS]
        ],
        "candidate_links": candidate_links_for_extraction(snapshot),
        "normalized_text": snapshot.normalized_text[:MAX_SNAPSHOT_TEXT_CHARS],
    }
    return "Extract event metadata from this page snapshot:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )


def parsed_to_event_extraction(
    parsed: OpenAIEventExtraction,
    snapshot: PageSnapshot,
) -> EventExtraction:
    distances = normalize_distances(parsed.distances)
    regions = normalize_regions(parsed.regions)
    official_url = parsed.official_url.strip() or snapshot.final_url

    return EventExtraction(
        name=parsed.name.strip() or "Unknown Event",
        public_id=normalize_public_id(parsed.public_id, parsed.city, distances),
        city=parsed.city.strip() or "Unknown",
        country=parsed.country.strip() or "Unknown",
        timezone=parsed.timezone.strip() or "Etc/UTC",
        event_date=optional_text(parsed.event_date),
        distances=distances,
        regions=regions,
        official_url=official_url,
        registration_url=optional_text(parsed.registration_url),
        confidence=clamp(parsed.confidence),
        evidence_snippets=tuple(
            snippet.strip()
            for snippet in parsed.evidence_snippets
            if snippet.strip()
        ),
        provider_name=OpenAIExtractorProvider.provider_name,
    )


def normalize_distances(values: list[str]) -> tuple[str, ...]:
    distances: list[str] = []
    for value in values:
        normalized = value.casefold().strip().replace("-", "_").replace(" ", "_")
        if normalized in {"42", "42k", "marathon"}:
            distances.append("marathon")
        elif normalized in {"21", "21k", "half", "half_marathon"}:
            distances.append("half_marathon")

    return tuple(dict.fromkeys(distances)) or ("marathon",)


def normalize_regions(values: list[str]) -> tuple[str, ...]:
    regions = tuple(
        dict.fromkeys(
            value.casefold().strip().removeprefix("#")
            for value in values
            if value.strip()
        )
    )
    return regions or ("global",)


def normalize_public_id(public_id: str, city: str, distances: tuple[str, ...]) -> str:
    normalized = normalize_event_id(public_id)
    if "." in normalized:
        return normalized

    suffix = DISTANCE_KEY_TO_CODE.get(distances[0], "42")
    place = normalize_event_id(city or normalized or "event")
    return f"{place}.{suffix}"


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def format_openai_error(error: OpenAIError) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code == 401:
        return "OpenAI authentication failed; check OPENAI_API_KEY."
    if status_code == 429:
        return "OpenAI rate limit or quota error."
    if status_code is not None:
        return f"OpenAI request failed with status {status_code}."

    return f"OpenAI request failed: {error.__class__.__name__}."


def candidate_links_for_extraction(snapshot: PageSnapshot) -> list[dict[str, object]]:
    candidates = []
    seen_urls: set[str] = set()
    for link in snapshot.links:
        if link.url in seen_urls:
            continue
        seen_urls.add(link.url)

        score, reasons = score_candidate_link(link.url, link.text)
        if score <= 0:
            continue

        candidates.append(
            {
                "url": link.url,
                "text": link.text,
                "score": score,
                "reasons": reasons,
            }
        )

    return sorted(candidates, key=lambda item: (-int(item["score"]), str(item["url"])))[
        :MAX_CANDIDATE_LINKS
    ]


def score_candidate_link(url: str, text: str) -> tuple[int, list[str]]:
    searchable = f"{url} {text}".casefold()
    score = 0
    reasons: list[str] = []

    if term_matches(searchable, REGISTRATION_LINK_TERMS):
        score += 100
        reasons.append("registration-term")
    if "wettbewerbe" in searchable:
        score += 60
        reasons.append("event-detail-page")
    if term_matches(searchable, ("marathon", "42")):
        score += 40
        reasons.append("marathon-distance-page")
    if term_matches(searchable, ("half", "halb", "21")):
        score += 40
        reasons.append("half-marathon-distance-page")
    if not reasons:
        return 0, []

    return score, reasons


def term_matches(value: str, terms: tuple[str, ...]) -> bool:
    return any(term in value for term in terms)
