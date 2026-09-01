from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

from run4221.ai.extraction_provider import (
    EventExtraction,
    ExtractorProvider,
    ExtractorProviderError,
)
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageSnapshot,
    blocked_page_reason,
    fetch_enriched_page_snapshot,
    store_page_snapshot,
)

type SnapshotFetcher = Callable[[str], Awaitable[PageSnapshot]]
type SnapshotStore = Callable[[PageSnapshot], Path]


@dataclass(frozen=True)
class EventDraft:
    source_url: str
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
    evidence: str
    registration_url_candidates: tuple[tuple[str, str], ...] = ()


STOPWORDS = {
    "www",
    "com",
    "org",
    "net",
    "de",
    "en",
    "race",
    "races",
    "run",
    "running",
    "registration",
    "register",
    "official",
    "event",
    "events",
    "tcs",
    "bmw",
    "edp",
    "generali",
    "bank",
    "america",
}

NON_PRIMARY_EVENT_TERMS = (
    "mini",
    "kids",
    "kid",
    "youth",
    "junior",
    "children",
    "school",
    "schueler",
    "schüler",
)

HALF_MARATHON_TERMS = (
    "half",
    "halfmarathon",
    "halb",
    "halbmarathon",
    "21",
    "21k",
    "21km",
)

FULL_MARATHON_TERMS = (
    "full",
    "42",
    "42k",
    "42km",
)


async def extract_event_draft_from_url(
    url: str,
    *,
    fetch_snapshot: SnapshotFetcher | None = fetch_enriched_page_snapshot,
    store_snapshot: SnapshotStore | None = store_page_snapshot,
    extractor_provider: ExtractorProvider | None = None,
) -> EventDraft:
    """Fetch a URL snapshot and create a structured event draft.

    The default provider is deterministic. OpenAI/Gemini providers can implement the same
    ExtractorProvider contract later.
    """

    if fetch_snapshot is None:
        return draft_from_url_text(url)

    try:
        snapshot = await fetch_snapshot(url)
    except PageFetchError as error:
        return draft_from_url_text(
            url,
            confidence=0.05,
            evidence=f"Page fetch failed: {error}. Fallback draft uses URL text only.",
        )

    snapshot_path = None
    storage_note = ""
    if store_snapshot is not None:
        try:
            snapshot_path = store_snapshot(snapshot)
        except OSError as error:
            storage_note = f" Snapshot storage failed: {error}."

    if reason := blocked_page_reason(snapshot):
        evidence_parts = snapshot_evidence(snapshot, snapshot_path, storage_note)
        evidence_parts.append(f"Page blocked: {reason}.")
        evidence_parts.append("Fallback extraction from URL text only.")
        evidence_parts.append("Extractor provider: url-fallback.")
        return draft_from_url_text(
            snapshot.source_url,
            confidence=0.03,
            evidence=" ".join(evidence_parts),
        )

    return await draft_from_page_snapshot(
        snapshot,
        snapshot_path=snapshot_path,
        storage_note=storage_note,
        extractor_provider=extractor_provider,
    )


def draft_from_url_text(
    url: str,
    *,
    confidence: float = 0.1,
    evidence: str = "Fallback extraction from URL text only. No page evidence was available.",
) -> EventDraft:
    parsed = urlparse(url)
    tokens = url_tokens(parsed.netloc, parsed.path)
    distance = infer_distance(tokens)
    city_token = first_content_token(tokens) or "event"
    public_id = f"{city_token}.{'21' if distance == 'half_marathon' else '42'}"
    name = infer_name(tokens, distance)

    return EventDraft(
        source_url=url,
        name=name,
        public_id=public_id,
        city=city_token.replace("-", " ").title(),
        country="Unknown",
        timezone="Etc/UTC",
        event_date=None,
        distances=(distance,),
        regions=("global",),
        official_url=url,
        registration_url=None,
        registration_url_candidates=(),
        confidence=confidence,
        evidence=evidence,
    )


async def draft_from_page_snapshot(
    snapshot: PageSnapshot,
    *,
    snapshot_path: Path | None,
    storage_note: str = "",
    extractor_provider: ExtractorProvider | None = None,
) -> EventDraft:
    provider = extractor_provider or HeuristicExtractorProvider()
    evidence_parts = snapshot_evidence(snapshot, snapshot_path, storage_note)
    try:
        extraction = await provider.extract(snapshot)
    except ExtractorProviderError as error:
        if isinstance(provider, HeuristicExtractorProvider):
            raise

        evidence_parts.append(
            f"Extractor provider {provider.provider_name} failed: {error}. "
            "Falling back to deterministic extractor."
        )
        provider = HeuristicExtractorProvider()
        extraction = await provider.extract(snapshot)

    if extraction.evidence_snippets:
        evidence_parts.extend(extraction.evidence_snippets)
    registration_url_candidates = collect_registration_url_candidates(snapshot, extraction)
    registration_url = resolve_registration_url(
        extraction.registration_url,
        snapshot,
        extraction,
        candidates=registration_url_candidates,
    )
    if extraction.registration_url is None and registration_url is not None:
        evidence_parts.append(f"Selected registration URL from page links: {registration_url}.")
    timezone = resolve_timezone(
        extraction.timezone,
        extraction.country,
        extraction.regions,
        extraction.city,
    )
    if timezone != extraction.timezone:
        evidence_parts.append(f"Selected timezone from location: {timezone}.")
    evidence_parts.append(f"Extractor provider: {extraction.provider_name}.")
    if isinstance(provider, HeuristicExtractorProvider):
        evidence_parts.append("AI provider is not configured yet; using deterministic fallback.")

    return EventDraft(
        source_url=snapshot.source_url,
        name=extraction.name,
        public_id=extraction.public_id,
        city=extraction.city,
        country=extraction.country,
        timezone=timezone,
        event_date=extraction.event_date,
        distances=extraction.distances,
        regions=extraction.regions,
        official_url=extraction.official_url,
        registration_url=registration_url,
        registration_url_candidates=registration_url_candidates,
        confidence=min(max(extraction.confidence, 0.0), 1.0),
        evidence=" ".join(evidence_parts),
    )


def snapshot_evidence(
    snapshot: PageSnapshot,
    snapshot_path: Path | None,
    storage_note: str,
) -> list[str]:
    evidence_parts = [
        f"Fetched page snapshot with status {snapshot.status_code}.",
        f"Text hash: {snapshot.text_hash[:12]}.",
    ]
    if snapshot_path is not None:
        evidence_parts.append(f"Stored snapshot: {snapshot_path}.")
    if snapshot.title:
        evidence_parts.append(f"Title: {snapshot.title}.")
    if storage_note:
        evidence_parts.append(storage_note.strip())

    return evidence_parts


class HeuristicExtractorProvider:
    provider_name = "heuristic"

    async def extract(self, snapshot: PageSnapshot) -> EventExtraction:
        tokens = snapshot_source_tokens(snapshot)
        distance = infer_distance(tokens)
        name = infer_name_from_snapshot(snapshot, tokens, distance)
        city_token = infer_city_token(name, tokens) or "event"
        public_id = f"{city_token}.{'21' if distance == 'half_marathon' else '42'}"
        event_date = infer_event_date(snapshot.normalized_text)
        registration_url = infer_registration_url(snapshot)

        confidence = 0.25
        if snapshot.title:
            confidence += 0.1
        if event_date:
            confidence += 0.1
        if registration_url:
            confidence += 0.1

        evidence = []
        if event_date:
            evidence.append(f"Detected event date: {event_date}.")
        if registration_url:
            evidence.append(f"Detected registration link: {registration_url}.")

        return EventExtraction(
            name=name,
            public_id=public_id,
            city=city_token.replace("-", " ").title(),
            country="Unknown",
            timezone="Etc/UTC",
            event_date=event_date,
            distances=(distance,),
            regions=("global",),
            official_url=snapshot.final_url,
            registration_url=registration_url,
            confidence=min(confidence, 0.6),
            evidence_snippets=tuple(evidence),
            provider_name=self.provider_name,
        )


def url_tokens(netloc: str, path: str) -> tuple[str, ...]:
    source = f"{netloc} {path}"
    tokens = tuple(token for token in re.split(r"[^a-zA-Z0-9]+", source.casefold()) if token)
    return tokens


def text_tokens(value: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-zA-Z0-9]+", value.casefold()) if token)


def snapshot_source_tokens(snapshot: PageSnapshot) -> tuple[str, ...]:
    parsed = urlparse(snapshot.final_url)
    return (
        *url_tokens(parsed.netloc, parsed.path),
        *text_tokens(snapshot.title or ""),
        *text_tokens(snapshot.normalized_text[:2_000]),
    )


def infer_distance(tokens: tuple[str, ...]) -> str:
    token_set = set(tokens)
    if {"half", "21", "21k", "halb"} & token_set:
        return "half_marathon"

    return "marathon"


def first_content_token(tokens: tuple[str, ...]) -> str | None:
    for token in tokens:
        if token not in STOPWORDS and token not in {"marathon", "half", "21", "42", "42k"}:
            return token

    return None


def infer_name(tokens: tuple[str, ...], distance: str) -> str:
    content_tokens = [
        token
        for token in tokens
        if token not in STOPWORDS and token not in {"21", "42", "42k"}
    ]
    if not content_tokens:
        return "Unknown Event"

    if "marathon" not in content_tokens:
        content_tokens.append("half marathon" if distance == "half_marathon" else "marathon")

    return " ".join(content_tokens[:5]).title()


def infer_name_from_snapshot(
    snapshot: PageSnapshot,
    tokens: tuple[str, ...],
    distance: str,
) -> str:
    candidates: list[str] = []
    if snapshot.title:
        candidates.extend(re.split(r"\s+[|–—-]\s+", snapshot.title))
        candidates.append(snapshot.title)
    candidates.extend(re.split(r"[.;]", snapshot.normalized_text[:600]))

    for candidate in candidates:
        name = clean_event_name_candidate(candidate)
        if "marathon" in name.casefold():
            return name

    return infer_name(tokens, distance)


def clean_event_name_candidate(value: str) -> str:
    cleaned = re.sub(r"\b20\d{2}\b", "", value)
    cleaned = re.sub(
        r"\b(registration|register|entries|entry|official website|official site|homepage)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -|–—")


def infer_city_token(name: str, tokens: tuple[str, ...]) -> str | None:
    for token in text_tokens(name):
        if token not in STOPWORDS and token not in {"marathon", "half", "21", "42", "42k"}:
            return token

    return first_content_token(tokens)


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def infer_event_date(text: str) -> str | None:
    for match in re.finditer(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text):
        parsed = build_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        if parsed:
            return parsed

    month_names = "|".join(MONTHS)
    day_month = re.compile(rf"\b(\d{{1,2}})\s+({month_names})\s+(20\d{{2}})\b", re.I)
    for match in day_month.finditer(text):
        parsed = build_date(
            int(match.group(3)),
            MONTHS[match.group(2).casefold()],
            int(match.group(1)),
        )
        if parsed:
            return parsed

    month_day = re.compile(rf"\b({month_names})\s+(\d{{1,2}}),?\s+(20\d{{2}})\b", re.I)
    for match in month_day.finditer(text):
        parsed = build_date(
            int(match.group(3)),
            MONTHS[match.group(1).casefold()],
            int(match.group(2)),
        )
        if parsed:
            return parsed

    return None


def build_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


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
)

URL_PATTERN = re.compile(r"https?://[^\s\"<>]+")
DATED_ARTICLE_PATH = re.compile(r"/20\d{2}/\d{1,2}/\d{1,2}/")
ARTICLE_PATH_TERMS = (
    "/blog/",
    "/news/",
    "/press/",
    "/story/",
    "/stories/",
    "/article/",
)

CITY_TIMEZONES = {
    "berlin": "Europe/Berlin",
    "karlsruhe": "Europe/Berlin",
    "lisbon": "Europe/Lisbon",
    "prague": "Europe/Prague",
    "copenhagen": "Europe/Copenhagen",
    "cardiff": "Europe/London",
    "valencia": "Europe/Madrid",
    "tokyo": "Asia/Tokyo",
    "boston": "America/New_York",
    "chicago": "America/Chicago",
    "new york": "America/New_York",
    "new york city": "America/New_York",
    "london": "Europe/London",
    "sydney": "Australia/Sydney",
}

COUNTRY_TIMEZONES = {
    "germany": "Europe/Berlin",
    "portugal": "Europe/Lisbon",
    "czech republic": "Europe/Prague",
    "czechia": "Europe/Prague",
    "denmark": "Europe/Copenhagen",
    "united kingdom": "Europe/London",
    "uk": "Europe/London",
    "spain": "Europe/Madrid",
    "japan": "Asia/Tokyo",
    "australia": "Australia/Sydney",
}

REGION_TIMEZONES = {
    "de": "Europe/Berlin",
    "pt": "Europe/Lisbon",
    "cz": "Europe/Prague",
    "dk": "Europe/Copenhagen",
    "gb": "Europe/London",
    "es": "Europe/Madrid",
    "jp": "Asia/Tokyo",
}


def infer_registration_url(
    snapshot: PageSnapshot,
    distances: tuple[str, ...] = (),
) -> str | None:
    for link in snapshot.links:
        searchable = f"{link.text} {link.url}".casefold()
        if (
            any(term in searchable for term in REGISTRATION_LINK_TERMS)
            and not conflicts_with_event_identity(link.url, link.text, distances)
        ):
            return link.url

    return None


def resolve_registration_url(
    explicit_url: str | None,
    snapshot: PageSnapshot,
    extraction: EventExtraction,
    *,
    candidates: tuple[tuple[str, str], ...] | None = None,
) -> str | None:
    if (
        explicit_url
        and not is_likely_article_url(explicit_url, extraction.event_date)
        and not conflicts_with_event_identity(explicit_url, "", extraction.distances)
    ):
        return explicit_url

    event_specific_url = infer_event_specific_url(
        snapshot,
        extraction,
        candidates=candidates,
    )
    if event_specific_url is not None:
        return event_specific_url

    return infer_registration_url(snapshot, extraction.distances)


def infer_event_specific_url(
    snapshot: PageSnapshot,
    extraction: EventExtraction,
    *,
    candidates: tuple[tuple[str, str], ...] | None = None,
) -> str | None:
    candidate_links = candidates or collect_registration_url_candidates(snapshot, extraction)
    return select_registration_url_for_distances(candidate_links, extraction.distances)


def collect_registration_url_candidates(
    snapshot: PageSnapshot,
    extraction: EventExtraction,
) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def append(url: str | None, text: str) -> None:
        if not url:
            return
        cleaned_url = clean_url_candidate(url)
        if cleaned_url is None:
            return
        cleaned_text = text.strip()
        key = (cleaned_url, cleaned_text)
        if key in seen:
            return
        candidates.append(key)
        seen.add(key)

    append(extraction.registration_url, "extracted registration URL")
    for url, text in url_candidates_from_evidence(extraction.evidence_snippets):
        append(url, text)
    for link in snapshot.links:
        append(link.url, link.text)

    return tuple(candidates)


def select_registration_url_for_distances(
    candidates: tuple[tuple[str, str], ...],
    distances: tuple[str, ...],
    *,
    fallback: str | None = None,
) -> str | None:
    if len(distances) != 1:
        return fallback

    for url, text in candidates:
        if is_homepage_url(url):
            continue
        if is_likely_article_url(url):
            continue
        if conflicts_with_event_identity(url, text, distances):
            continue
        if matches_distance_url(url, text, distances):
            return url

    return fallback


def url_candidates_from_evidence(snippets: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    candidates: list[tuple[str, str]] = []
    for snippet in snippets:
        for match in URL_PATTERN.finditer(snippet):
            url = clean_url_candidate(match.group(0))
            if url:
                candidates.append((url, snippet))

    return tuple(candidates)


def clean_url_candidate(value: str) -> str | None:
    candidate = value.strip().strip("'\"").rstrip(".,);]")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    return candidate


def is_likely_article_url(value: str, event_date: str | None = None) -> bool:
    parsed = urlparse(value)
    path = parsed.path.casefold()
    searchable = f"{path} {parsed.query}".casefold()
    has_registration_term = any(term in searchable for term in REGISTRATION_LINK_TERMS)
    if has_registration_term:
        return False

    if any(term in path for term in ARTICLE_PATH_TERMS):
        return True

    if DATED_ARTICLE_PATH.search(path):
        return True

    if event_date:
        event_year = event_date[:4]
        years = re.findall(r"\b20\d{2}\b", path)
        return any(year < event_year for year in years)

    return False


def is_homepage_url(value: str) -> bool:
    parsed = urlparse(value)
    return (parsed.path.rstrip("/") or "/") == "/" and not parsed.query


def matches_distance_url(url: str, text: str, distances: tuple[str, ...]) -> bool:
    tokens = event_identity_tokens(url, text)
    if "half_marathon" in distances:
        return any(term in tokens for term in HALF_MARATHON_TERMS)
    if "marathon" in distances:
        return "marathon" in tokens and not any(
            term in tokens for term in HALF_MARATHON_TERMS
        )

    return False


def conflicts_with_event_identity(
    url: str,
    text: str,
    distances: tuple[str, ...],
) -> bool:
    if not distances:
        return False
    tokens = event_identity_tokens(url, text)
    if any(term in tokens for term in NON_PRIMARY_EVENT_TERMS):
        return True
    if "marathon" in distances and "half_marathon" not in distances:
        return any(term in tokens for term in HALF_MARATHON_TERMS)
    if "half_marathon" in distances and "marathon" not in distances:
        return any(term in tokens for term in FULL_MARATHON_TERMS)
    return False


def event_identity_tokens(url: str, text: str) -> frozenset[str]:
    parsed = urlparse(url)
    searchable = unquote(f"{parsed.path} {parsed.query} {text}").casefold()
    return frozenset(re.findall(r"[^\W_]+", searchable))


def resolve_timezone(
    timezone: str,
    country: str,
    regions: tuple[str, ...],
    city: str,
) -> str:
    cleaned_timezone = timezone.strip()
    if cleaned_timezone and cleaned_timezone != "Etc/UTC":
        return cleaned_timezone

    normalized_city = normalize_lookup_key(city)
    if normalized_city in CITY_TIMEZONES:
        return CITY_TIMEZONES[normalized_city]

    normalized_country = normalize_lookup_key(country)
    if normalized_country in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[normalized_country]

    for region in regions:
        normalized_region = region.casefold().strip().removeprefix("#")
        if normalized_region in REGION_TIMEZONES:
            return REGION_TIMEZONES[normalized_region]

    return cleaned_timezone or "Etc/UTC"


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())
