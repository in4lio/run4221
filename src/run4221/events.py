from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def normalize_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold()).replace("_", " ")
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[-.]+", " ", normalized)
    normalized = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in normalized
    )
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_tag_alias(value: str) -> str:
    normalized = normalize_query(value).replace(" ", "_").strip("_#")
    return normalized.removeprefix("#")


@dataclass(frozen=True)
class TagDefinition:
    tag: str
    label: str
    kind: str
    aliases: tuple[str, ...]

    @property
    def display_alias(self) -> str:
        return self.aliases[0] if self.aliases else self.tag


TAG_DEFINITIONS = (
    TagDefinition("global", "Global", "region", ("global",)),
    TagDefinition("eu", "EU", "region", ("eu", "europe")),
    TagDefinition("ru", "Russia", "region", ("ru", "russia")),
    TagDefinition("de", "Germany", "region", ("de", "germany", "deutschland")),
    TagDefinition("us", "United States", "region", ("us", "united states", "united_states")),
    TagDefinition("marathon", "Marathon", "distance", ("42", "42k", "marathon")),
    TagDefinition(
        "half_marathon",
        "Half marathon",
        "distance",
        ("21", "21k", "half", "half marathon", "half_marathon"),
    ),
    TagDefinition(
        "major",
        "Major",
        "group",
        ("major", "majors", "world marathon majors", "world_marathon_majors"),
    ),
    TagDefinition(
        "superhalf",
        "SuperHalf",
        "group",
        ("superhalf", "superhalfs", "super half", "super_half", "super_halfs"),
    ),
)

TAG_BY_TAG = {definition.tag: definition for definition in TAG_DEFINITIONS}
TAG_ALIAS_TO_TAG = {
    normalize_tag_alias(alias): definition.tag
    for definition in TAG_DEFINITIONS
    for alias in (definition.tag, definition.label, *definition.aliases)
}

DISTANCE_LABELS = {
    definition.tag: definition.label
    for definition in TAG_DEFINITIONS
    if definition.kind == "distance"
}

DISTANCE_CODE_TO_KEY = {
    definition.display_alias: definition.tag
    for definition in TAG_DEFINITIONS
    if definition.kind == "distance" and definition.display_alias.isdigit()
}

DISTANCE_KEY_TO_CODE = {value: key for key, value in DISTANCE_CODE_TO_KEY.items()}

COLLECTION_LABELS = {
    "world_marathon_majors": "World Marathon Majors",
    "superhalfs": "SuperHalfs",
}

REGION_LABELS = {
    definition.tag: definition.label
    for definition in TAG_DEFINITIONS
    if definition.kind == "region"
}

COLLECTION_TAG_ALIASES = {
    "world_marathon_majors": "major",
    "superhalfs": "superhalf",
}

TAG_LABELS = {
    definition.tag: definition.label
    for definition in TAG_DEFINITIONS
}


@dataclass(frozen=True)
class EventLookup:
    exact: TrackedEvent | None
    suggestions: tuple[TrackedEvent, ...]


@dataclass(frozen=True)
class TrackedEvent:
    id: str
    public_id: str
    legacy_ids: tuple[str, ...]
    search_keywords: tuple[str, ...]
    name: str
    city: str
    country: str
    timezone: str
    distances: tuple[str, ...]
    regions: tuple[str, ...]
    collections: tuple[str, ...]
    event_date: str | None
    registration_status: str
    official_url: str
    registration_url: str | None
    registration_open_at: str | None = None
    registration_open_precision: str = "unknown"
    registration_close_at: str | None = None

    @property
    def location(self) -> str:
        return f"{self.city}, {self.country}"

    @property
    def distance_label(self) -> str:
        return ", ".join(DISTANCE_LABELS.get(distance, distance) for distance in self.distances)

    @property
    def region_label(self) -> str:
        return ", ".join(REGION_LABELS.get(region, region.upper()) for region in self.regions)

    @property
    def collection_label(self) -> str:
        return ", ".join(
            COLLECTION_LABELS.get(collection, collection) for collection in self.collections
        )

    @property
    def tags(self) -> tuple[str, ...]:
        collection_tags = (
            COLLECTION_TAG_ALIASES.get(collection, collection)
            for collection in self.collections
        )
        tags = [
            *self.regions,
            *collection_tags,
            *self.distances,
        ]
        return tuple(dict.fromkeys(normalize_tag(tag) for tag in tags if tag))

    @property
    def tag_label(self) -> str:
        return ", ".join(format_tag_display(tag) for tag in self.tags)

    @property
    def search_text(self) -> str:
        tag_terms = [term for tag in self.tags for term in tag_search_terms(tag)]
        parts = [
            self.id,
            self.public_id,
            *self.legacy_ids,
            *self.search_keywords,
            self.name,
            self.city,
            self.country,
            self.timezone,
            *self.distances,
            *self.regions,
            *self.collections,
            *self.tags,
            *tag_terms,
            self.distance_label,
            self.region_label,
            self.collection_label,
            self.tag_label,
        ]
        return normalize_query(" ".join(parts))


def normalize_tag(value: str) -> str:
    normalized = normalize_tag_alias(value)
    return TAG_ALIAS_TO_TAG.get(normalized, normalized)


def tag_search_terms(tag: str) -> tuple[str, ...]:
    normalized = normalize_tag(tag)
    definition = TAG_BY_TAG.get(normalized)
    if definition is None:
        return (normalized, format_tag_label(normalized))

    terms = (definition.tag, definition.label, *definition.aliases)
    return tuple(dict.fromkeys(term for term in terms if term))


def format_tag_display(tag: str) -> str:
    definition = TAG_BY_TAG.get(normalize_tag(tag))
    if definition is not None:
        return definition.display_alias

    return tag


def format_tag_label(tag: str) -> str:
    normalized = normalize_tag(tag)
    if normalized in TAG_LABELS:
        return TAG_LABELS[normalized]
    if normalized in REGION_LABELS:
        return REGION_LABELS[normalized]
    if normalized in DISTANCE_LABELS:
        return DISTANCE_LABELS[normalized]

    return normalized.upper() if len(normalized) <= 3 else normalized.replace("_", " ").title()


def event_has_tag(event: TrackedEvent, tag: str) -> bool:
    normalized = normalize_tag(tag)
    return normalized in event.tags


def normalize_event_id(value: str) -> str:
    normalized = value.casefold().strip()
    normalized = re.sub(r"^/show_event(?:@\w+)?\s+", "", normalized)
    normalized = re.sub(r"@\w+$", "", normalized)
    normalized = normalized.replace("<code>", "").replace("</code>", "").replace("`", "")
    normalized = normalized.replace("_", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9.-]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-.")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def seed_events_path() -> Path:
    configured_path = os.getenv("SEED_EVENTS_PATH") or os.getenv("RUN4221_SEED_EVENTS_PATH")
    if configured_path:
        return Path(configured_path)

    private_path = project_root() / "private" / "data" / "seed_events" / "events.json"
    if private_path.exists():
        return private_path

    return project_root() / "data" / "seed_events" / "events.json"


def load_seed_events(path: Path | None = None) -> tuple[TrackedEvent, ...]:
    source = path or seed_events_path()
    if not source.exists():
        return ()

    raw_events = json.loads(source.read_text(encoding="utf-8"))
    return tuple(
        TrackedEvent(
            id=item["id"],
            public_id=item["public_id"],
            legacy_ids=tuple(item.get("legacy_ids", ())),
            search_keywords=tuple(item.get("search_keywords", ())),
            name=item["name"],
            city=item["city"],
            country=item["country"],
            timezone=item["timezone"],
            distances=tuple(item["distances"]),
            regions=tuple(item["regions"]),
            collections=tuple(item["collections"]),
            event_date=item.get("event_date"),
            registration_status=item["registration_status"],
            official_url=item["official_url"],
            registration_url=item.get("registration_url"),
            registration_open_at=item.get("registration_open_at"),
            registration_open_precision=item.get("registration_open_precision", "unknown"),
            registration_close_at=item.get("registration_close_at"),
        )
        for item in raw_events
    )


def get_events() -> tuple[TrackedEvent, ...]:
    from run4221.db.repository import get_events as get_database_events

    return get_database_events()


def find_event(
    event_id: str,
    events: tuple[TrackedEvent, ...] | None = None,
) -> TrackedEvent | None:
    if events is None:
        from run4221.db.repository import find_event as find_database_event

        return find_database_event(event_id)

    normalized_id = normalize_event_id(event_id)
    for event in events:
        candidate_ids = (event.public_id, event.id, *event.legacy_ids)
        if any(normalize_event_id(candidate_id) == normalized_id for candidate_id in candidate_ids):
            return event
    return None


def search_events(
    query: str,
    events: tuple[TrackedEvent, ...] | None = None,
) -> tuple[TrackedEvent, ...]:
    if events is None:
        from run4221.db.repository import search_events as search_database_events

        return search_database_events(query)

    normalized = normalize_query(query)
    if not normalized:
        return ()

    terms = normalized.split()
    matches = [event for event in events if matches_search_terms(event, terms)]
    return tuple(matches)


def list_events_by_tag(
    tag: str,
    events: tuple[TrackedEvent, ...] | None = None,
    *,
    limit: int = 10,
) -> tuple[TrackedEvent, ...]:
    if events is None:
        from run4221.db.repository import list_events_by_tag as list_database_events_by_tag

        return list_database_events_by_tag(tag, limit=limit)

    tagged_events = [event for event in events if event_has_tag(event, tag)]
    return tuple(sorted(tagged_events, key=event_sort_key)[:limit])


def matches_search_terms(event: TrackedEvent, terms: list[str]) -> bool:
    search_text = event.search_text
    search_tokens = set(search_text.split())
    return all(term in search_tokens if len(term) <= 3 else term in search_text for term in terms)


def resolve_event_lookup(
    query: str,
    events: tuple[TrackedEvent, ...] | None = None,
    *,
    limit: int = 5,
) -> EventLookup:
    if events is None:
        from run4221.db.repository import resolve_event_lookup as resolve_database_event_lookup

        return resolve_database_event_lookup(query, limit=limit)

    exact = find_event(query, events)
    if exact is not None:
        return EventLookup(exact=exact, suggestions=())

    return EventLookup(exact=None, suggestions=search_events(query, events)[:limit])


def list_events(
    events: tuple[TrackedEvent, ...] | None = None,
    *,
    limit: int = 10,
) -> tuple[TrackedEvent, ...]:
    if events is None:
        from run4221.db.repository import list_events as list_database_events

        return list_database_events(limit=limit)

    return tuple(sorted(events, key=event_sort_key)[:limit])


def list_open_events(
    events: tuple[TrackedEvent, ...] | None = None,
    *,
    limit: int = 10,
    tag: str | None = None,
) -> tuple[TrackedEvent, ...]:
    if events is None:
        from run4221.db.repository import list_open_events as list_open_database_events

        return list_open_database_events(limit=limit, tag=tag)

    open_events = [
        event
        for event in events
        if event.registration_status in {"open", "waitlist"}
    ]
    if tag:
        open_events = [event for event in open_events if event_has_tag(event, tag)]

    return tuple(sorted(open_events, key=event_sort_key)[:limit])


def event_sort_key(event: TrackedEvent) -> tuple[str, str, str]:
    return (event.event_date or "9999-12-31", event.name.casefold(), event.id)
