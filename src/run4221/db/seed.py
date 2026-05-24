from __future__ import annotations

from sqlalchemy.orm import Session

from run4221.db import models
from run4221.events import COLLECTION_LABELS, REGION_LABELS, TrackedEvent, load_seed_events

REGION_SCOPES = {
    "global": "global",
    "eu": "continent",
    "ru": "country",
    "de": "country",
    "us": "country",
}


def seed_initial_data(session: Session, events: tuple[TrackedEvent, ...] | None = None) -> None:
    seed_regions(session)
    seed_collections(session)

    seed_events = load_seed_events() if events is None else events
    for index, event in enumerate(seed_events, start=1):
        upsert_event(session, event, sort_order=index)


def seed_regions(session: Session) -> None:
    for tag, name in REGION_LABELS.items():
        region = session.get(models.Region, tag)
        if region is None:
            region = models.Region(tag=tag, name=name, scope=REGION_SCOPES.get(tag, "custom"))
            session.add(region)
        else:
            region.name = name
            region.scope = REGION_SCOPES.get(tag, region.scope)
            region.active = True


def seed_collections(session: Session) -> None:
    for slug, name in COLLECTION_LABELS.items():
        collection = session.get(models.EventCollection, slug)
        if collection is None:
            collection = models.EventCollection(slug=slug, name=name)
            session.add(collection)
        else:
            collection.name = name
            collection.active = True


def upsert_event(session: Session, event: TrackedEvent, *, sort_order: int) -> None:
    model = session.get(models.Event, event.id)
    if model is None:
        model = models.Event(id=event.id, public_id=event.public_id)
        session.add(model)

    model.public_id = event.public_id
    model.canonical_name = event.name
    model.city = event.city
    model.country = event.country
    model.timezone = event.timezone
    model.next_event_date = event.event_date
    model.distances = list(event.distances)
    model.registration_status = event.registration_status
    model.status = (
        event.registration_status if event.registration_status != "unknown" else "monitoring"
    )
    model.recurrence = "annual"
    model.official_url = event.official_url
    model.registration_url = event.registration_url
    model.creation_source = "seed_collection"
    model.removed_at = None

    replace_children(
        model.legacy_ids,
        [
            models.EventLegacyId(legacy_id=legacy_id, reason="seed_slug")
            for legacy_id in event.legacy_ids
        ],
    )
    replace_children(
        model.search_keywords,
        [
            models.EventSearchKeyword(keyword=keyword, keyword_type="seed")
            for keyword in event.search_keywords
        ],
    )
    replace_children(
        model.regions,
        [
            models.EventRegion(region_tag=region_tag, is_primary=index == 0)
            for index, region_tag in enumerate(event.regions)
        ],
    )
    replace_children(
        model.collections,
        [
            models.EventCollectionMember(collection_slug=collection_slug, sort_order=sort_order)
            for collection_slug in event.collections
        ],
    )
    replace_children(
        model.sources,
        build_sources(event),
    )
    replace_children(
        model.editions,
        build_editions(event),
    )
    session.flush()

    current_edition = next((edition for edition in model.editions if edition.is_current), None)
    model.current_edition_id = current_edition.id if current_edition is not None else None


def replace_children(target: list[object], replacement: list[object]) -> None:
    target.clear()
    target.extend(replacement)


def build_sources(event: TrackedEvent) -> list[models.EventSource]:
    sources = [
        models.EventSource(url=event.official_url, source_type="official_site", priority=10),
    ]
    if event.registration_url and event.registration_url != event.official_url:
        sources.append(
            models.EventSource(
                url=event.registration_url,
                source_type="registration_page",
                priority=20,
            )
        )

    return sources


def build_editions(event: TrackedEvent) -> list[models.EventEdition]:
    if event.event_date is None:
        return []

    edition_year = int(event.event_date[:4])
    return [
        models.EventEdition(
            edition_year=edition_year,
            edition_label=str(edition_year),
            event_date=event.event_date,
            status="date_announced",
            is_current=True,
        )
    ]
