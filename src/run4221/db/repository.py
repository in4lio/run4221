from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse, urlunparse

from sqlalchemy import Select, delete, select
from sqlalchemy import update as sqlalchemy_update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload

from run4221.db import models
from run4221.db.bootstrap import ensure_database_schema
from run4221.db.models import utcnow
from run4221.db.session import run_serialized_transaction, session_scope
from run4221.events import (
    DISTANCE_CODE_TO_KEY,
    REGION_LABELS,
    EventLookup,
    TrackedEvent,
    event_has_tag,
    event_sort_key,
    matches_search_terms,
    normalize_event_id,
    normalize_query,
)

OPEN_REGISTRATION_STATUSES = {"open", "waitlist"}
REGISTRATION_STATUSES = {"unknown", "announced", "open", "waitlist", "closed", "sold_out"}
REGISTRATION_OPEN_PRECISIONS = {
    "unknown",
    "date_only",
    "datetime",
    "month_only",
    "estimated",
}
EVENT_SUGGESTION_STATUSES = {"pending", "converted", "removed"}
EVENT_SUGGESTION_MAX_PENDING_TOTAL = 30
EVENT_SUGGESTION_MAX_PENDING_PER_USER = 3


@dataclass(frozen=True)
class EventCreate:
    public_id: str
    name: str
    city: str
    country: str
    timezone: str
    distances: tuple[str, ...]
    regions: tuple[str, ...]
    official_url: str
    registration_url: str | None = None
    event_date: str | None = None
    registration_status: str = "unknown"
    registration_open_at: str | None = None
    registration_open_precision: str = "unknown"
    registration_close_at: str | None = None


@dataclass(frozen=True)
class EventUpdate:
    name: str
    city: str
    country: str
    timezone: str
    distances: tuple[str, ...]
    regions: tuple[str, ...]
    official_url: str
    registration_url: str | None = None
    event_date: str | None = None
    registration_status: str = "unknown"
    registration_open_at: str | None = None
    registration_open_precision: str = "unknown"
    registration_close_at: str | None = None


@dataclass(frozen=True)
class EventSuggestionCreate:
    event_name: str
    url: str | None
    event_date: str | None
    location: str | None
    region_tags: tuple[str, ...]
    distances: tuple[str, ...]
    note: str | None
    submitter_user_id: str | None
    submitter_username: str | None
    submitter_display_name: str | None
    submitter_is_moderator: bool = False


@dataclass(frozen=True)
class EventSuggestionRecord:
    id: int
    status: str
    event_name: str
    url: str | None
    event_date: str | None
    location: str | None
    region_tags: tuple[str, ...]
    distances: tuple[str, ...]
    note: str | None
    submitter_user_id: str | None
    submitter_username: str | None
    submitter_display_name: str | None


@dataclass(frozen=True)
class ProposedEventUpdateCreate:
    event_id: str
    update_type: str
    current_fields: dict[str, Any]
    proposed_fields: dict[str, Any]
    evidence: tuple[str, ...]
    confidence: float
    change_summary: str | None = None


@dataclass(frozen=True)
class ProposedEventUpdateRecord:
    id: int
    event_id: str
    update_type: str
    current_fields: dict[str, Any]
    proposed_fields: dict[str, Any]
    evidence: tuple[str, ...]
    confidence: float
    status: str
    change_summary: str | None


@dataclass(frozen=True)
class RegistrationWindowApply:
    registration_status: str
    registration_open_at: str | None = None
    registration_open_precision: str = "unknown"
    registration_close_at: str | None = None
    registration_url: str | None = None
    event_date: str | None = None


@dataclass(frozen=True)
class ProposedEventUpdateApplyResult:
    update: ProposedEventUpdateRecord
    event: TrackedEvent


@dataclass(frozen=True)
class ProposedEventUpdatePartialApplyResult:
    update: ProposedEventUpdateRecord
    event: TrackedEvent
    follow_up_update: ProposedEventUpdateRecord | None
    applied_fields: tuple[str, ...]
    remaining_fields: tuple[str, ...]


@dataclass(frozen=True)
class ArchivedEventRecord:
    event: TrackedEvent
    removed_at: str | None


class EventWriteError(ValueError):
    pass


def get_events(database_url: str | None = None) -> tuple[TrackedEvent, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        events = session.scalars(base_event_query()).all()
        return tuple(event_to_domain(event) for event in events)


def list_events(*, limit: int = 10, database_url: str | None = None) -> tuple[TrackedEvent, ...]:
    return tuple(sorted(get_events(database_url), key=event_sort_key)[:limit])


def list_archived_events(
    *,
    limit: int = 10,
    database_url: str | None = None,
) -> tuple[ArchivedEventRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        rows = session.scalars(
            archived_event_query()
            .order_by(models.Event.removed_at.desc(), models.Event.public_id)
            .limit(limit)
        ).all()
        return tuple(archived_event_to_record(row) for row in rows)


def list_events_by_tag(
    tag: str,
    *,
    limit: int = 10,
    database_url: str | None = None,
) -> tuple[TrackedEvent, ...]:
    events = [event for event in get_events(database_url) if event_has_tag(event, tag)]
    return tuple(sorted(events, key=event_sort_key)[:limit])


def list_open_events(
    *,
    limit: int = 10,
    tag: str | None = None,
    database_url: str | None = None,
) -> tuple[TrackedEvent, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        open_window_event_ids = set(
            session.scalars(
                select(models.RegistrationWindow.event_id).where(
                    models.RegistrationWindow.status.in_(OPEN_REGISTRATION_STATUSES)
                )
            ).all()
        )
        events = [
            event_to_domain(event)
            for event in session.scalars(base_event_query()).all()
            if event.registration_status in OPEN_REGISTRATION_STATUSES
            or event.id in open_window_event_ids
        ]
        if tag:
            events = [event for event in events if event_has_tag(event, tag)]

    return tuple(sorted(events, key=event_sort_key)[:limit])


def find_event(event_id: str, database_url: str | None = None) -> TrackedEvent | None:
    normalized_id = normalize_event_id(event_id)
    for event in get_events(database_url):
        candidate_ids = (event.public_id, event.id, *event.legacy_ids)
        if any(normalize_event_id(candidate_id) == normalized_id for candidate_id in candidate_ids):
            return event

    return None


def find_archived_event(
    event_id: str,
    database_url: str | None = None,
) -> ArchivedEventRecord | None:
    normalized_id = normalize_event_id(event_id)
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        for event in session.scalars(archived_event_query()).all():
            candidate_ids = (event.public_id, event.id, *event.legacy_ids)
            if any(
                normalize_event_id(candidate_id) == normalized_id
                for candidate_id in candidate_ids
            ):
                return archived_event_to_record(event)

    return None


def list_events_by_url(url: str, database_url: str | None = None) -> tuple[TrackedEvent, ...]:
    normalized_url = normalize_url(url)
    if not normalized_url:
        return ()

    ensure_database_schema(database_url)
    matches = []
    with session_scope(database_url) as session:
        for event in session.scalars(base_event_query()).all():
            candidate_urls = [
                event.official_url,
                event.registration_url,
                *(source.url for source in event.sources),
            ]
            if any(
                normalize_url(candidate_url) == normalized_url
                for candidate_url in candidate_urls
            ):
                matches.append(event_to_domain(event))

    return tuple(sorted(matches, key=event_sort_key))


def find_event_by_url(url: str, database_url: str | None = None) -> TrackedEvent | None:
    matches = list_events_by_url(url, database_url)
    return matches[0] if matches else None


def search_events(
    query: str,
    *,
    database_url: str | None = None,
) -> tuple[TrackedEvent, ...]:
    normalized = normalize_query(query)
    if not normalized:
        return ()

    terms = normalized.split()
    return tuple(
        event for event in get_events(database_url) if matches_search_terms(event, terms)
    )


def resolve_event_lookup(
    query: str,
    *,
    limit: int = 5,
    database_url: str | None = None,
) -> EventLookup:
    exact = find_event(query, database_url)
    if exact is not None:
        return EventLookup(exact=exact, suggestions=())

    suggestions = search_events(query, database_url=database_url)[:limit]
    return EventLookup(exact=None, suggestions=suggestions)


def add_event(event: EventCreate, database_url: str | None = None) -> TrackedEvent:
    ensure_database_schema(database_url)
    normalized_public_id = normalize_public_id(event.public_id)
    validate_event_create(event, normalized_public_id)

    with session_scope(database_url) as session:
        return add_event_in_session(session, event, normalized_public_id)


def add_event_from_suggestion(
    event: EventCreate,
    suggestion_id: int,
    database_url: str | None = None,
) -> TrackedEvent:
    ensure_database_schema(database_url)
    normalized_public_id = normalize_public_id(event.public_id)
    validate_event_create(event, normalized_public_id)

    with session_scope(database_url) as session:
        suggestion = session.get(models.EventSuggestion, suggestion_id)
        if suggestion is None or suggestion.status != "pending":
            raise EventWriteError(f"Pending suggestion not found: #{suggestion_id}")

        result = add_event_in_session(session, event, normalized_public_id)
        suggestion.status = "converted"
        session.flush()
        return result


def add_event_in_session(
    session: Session,
    event: EventCreate,
    normalized_public_id: str,
) -> TrackedEvent:
    existing = session.scalar(
        select(models.Event).where(models.Event.public_id == normalized_public_id)
    )
    if existing is not None and existing.removed_at is None:
        raise EventWriteError(f"Event ID already exists: {normalized_public_id}")

    for region_tag in event.regions:
        ensure_region(session, region_tag)

    if existing is not None:
        model = existing
        clear_registration_windows(session, model.id)
        clear_event_children(model)
        session.flush()
    else:
        model = models.Event(id=normalized_public_id, public_id=normalized_public_id)
        session.add(model)

    model.public_id = normalized_public_id
    model.status = "monitoring"
    model.recurrence = "annual"
    model.creation_source = "moderator_direct"
    model.removed_at = None

    replace_children(model.legacy_ids, [])
    apply_event_fields(model, event, normalized_public_id)
    replace_children(model.collections, [])
    session.flush()

    current_edition = next((edition for edition in model.editions if edition.is_current), None)
    model.current_edition_id = current_edition.id if current_edition is not None else None
    registration_window = apply_event_registration_fields(session, model, event)
    if registration_window is not None:
        return event_to_domain(model, registration_window=registration_window)
    return event_to_domain(model)


def update_event(
    event_id: str,
    update: EventUpdate,
    database_url: str | None = None,
) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    existing_event = find_event(event_id, database_url)
    if existing_event is None:
        return None

    event = EventCreate(
        public_id=existing_event.public_id,
        name=update.name,
        city=update.city,
        country=update.country,
        timezone=update.timezone,
        distances=update.distances,
        regions=update.regions,
        official_url=update.official_url,
        registration_url=update.registration_url,
        event_date=update.event_date,
        registration_status=update.registration_status,
        registration_open_at=update.registration_open_at,
        registration_open_precision=update.registration_open_precision,
        registration_close_at=update.registration_close_at,
    )
    validate_event_create(event, existing_event.public_id)

    with session_scope(database_url) as session:
        model = session.get(models.Event, existing_event.id)
        if model is None or model.removed_at is not None:
            return None

        for region_tag in event.regions:
            ensure_region(session, region_tag)

        clear_registration_windows(session, model.id)
        model.current_edition_id = None
        replace_children(model.search_keywords, [])
        replace_children(model.regions, [])
        replace_children(model.sources, [])
        replace_children(model.editions, [])
        session.flush()

        apply_event_fields(model, event, existing_event.public_id)
        session.flush()

        current_edition = next((edition for edition in model.editions if edition.is_current), None)
        model.current_edition_id = current_edition.id if current_edition is not None else None
        registration_window = apply_event_registration_fields(session, model, event)
        if registration_window is not None:
            return event_to_domain(model, registration_window=registration_window)
        return event_to_domain(model)


def add_event_suggestion(
    suggestion: EventSuggestionCreate,
    database_url: str | None = None,
) -> EventSuggestionRecord:
    ensure_database_schema(database_url)
    validate_event_suggestion_create(suggestion)

    def add_in_transaction(session: Session) -> EventSuggestionRecord:
        validate_event_suggestion_queue_capacity(session, suggestion)
        model = add_event_suggestion_in_session(session, suggestion)
        return event_suggestion_to_record(model)

    return run_serialized_transaction(add_in_transaction, database_url=database_url)


def add_event_suggestion_in_session(
    session: Session,
    suggestion: EventSuggestionCreate,
) -> models.EventSuggestion:
    model = models.EventSuggestion(
        event_name=suggestion.event_name.strip(),
        url=optional_text(suggestion.url),
        event_date=suggestion.event_date,
        location=optional_text(suggestion.location),
        region_tags=list(suggestion.region_tags),
        distances=list(suggestion.distances),
        note=optional_text(suggestion.note),
        submitter_user_id=optional_text(suggestion.submitter_user_id),
        submitter_username=optional_text(suggestion.submitter_username),
        submitter_display_name=optional_text(suggestion.submitter_display_name),
    )
    session.add(model)
    session.flush()
    return model


def list_event_suggestions(
    *,
    status: str = "pending",
    limit: int = 10,
    database_url: str | None = None,
) -> tuple[EventSuggestionRecord, ...]:
    ensure_database_schema(database_url)
    validate_event_suggestion_status(status)
    with session_scope(database_url) as session:
        rows = session.scalars(
            select(models.EventSuggestion)
            .where(models.EventSuggestion.status == status)
            .order_by(models.EventSuggestion.created_at, models.EventSuggestion.id)
            .limit(limit)
        ).all()
        return tuple(event_suggestion_to_record(row) for row in rows)


def count_event_suggestions(
    *,
    status: str = "pending",
    database_url: str | None = None,
) -> int:
    ensure_database_schema(database_url)
    validate_event_suggestion_status(status)
    with session_scope(database_url) as session:
        return len(
            session.scalars(
                select(models.EventSuggestion.id).where(models.EventSuggestion.status == status)
            ).all()
        )


def get_event_suggestion(
    suggestion_id: int,
    *,
    status: str | None = None,
    database_url: str | None = None,
) -> EventSuggestionRecord | None:
    ensure_database_schema(database_url)
    if status is not None:
        validate_event_suggestion_status(status)

    with session_scope(database_url) as session:
        query = select(models.EventSuggestion).where(models.EventSuggestion.id == suggestion_id)
        if status is not None:
            query = query.where(models.EventSuggestion.status == status)

        row = session.scalar(query)
        return event_suggestion_to_record(row) if row is not None else None


def update_event_suggestion_status(
    suggestion_id: int,
    status: str,
    database_url: str | None = None,
) -> EventSuggestionRecord | None:
    ensure_database_schema(database_url)
    validate_event_suggestion_status(status)

    with session_scope(database_url) as session:
        model = session.get(models.EventSuggestion, suggestion_id)
        if model is None:
            return None

        model.status = status
        session.flush()
        return event_suggestion_to_record(model)


def archive_event(event_id: str, database_url: str | None = None) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    event = find_event(event_id, database_url)
    if event is None:
        return None

    with session_scope(database_url) as session:
        model = session.get(models.Event, event.id)
        if model is None or model.removed_at is not None:
            return None

        model.status = "removed"
        model.removed_at = utcnow()
        return event_to_domain(model)


def restore_event(event_id: str, database_url: str | None = None) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    archived = find_archived_event(event_id, database_url)
    if archived is None:
        return None

    with session_scope(database_url) as session:
        model = session.get(models.Event, archived.event.id)
        if model is None or model.removed_at is None:
            return None

        model.status = "monitoring"
        model.removed_at = None
        return event_to_domain(model)


def delete_event(event_id: str, database_url: str | None = None) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    event = find_event(event_id, database_url)
    if event is None:
        archived = find_archived_event(event_id, database_url)
        event = archived.event if archived is not None else None
    if event is None:
        return None

    with session_scope(database_url) as session:
        model = session.get(models.Event, event.id)
        if model is None:
            return None

        result = event_to_domain(model)
        model.current_edition_id = None
        for window in session.scalars(
            select(models.RegistrationWindow).where(
                models.RegistrationWindow.event_id == model.id
            )
        ).all():
            session.delete(window)
        for proposed_update in session.scalars(
            select(models.ProposedEventUpdate).where(
                models.ProposedEventUpdate.event_id == model.id
            )
        ).all():
            release_pending_proposal_key(session, proposed_update.id)
            session.delete(proposed_update)
        session.delete(model)
        return result


def proposed_update_key(update: ProposedEventUpdateCreate) -> str:
    return proposed_update_key_for_fields(
        event_id=update.event_id,
        update_type=update.update_type,
        current_fields=update.current_fields,
        proposed_fields=update.proposed_fields,
    )


def proposed_update_key_for_fields(
    *,
    event_id: str,
    update_type: str,
    current_fields: dict[str, Any],
    proposed_fields: dict[str, Any],
) -> str:
    payload = {
        "event_id": normalize_event_id(event_id),
        "update_type": update_type,
        "current_fields": current_fields,
        "proposed_fields": proposed_fields,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pending_proposal_for_key(
    session: Session,
    proposal_key: str,
) -> models.ProposedEventUpdate | None:
    key_model = session.get(models.PendingProposalKey, proposal_key)
    if key_model is None:
        return None
    update_model = session.get(models.ProposedEventUpdate, key_model.update_id)
    if update_model is not None and update_model.status in {"pending", "applying"}:
        return update_model
    session.delete(key_model)
    session.flush()
    return None


def create_proposed_event_update_in_session(
    session: Session,
    update: ProposedEventUpdateCreate,
) -> models.ProposedEventUpdate:
    event = session.get(models.Event, normalize_event_id(update.event_id))
    if event is None or event.removed_at is not None:
        raise EventWriteError(f"Event not found: {update.event_id}")

    proposal_key = proposed_update_key(update)
    existing = pending_proposal_for_key(session, proposal_key)
    if existing is not None:
        return existing

    # Existing databases can contain pending proposals created before the key table
    # was introduced. Adopt a matching legacy proposal instead of creating one
    # duplicate on the first scan after deployment.
    legacy_candidates = session.scalars(
        select(models.ProposedEventUpdate).where(
            models.ProposedEventUpdate.event_id == event.id,
            models.ProposedEventUpdate.update_type == update.update_type,
            models.ProposedEventUpdate.status == "pending",
        )
    ).all()
    for candidate in legacy_candidates:
        candidate_key = proposed_update_key_for_fields(
            event_id=candidate.event_id,
            update_type=candidate.update_type,
            current_fields=dict(candidate.current_fields or {}),
            proposed_fields=dict(candidate.proposed_fields or {}),
        )
        if candidate_key != proposal_key:
            continue
        session.add(
            models.PendingProposalKey(
                proposal_key=proposal_key,
                update_id=candidate.id,
            )
        )
        session.flush()
        return candidate

    model = models.ProposedEventUpdate(
        event_id=event.id,
        update_type=update.update_type,
        current_fields=update.current_fields,
        proposed_fields=update.proposed_fields,
        evidence=list(update.evidence),
        confidence=min(max(update.confidence, 0.0), 1.0),
        change_summary=optional_text(update.change_summary),
    )
    session.add(model)
    session.flush()
    session.add(
        models.PendingProposalKey(
            proposal_key=proposal_key,
            update_id=model.id,
        )
    )
    session.flush()
    return model


def release_pending_proposal_key(session: Session, update_id: int) -> None:
    session.execute(
        delete(models.PendingProposalKey).where(
            models.PendingProposalKey.update_id == update_id
        )
    )


def create_proposed_event_update(
    update: ProposedEventUpdateCreate,
    database_url: str | None = None,
) -> ProposedEventUpdateRecord:
    ensure_database_schema(database_url)
    try:
        with session_scope(database_url) as session:
            model = create_proposed_event_update_in_session(session, update)
            return proposed_event_update_to_record(model)
    except IntegrityError:
        with session_scope(database_url) as session:
            existing = pending_proposal_for_key(session, proposed_update_key(update))
            if existing is None:
                raise
            return proposed_event_update_to_record(existing)


def list_proposed_event_updates(
    *,
    event_id: str | None = None,
    status: str = "pending",
    limit: int = 10,
    database_url: str | None = None,
) -> tuple[ProposedEventUpdateRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        query = select(models.ProposedEventUpdate).where(
            models.ProposedEventUpdate.status == status
        )
        if event_id is not None:
            query = query.where(models.ProposedEventUpdate.event_id == normalize_event_id(event_id))
        rows = session.scalars(
            query.order_by(
                models.ProposedEventUpdate.created_at,
                models.ProposedEventUpdate.id,
            ).limit(limit)
        ).all()
        return tuple(proposed_event_update_to_record(row) for row in rows)


def count_proposed_event_updates(
    *,
    status: str = "pending",
    database_url: str | None = None,
) -> int:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        return len(
            session.scalars(
                select(models.ProposedEventUpdate.id).where(
                    models.ProposedEventUpdate.status == status
                )
            ).all()
        )


def get_proposed_event_update(
    update_id: int,
    *,
    status: str | None = None,
    database_url: str | None = None,
) -> ProposedEventUpdateRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        query = select(models.ProposedEventUpdate).where(models.ProposedEventUpdate.id == update_id)
        if status is not None:
            query = query.where(models.ProposedEventUpdate.status == status)
        row = session.scalar(query)
        return proposed_event_update_to_record(row) if row is not None else None


def approve_proposed_event_update(
    update_id: int,
    *,
    reviewer_user_id: str | None = None,
    database_url: str | None = None,
) -> ProposedEventUpdateApplyResult | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        update_model = claim_pending_proposal(session, update_id)
        if update_model is None:
            return None
        if update_model.update_type != "registration_window":
            raise EventWriteError(
                f"Unsupported proposed update type: {update_model.update_type}"
            )

        event_model = session.get(models.Event, update_model.event_id)
        if event_model is None or event_model.removed_at is not None:
            raise EventWriteError(f"Event not found: {update_model.event_id}")
        ensure_proposal_is_current(update_model, event_model)
        event = apply_registration_window_selected_fields_in_session(
            session,
            event_model,
            dict(update_model.proposed_fields or {}),
        )

        now = utcnow()
        update_model.status = "applied"
        update_model.reviewed_by_user_id = optional_text(reviewer_user_id)
        update_model.reviewed_at = now
        update_model.applied_at = now
        release_pending_proposal_key(session, update_model.id)
        session.flush()
        return ProposedEventUpdateApplyResult(
            update=proposed_event_update_to_record(update_model),
            event=event,
        )


def partial_apply_proposed_event_update(
    update_id: int,
    *,
    selected_fields: tuple[str, ...],
    reviewer_user_id: str | None = None,
    database_url: str | None = None,
) -> ProposedEventUpdatePartialApplyResult | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        update_model = claim_pending_proposal(session, update_id)
        if update_model is None:
            return None
        if update_model.update_type != "registration_window":
            raise EventWriteError(
                f"Unsupported proposed update type: {update_model.update_type}"
            )

        update = proposed_event_update_to_record(update_model)
        changed_fields = proposed_update_changed_fields(update)
        selected = tuple(field for field in selected_fields if field in changed_fields)
        if not selected:
            raise EventWriteError("Select at least one changed field to apply.")

        selected_set = set(selected)
        remaining = tuple(field for field in changed_fields if field not in selected_set)
        selected_proposed_fields = {
            field: update.proposed_fields[field]
            for field in selected
            if field in update.proposed_fields
        }

        event_model = session.get(models.Event, update.event_id)
        if event_model is None or event_model.removed_at is not None:
            raise EventWriteError(f"Event not found: {update.event_id}")
        ensure_proposal_is_current(update_model, event_model)
        event = apply_registration_window_selected_fields_in_session(
            session,
            event_model,
            selected_proposed_fields,
        )

        now = utcnow()
        update_model.status = "applied" if not remaining else "applied_partial"
        update_model.reviewed_by_user_id = optional_text(reviewer_user_id)
        update_model.reviewed_at = now
        update_model.applied_at = now
        release_pending_proposal_key(session, update_model.id)

        follow_up_model = None
        if remaining:
            live_fields = registration_update_fields(event)
            follow_up_model = create_proposed_event_update_in_session(
                session,
                ProposedEventUpdateCreate(
                    event_id=update.event_id,
                    update_type=update.update_type,
                    current_fields={
                        field: live_fields.get(field)
                        for field in remaining
                    },
                    proposed_fields={
                        field: update.proposed_fields[field]
                        for field in remaining
                        if field in update.proposed_fields
                    },
                    evidence=(
                        *update.evidence,
                        f"Created from partial apply of update #{update.id}.",
                    ),
                    confidence=update.confidence,
                    change_summary=update.change_summary,
                ),
            )

        session.flush()
        return ProposedEventUpdatePartialApplyResult(
            update=proposed_event_update_to_record(update_model),
            event=event,
            follow_up_update=(
                proposed_event_update_to_record(follow_up_model)
                if follow_up_model is not None
                else None
            ),
            applied_fields=selected,
            remaining_fields=remaining,
        )


def reject_proposed_event_update(
    update_id: int,
    *,
    reviewer_user_id: str | None = None,
    database_url: str | None = None,
) -> ProposedEventUpdateRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        model = claim_pending_proposal(session, update_id)
        if model is None:
            return None

        model.status = "rejected"
        model.reviewed_by_user_id = optional_text(reviewer_user_id)
        model.reviewed_at = utcnow()
        release_pending_proposal_key(session, model.id)
        session.flush()
        return proposed_event_update_to_record(model)


def claim_pending_proposal(
    session: Session,
    update_id: int,
) -> models.ProposedEventUpdate | None:
    claimed = session.execute(
        sqlalchemy_update(models.ProposedEventUpdate)
        .where(
            models.ProposedEventUpdate.id == update_id,
            models.ProposedEventUpdate.status == "pending",
        )
        .values(status="applying")
    )
    if claimed.rowcount != 1:
        return None
    return session.get(models.ProposedEventUpdate, update_id)


def apply_registration_window_update(
    event_id: str,
    update: RegistrationWindowApply,
    database_url: str | None = None,
) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    normalized_id = normalize_event_id(event_id)
    with session_scope(database_url) as session:
        model = session.get(models.Event, normalized_id)
        if model is None or model.removed_at is not None:
            return None

        model.registration_status = update.registration_status
        if update.registration_url is not None:
            model.registration_url = update.registration_url
        if update.event_date is not None:
            model.next_event_date = update.event_date

        current_edition = ensure_current_edition(session, model, update.event_date)
        window = current_registration_window(session, model.id, current_edition)
        window.registration_open_at = update.registration_open_at
        window.registration_open_precision = update.registration_open_precision
        window.registration_close_at = update.registration_close_at
        window.status = update.registration_status
        window.approved_at = utcnow()

        return event_to_domain(model)


def apply_registration_window_selected_fields(
    event_id: str,
    proposed_fields: dict[str, Any],
    database_url: str | None = None,
) -> TrackedEvent | None:
    ensure_database_schema(database_url)
    normalized_id = normalize_event_id(event_id)
    with session_scope(database_url) as session:
        model = session.get(models.Event, normalized_id)
        if model is None or model.removed_at is not None:
            return None

        return apply_registration_window_selected_fields_in_session(
            session,
            model,
            proposed_fields,
        )


def apply_registration_window_selected_fields_in_session(
    session: Session,
    model: models.Event,
    proposed_fields: dict[str, Any],
) -> TrackedEvent:
    registration_window = None
    previous_edition_id = model.current_edition_id
    if "registration_status" in proposed_fields:
        model.registration_status = field_text(
            proposed_fields.get("registration_status"),
            fallback=model.registration_status,
        ) or "unknown"
    if "registration_url" in proposed_fields:
        model.registration_url = field_text(proposed_fields.get("registration_url"))
    if "event_date" in proposed_fields:
        model.next_event_date = field_text(proposed_fields.get("event_date"))

    needs_window = any(
        field in proposed_fields
        for field in (
            "registration_status",
            "registration_open_at",
            "registration_open_precision",
            "registration_close_at",
            "event_date",
        )
    )
    if needs_window:
        current_edition = ensure_current_edition(session, model, model.next_event_date)
        registration_window = current_registration_window(session, model.id, current_edition)
        current_edition_id = current_edition.id if current_edition is not None else None
        if (
            current_edition_id != previous_edition_id
            and "registration_status" not in proposed_fields
        ):
            model.registration_status = registration_window.status
        if "registration_open_at" in proposed_fields:
            registration_window.registration_open_at = field_text(
                proposed_fields.get("registration_open_at")
            )
        if "registration_open_precision" in proposed_fields:
            registration_window.registration_open_precision = (
                field_text(
                    proposed_fields.get("registration_open_precision"),
                    fallback=registration_window.registration_open_precision,
                )
                or "unknown"
            )
        if "registration_close_at" in proposed_fields:
            registration_window.registration_close_at = field_text(
                proposed_fields.get("registration_close_at")
            )
        if "registration_status" in proposed_fields:
            registration_window.status = model.registration_status
        registration_window.approved_at = utcnow()

    session.flush()
    return event_to_domain(model, registration_window=registration_window)


def ensure_proposal_is_current(
    update: models.ProposedEventUpdate,
    event: models.Event,
) -> None:
    live_event = event_to_domain(event)
    live_fields = registration_update_fields(live_event)
    stale_fields = tuple(
        field
        for field, expected_value in dict(update.current_fields or {}).items()
        if live_fields.get(field) != expected_value
    )
    if stale_fields:
        raise EventWriteError(
            "Proposed update is stale; live event fields changed: "
            + ", ".join(stale_fields)
        )


def registration_update_fields(event: TrackedEvent) -> dict[str, str | None]:
    return {
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at,
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at,
        "registration_url": event.registration_url,
        "event_date": event.event_date,
    }


def registration_window_apply_from_proposed_fields(
    current_fields: dict[str, Any],
    proposed_fields: dict[str, Any],
) -> RegistrationWindowApply:
    return RegistrationWindowApply(
        registration_status=field_text(
            proposed_fields.get("registration_status"),
            fallback=field_text(current_fields.get("registration_status")) or "unknown",
        )
        or "unknown",
        registration_open_at=field_text(proposed_fields.get("registration_open_at")),
        registration_open_precision=field_text(
            proposed_fields.get("registration_open_precision"),
            fallback=field_text(current_fields.get("registration_open_precision")) or "unknown",
        )
        or "unknown",
        registration_close_at=field_text(proposed_fields.get("registration_close_at")),
        registration_url=field_text(proposed_fields.get("registration_url")),
        event_date=field_text(proposed_fields.get("event_date")),
    )


def field_text(value: Any, *, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    if not isinstance(value, str):
        return str(value)

    stripped = value.strip()
    return stripped if stripped and stripped != "unknown" else fallback


def proposed_update_changed_fields(update: ProposedEventUpdateRecord) -> tuple[str, ...]:
    return tuple(
        field
        for field, proposed_value in update.proposed_fields.items()
        if proposed_value is not None
        and proposed_value != ""
        and proposed_value != "unknown"
        and update.current_fields.get(field) != proposed_value
    )


def validate_event_suggestion_create(suggestion: EventSuggestionCreate) -> None:
    if not suggestion.event_name.strip():
        raise EventWriteError("Event suggestion requires an event name.")
    if not suggestion.distances:
        raise EventWriteError("Event suggestion requires at least one distance.")


def validate_event_suggestion_status(status: str) -> None:
    if status not in EVENT_SUGGESTION_STATUSES:
        raise EventWriteError(f"Unknown event suggestion status: {status}")


def validate_event_suggestion_queue_capacity(
    session: Session,
    suggestion: EventSuggestionCreate,
) -> None:
    pending_total = len(
        session.scalars(
            select(models.EventSuggestion.id).where(models.EventSuggestion.status == "pending")
        ).all()
    )
    if pending_total >= EVENT_SUGGESTION_MAX_PENDING_TOTAL:
        raise EventWriteError(
            "Suggestion queue is full. Please try again after a moderator handles "
            f"some pending suggestions. Current limit: {EVENT_SUGGESTION_MAX_PENDING_TOTAL}."
        )

    if suggestion.submitter_is_moderator:
        return

    submitter_user_id = optional_text(suggestion.submitter_user_id)
    if submitter_user_id is None:
        return

    pending_for_user = len(
        session.scalars(
            select(models.EventSuggestion.id).where(
                models.EventSuggestion.status == "pending",
                models.EventSuggestion.submitter_user_id == submitter_user_id,
            )
        ).all()
    )
    if pending_for_user >= EVENT_SUGGESTION_MAX_PENDING_PER_USER:
        raise EventWriteError(
            "You already have 3 pending suggestions. Please wait until a moderator "
            "handles one before submitting another."
        )


def event_suggestion_to_record(model: models.EventSuggestion) -> EventSuggestionRecord:
    return EventSuggestionRecord(
        id=model.id,
        status=model.status,
        event_name=model.event_name,
        url=model.url,
        event_date=model.event_date,
        location=model.location,
        region_tags=tuple(model.region_tags or ()),
        distances=tuple(model.distances or ()),
        note=model.note,
        submitter_user_id=model.submitter_user_id,
        submitter_username=model.submitter_username,
        submitter_display_name=model.submitter_display_name,
    )


def apply_event_fields(
    model: models.Event,
    event: EventCreate,
    normalized_public_id: str,
) -> None:
    model.canonical_name = event.name.strip()
    model.city = event.city.strip()
    model.country = event.country.strip()
    model.timezone = event.timezone.strip()
    model.next_event_date = event.event_date
    model.distances = list(event.distances)
    model.registration_status = event.registration_status
    model.official_url = event.official_url.strip()
    model.registration_url = optional_text(event.registration_url)
    replace_children(
        model.search_keywords,
        [
            models.EventSearchKeyword(keyword=keyword, keyword_type="moderator")
            for keyword in event_keywords(event, normalized_public_id)
        ],
    )
    replace_children(
        model.regions,
        [
            models.EventRegion(region_tag=region_tag, is_primary=index == 0)
            for index, region_tag in enumerate(event.regions)
        ],
    )
    replace_children(model.sources, event_sources(event))
    replace_children(model.editions, event_editions(event))


def apply_event_registration_fields(
    session: Session,
    model: models.Event,
    event: EventCreate,
) -> models.RegistrationWindow | None:
    model.registration_status = event.registration_status
    if not has_registration_window_data(event):
        return None

    current_edition = ensure_current_edition(session, model, event.event_date)
    if current_edition is not None:
        model.current_edition_id = current_edition.id
    window = current_registration_window(session, model.id, current_edition)
    window.registration_open_at = event.registration_open_at
    window.registration_open_precision = event.registration_open_precision
    window.registration_close_at = event.registration_close_at
    window.status = event.registration_status
    window.approved_at = utcnow()
    return window


def has_registration_window_data(event: EventCreate) -> bool:
    return (
        event.registration_status != "unknown"
        or event.registration_open_at is not None
        or event.registration_open_precision != "unknown"
        or event.registration_close_at is not None
    )


def clear_registration_windows(session: Session, event_id: str) -> None:
    for window in session.scalars(
        select(models.RegistrationWindow).where(models.RegistrationWindow.event_id == event_id)
    ):
        session.delete(window)


def base_event_query() -> Select[tuple[models.Event]]:
    return (
        select(models.Event)
        .where(models.Event.removed_at.is_(None))
        .options(*event_query_options())
    )


def archived_event_query() -> Select[tuple[models.Event]]:
    return (
        select(models.Event)
        .where(models.Event.removed_at.is_not(None))
        .options(*event_query_options())
    )


def event_query_options():
    return (
        selectinload(models.Event.legacy_ids),
        selectinload(models.Event.search_keywords),
        selectinload(models.Event.regions).selectinload(models.EventRegion.region),
        selectinload(models.Event.collections).selectinload(
            models.EventCollectionMember.collection
        ),
        selectinload(models.Event.sources),
        selectinload(models.Event.editions),
        selectinload(models.Event.registration_windows),
    )


def archived_event_to_record(event: models.Event) -> ArchivedEventRecord:
    return ArchivedEventRecord(
        event=event_to_domain(event),
        removed_at=event.removed_at.isoformat(timespec="seconds")
        if event.removed_at is not None
        else None,
    )


def event_to_domain(
    event: models.Event,
    *,
    registration_window: models.RegistrationWindow | None = None,
) -> TrackedEvent:
    regions = tuple(
        region.region_tag for region in sorted(event.regions, key=lambda item: item.region_tag)
    )
    collections = tuple(
        collection.collection_slug
        for collection in sorted(event.collections, key=lambda item: item.sort_order)
    )
    legacy_ids = tuple(legacy_id.legacy_id for legacy_id in event.legacy_ids)
    search_keywords = tuple(keyword.keyword for keyword in event.search_keywords)
    window = registration_window or current_domain_registration_window(event)

    return TrackedEvent(
        id=event.id,
        public_id=event.public_id,
        legacy_ids=legacy_ids,
        search_keywords=search_keywords,
        name=event.canonical_name,
        city=event.city,
        country=event.country,
        timezone=event.timezone,
        distances=tuple(event.distances or ()),
        regions=regions,
        collections=collections,
        event_date=event.next_event_date,
        registration_status=event.registration_status,
        official_url=event.official_url,
        registration_url=event.registration_url,
        registration_open_at=window.registration_open_at if window is not None else None,
        registration_open_precision=(
            window.registration_open_precision if window is not None else "unknown"
        ),
        registration_close_at=window.registration_close_at if window is not None else None,
    )


def current_domain_registration_window(
    event: models.Event,
) -> models.RegistrationWindow | None:
    windows = list(event.registration_windows or ())
    if not windows:
        return None

    if event.current_edition_id is not None:
        edition_windows = [
            window
            for window in windows
            if window.event_edition_id == event.current_edition_id
        ]
        if edition_windows:
            return max(edition_windows, key=lambda window: window.id)

    no_edition_windows = [window for window in windows if window.event_edition_id is None]
    if no_edition_windows:
        return max(no_edition_windows, key=lambda window: window.id)

    return max(windows, key=lambda window: window.id)


def proposed_event_update_to_record(
    update: models.ProposedEventUpdate,
) -> ProposedEventUpdateRecord:
    return ProposedEventUpdateRecord(
        id=update.id,
        event_id=update.event_id,
        update_type=update.update_type,
        current_fields=dict(update.current_fields or {}),
        proposed_fields=dict(update.proposed_fields or {}),
        evidence=tuple(update.evidence or ()),
        confidence=update.confidence,
        status=update.status,
        change_summary=update.change_summary,
    )


def ensure_current_edition(
    session: Session,
    event: models.Event,
    event_date: str | None,
) -> models.EventEdition | None:
    edition_label = event_date[:4] if event_date is not None else None
    if event.current_edition_id is not None:
        edition = session.get(models.EventEdition, event.current_edition_id)
        if edition is not None:
            if event_date is None:
                return edition

            if edition.edition_label == edition_label:
                edition.event_date = event_date
                edition.edition_year = int(edition_label)
                return edition

            edition.is_current = False
            next_edition = session.scalar(
                select(models.EventEdition).where(
                    models.EventEdition.event_id == event.id,
                    models.EventEdition.edition_label == edition_label,
                )
            )
            if next_edition is not None:
                next_edition.event_date = event_date
                next_edition.edition_year = int(edition_label)
                next_edition.status = "date_announced"
                next_edition.is_current = True
                event.current_edition_id = next_edition.id
                return next_edition

    if event_date is None:
        return None


    existing_edition = session.scalar(
        select(models.EventEdition).where(
            models.EventEdition.event_id == event.id,
            models.EventEdition.edition_label == edition_label,
        )
    )
    if existing_edition is not None:
        existing_edition.event_date = event_date
        existing_edition.edition_year = int(edition_label)
        existing_edition.status = "date_announced"
        existing_edition.is_current = True
        event.current_edition_id = existing_edition.id
        return existing_edition

    edition = models.EventEdition(
        event_id=event.id,
        edition_year=int(edition_label),
        edition_label=edition_label,
        event_date=event_date,
        status="date_announced",
        is_current=True,
    )
    session.add(edition)
    session.flush()
    event.current_edition_id = edition.id
    return edition


def current_registration_window(
    session: Session,
    event_id: str,
    edition: models.EventEdition | None,
) -> models.RegistrationWindow:
    query = select(models.RegistrationWindow).where(models.RegistrationWindow.event_id == event_id)
    if edition is not None:
        query = query.where(models.RegistrationWindow.event_edition_id == edition.id)
    window = session.scalar(query.order_by(models.RegistrationWindow.id.desc()).limit(1))
    if window is not None:
        return window

    window = models.RegistrationWindow(
        event_id=event_id,
        event_edition_id=edition.id if edition is not None else None,
    )
    session.add(window)
    session.flush()
    return window


def count_events(session: Session) -> int:
    return len(session.scalars(select(models.Event.id)).all())


def normalize_public_id(public_id: str) -> str:
    normalized = normalize_event_id(public_id)
    distance_code = normalized.rsplit(".", maxsplit=1)[-1]
    if distance_code not in DISTANCE_CODE_TO_KEY:
        raise EventWriteError(
            "Public ID must use the <place>.<distance> format with a supported distance code, "
            "for example zurich.42"
        )

    return normalized


def validate_event_create(event: EventCreate, normalized_public_id: str) -> None:
    required_fields = {
        "name": event.name,
        "city": event.city,
        "country": event.country,
        "timezone": event.timezone,
        "official URL": event.official_url,
    }
    missing = [label for label, value in required_fields.items() if not value.strip()]
    if missing:
        raise EventWriteError(f"Missing required field: {', '.join(missing)}")
    if not event.distances:
        raise EventWriteError("At least one distance is required.")
    if not event.regions:
        raise EventWriteError("At least one region tag is required.")
    if event.registration_status not in REGISTRATION_STATUSES:
        raise EventWriteError(f"Unknown registration status: {event.registration_status}")
    if event.registration_open_precision not in REGISTRATION_OPEN_PRECISIONS:
        raise EventWriteError(
            f"Unknown registration open precision: {event.registration_open_precision}"
        )
    distance_code = normalized_public_id.rsplit(".", maxsplit=1)[-1]
    expected_distance = DISTANCE_CODE_TO_KEY.get(distance_code)
    if expected_distance is not None and expected_distance not in event.distances:
        raise EventWriteError(
            f"Public ID ending in .{distance_code} must include {expected_distance} distance."
        )


def ensure_region(session: Session, region_tag: str) -> None:
    if session.get(models.Region, region_tag) is not None:
        return

    session.add(
        models.Region(
            tag=region_tag,
            name=REGION_LABELS.get(region_tag, region_tag.upper()),
            scope="country" if len(region_tag) == 2 else "custom",
        )
    )


def optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def normalize_url(value: str | None) -> str:
    if not value:
        return ""

    parsed = urlparse(value.strip())
    scheme = parsed.scheme.casefold()
    netloc = parsed.netloc.casefold()
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((scheme, netloc, path, "", parsed.query, ""))


def event_keywords(event: EventCreate, public_id: str) -> tuple[str, ...]:
    keywords = {
        public_id,
        public_id.replace(".", " "),
        event.name.strip(),
        event.city.strip(),
        event.country.strip(),
        *event.regions,
        *event.distances,
    }
    if "marathon" in event.distances:
        keywords.add("marathon")
        keywords.add("42")
    if "half_marathon" in event.distances:
        keywords.add("half marathon")
        keywords.add("21")

    return tuple(sorted(keyword for keyword in keywords if keyword))


def event_sources(event: EventCreate) -> list[models.EventSource]:
    sources = [
        models.EventSource(url=event.official_url.strip(), source_type="official_site", priority=10)
    ]
    registration_url = optional_text(event.registration_url)
    if registration_url and registration_url != event.official_url.strip():
        sources.append(
            models.EventSource(
                url=registration_url,
                source_type="registration_page",
                priority=20,
            )
        )

    return sources


def event_editions(event: EventCreate) -> list[models.EventEdition]:
    if not event.event_date:
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


def replace_children(target: list[object], replacement: list[object]) -> None:
    target.clear()
    target.extend(replacement)


def clear_event_children(event: models.Event) -> None:
    event.current_edition_id = None
    replace_children(event.legacy_ids, [])
    replace_children(event.search_keywords, [])
    replace_children(event.regions, [])
    replace_children(event.collections, [])
    replace_children(event.sources, [])
    replace_children(event.editions, [])
