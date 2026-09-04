from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from itertools import chain
from typing import Literal

from sqlalchemy import Text, cast, func, or_, select, update
from sqlalchemy.orm import Session

from run4221.db import models
from run4221.db.bootstrap import ensure_database_schema
from run4221.db.repository import (
    EventSuggestionCreate,
    EventSuggestionRecord,
    EventWriteError,
    ProposedEventUpdateCreate,
    ProposedEventUpdateRecord,
    add_event_suggestion_in_session,
    base_event_query,
    create_proposed_event_update_in_session,
    event_suggestion_to_record,
    event_to_domain,
    normalize_url,
    proposed_event_update_to_record,
    validate_event_suggestion_create,
)
from run4221.db.session import run_serialized_transaction, session_scope
from run4221.events import TrackedEvent, normalize_event_id
from run4221.researcher.schemas import RESEARCHER_MAX_PENDING_SUGGESTIONS

RESEARCHER_DEFAULT_MAX_PENDING_UPDATES = 50
_ACTIVE_PROPOSAL_STATUSES = ("pending", "applying")

SuggestionAdmissionOutcome = Literal["admitted", "duplicate", "queue_full"]
ProposalAdmissionOutcome = Literal["admitted", "conflicting_pending", "queue_full"]
ResearchQueueKind = Literal["suggest_event", "propose_update"]


@dataclass(frozen=True)
class ResearchSourceRecord:
    source_id: int
    event: TrackedEvent
    url: str


@dataclass(frozen=True)
class SuggestionAdmission:
    outcome: SuggestionAdmissionOutcome
    suggestion: EventSuggestionRecord | None = None


@dataclass(frozen=True)
class ProposalAdmission:
    outcome: ProposalAdmissionOutcome
    update: ProposedEventUpdateRecord | None = None
    conflicting_update_id: int | None = None


def list_due_sources(
    *,
    due_before: datetime,
    limit: int,
    database_url: str | None = None,
) -> tuple[ResearchSourceRecord, ...]:
    """Return bounded active sources without exposing general event mutation."""

    if limit < 1:
        return ()

    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        sources = session.scalars(
            select(models.EventSource)
            .join(models.Event, models.Event.id == models.EventSource.event_id)
            .where(
                models.EventSource.active.is_(True),
                models.Event.removed_at.is_(None),
                models.Event.status == "monitoring",
                or_(
                    models.EventSource.last_checked_at.is_(None),
                    models.EventSource.last_checked_at <= due_before,
                ),
            )
            .order_by(
                models.EventSource.last_checked_at.is_not(None),
                models.EventSource.last_checked_at,
                models.EventSource.priority,
                models.EventSource.id,
            )
            .limit(limit)
        ).all()
        event_ids = {source.event_id for source in sources}
        events = session.scalars(
            base_event_query().where(models.Event.id.in_(event_ids))
        ).all()
        events_by_id = {event.id: event_to_domain(event) for event in events}
        return tuple(
            ResearchSourceRecord(
                source_id=source.id,
                event=events_by_id[source.event_id],
                url=source.url,
            )
            for source in sources
        )


def get_research_source(
    event_id: str,
    *,
    database_url: str | None = None,
) -> ResearchSourceRecord | None:
    """Return the highest-priority active source for an operator one-shot."""

    normalized_event_id = normalize_event_id(event_id)
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        source = session.scalar(
            select(models.EventSource)
            .join(models.Event, models.Event.id == models.EventSource.event_id)
            .where(
                models.EventSource.event_id == normalized_event_id,
                models.EventSource.active.is_(True),
                models.Event.removed_at.is_(None),
                models.Event.status == "monitoring",
            )
            .order_by(models.EventSource.priority, models.EventSource.id)
            .limit(1)
        )
        if source is None:
            return None
        event = session.scalar(base_event_query().where(models.Event.id == source.event_id))
        assert event is not None
        return ResearchSourceRecord(
            source_id=source.id,
            event=event_to_domain(event),
            url=source.url,
        )


def get_refresh_source(
    event_id: str,
    *,
    database_url: str | None = None,
) -> ResearchSourceRecord | None:
    """Return the active registration_page source when present, else top priority.

    A one-shot refresh watches where registration actually happens; the
    registration page outranks the official site whenever both are active.
    """

    normalized_event_id = normalize_event_id(event_id)
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        source = session.scalar(
            select(models.EventSource)
            .join(models.Event, models.Event.id == models.EventSource.event_id)
            .where(
                models.EventSource.event_id == normalized_event_id,
                models.EventSource.active.is_(True),
                models.Event.removed_at.is_(None),
                models.Event.status == "monitoring",
            )
            .order_by(
                models.EventSource.source_type != "registration_page",
                models.EventSource.priority,
                models.EventSource.id,
            )
            .limit(1)
        )
        if source is None:
            return None
        event = session.scalar(base_event_query().where(models.Event.id == source.event_id))
        assert event is not None
        return ResearchSourceRecord(
            source_id=source.id,
            event=event_to_domain(event),
            url=source.url,
        )


def find_research_queue_reference(
    queue_kind: ResearchQueueKind,
    *,
    decision_marker: str,
    database_url: str | None = None,
) -> str | None:
    """Resolve one exact prepared-decision marker after an interrupted finalization."""

    clean_marker = decision_marker.strip()
    if not clean_marker:
        raise ValueError("Research queue marker cannot be empty.")
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        if queue_kind == "suggest_event":
            ids = session.scalars(
                select(models.EventSuggestion.id).where(
                    models.EventSuggestion.note.contains(clean_marker)
                )
            ).all()
            prefix = "event_suggestion"
        elif queue_kind == "propose_update":
            ids = session.scalars(
                select(models.ProposedEventUpdate.id).where(
                    cast(models.ProposedEventUpdate.evidence, Text).contains(clean_marker)
                )
            ).all()
            prefix = "proposed_event_update"
        else:
            raise ValueError(f"Unsupported research queue kind: {queue_kind}")
    if len(ids) > 1:
        raise EventWriteError("Prepared decision matched more than one queue record.")
    return f"{prefix}:{ids[0]}" if ids else None


def mark_source_checked(
    source_id: int,
    *,
    checked_at: datetime,
    database_url: str | None = None,
) -> bool:
    ensure_database_schema(database_url)

    def mark(session: Session) -> bool:
        result = session.execute(
            update(models.EventSource)
            .where(models.EventSource.id == source_id)
            .values(last_checked_at=checked_at)
        )
        return result.rowcount == 1

    return run_serialized_transaction(mark, database_url=database_url)


def admit_suggestion(
    suggestion: EventSuggestionCreate,
    *,
    max_pending: int = RESEARCHER_MAX_PENDING_SUGGESTIONS,
    database_url: str | None = None,
) -> SuggestionAdmission:
    """Admit one system-authored discovery after duplicate and reserve checks."""

    _validate_suggestion_limit(max_pending)
    system_suggestion = replace(
        suggestion,
        submitter_user_id=None,
        submitter_username=None,
        submitter_display_name=None,
        submitter_is_moderator=False,
    )
    validate_event_suggestion_create(system_suggestion)
    candidate_url = normalize_url(system_suggestion.url)
    if not candidate_url:
        raise EventWriteError("Research suggestions require a candidate URL.")
    ensure_database_schema(database_url)

    def admit(session: Session) -> SuggestionAdmission:
        if _known_candidate_url(session, candidate_url):
            return SuggestionAdmission(outcome="duplicate")

        pending_count = session.scalar(
            select(func.count(models.EventSuggestion.id)).where(
                models.EventSuggestion.status == "pending"
            )
        )
        if (pending_count or 0) >= max_pending:
            return SuggestionAdmission(outcome="queue_full")

        model = add_event_suggestion_in_session(session, system_suggestion)
        return SuggestionAdmission(
            outcome="admitted",
            suggestion=event_suggestion_to_record(model),
        )

    return run_serialized_transaction(admit, database_url=database_url)


def admit_proposed_update(
    proposed_update: ProposedEventUpdateCreate,
    *,
    max_pending: int = RESEARCHER_DEFAULT_MAX_PENDING_UPDATES,
    database_url: str | None = None,
) -> ProposalAdmission:
    """Admit one queue-only event update, never a direct event mutation."""

    if max_pending < 0:
        raise ValueError("max_pending must be non-negative")
    ensure_database_schema(database_url)

    def admit(session: Session) -> ProposalAdmission:
        event_id = normalize_event_id(proposed_update.event_id)
        event = session.get(models.Event, event_id)
        if event is None or event.removed_at is not None:
            raise EventWriteError(f"Event not found: {proposed_update.event_id}")

        conflict = session.scalar(
            select(models.ProposedEventUpdate.id).where(
                models.ProposedEventUpdate.event_id == event_id,
                models.ProposedEventUpdate.update_type == proposed_update.update_type,
                models.ProposedEventUpdate.status.in_(_ACTIVE_PROPOSAL_STATUSES),
            )
        )
        if conflict is not None:
            return ProposalAdmission(
                outcome="conflicting_pending",
                conflicting_update_id=conflict,
            )

        pending_count = session.scalar(
            select(func.count(models.ProposedEventUpdate.id)).where(
                models.ProposedEventUpdate.status.in_(_ACTIVE_PROPOSAL_STATUSES)
            )
        )
        if (pending_count or 0) >= max_pending:
            return ProposalAdmission(outcome="queue_full")

        model = create_proposed_event_update_in_session(session, proposed_update)
        return ProposalAdmission(
            outcome="admitted",
            update=proposed_event_update_to_record(model),
        )

    return run_serialized_transaction(admit, database_url=database_url)


def _validate_suggestion_limit(max_pending: int) -> None:
    if not 0 <= max_pending <= RESEARCHER_MAX_PENDING_SUGGESTIONS:
        raise ValueError(
            "max_pending must preserve the researcher queue reserve "
            f"(0..{RESEARCHER_MAX_PENDING_SUGGESTIONS})"
        )


def _known_candidate_url(session: Session, candidate_url: str) -> bool:
    event_urls = chain.from_iterable(
        session.execute(select(models.Event.official_url, models.Event.registration_url))
    )
    known_urls = chain(
        event_urls,
        session.scalars(select(models.EventSource.url)),
        session.scalars(select(models.EventSuggestion.url)),
    )
    return any(normalize_url(url) == candidate_url for url in known_urls)
