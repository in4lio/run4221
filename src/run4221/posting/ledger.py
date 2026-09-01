from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, not_, select, update
from sqlalchemy.orm import Session, selectinload

from run4221.config import get_telegram_channel_settings
from run4221.db import models
from run4221.db.bootstrap import ensure_database_schema
from run4221.db.session import run_serialized_transaction, session_scope
from run4221.events import TrackedEvent, normalize_event_id
from run4221.posting.templates import (
    MESSAGE_HEADINGS,
    format_public_value,
    render_channel_message,
)

CHANNEL_MESSAGE_TYPES = frozenset(MESSAGE_HEADINGS)
SCHEDULED_MESSAGE_TYPES = (
    "opens_tomorrow",
    "opens_today",
    "closes_tomorrow",
    "registration_closed",
)
REGISTRATION_NEWS_TYPES = frozenset(
    {
        "registration_date_discovered",
        "registration_open",
        "registration_updated",
        "registration_closed",
        "sold_out",
        "waitlist",
        "correction",
    }
)
REPEATABLE_NEWS_TYPES = frozenset({"registration_updated", "correction"})
CANCELLABLE_STATUSES = frozenset(
    {"pending_review", "pending", "scheduled", "failed", "reconciled_absent"}
)
REFRESHABLE_DRAFT_STATUSES = frozenset(
    {"scheduled", "cancelled", "pending", "pending_review", "failed", "reconciled_absent"}
)
ACTIONABLE_STATUSES = ("pending_review", "failed", "ambiguous")

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChannelMessageRecord:
    id: int
    event_id: str
    registration_window_id: int | None
    message_type: str
    target_chat_id: str
    idempotency_key: str
    status: str
    text: str
    source_url: str
    scheduled_for: datetime | None
    approved_by_user_id: str | None
    attempt_count: int
    next_attempt_at: datetime | None
    telegram_message_id: int | None
    failure_reason: str | None
    published_at: datetime | None


@dataclass(frozen=True)
class ChannelMessageReconciliationRecord:
    id: int
    channel_message_id: int
    decision: str
    reviewer_user_id: str
    created_at: datetime


def target_channel_id() -> str:
    configured = get_telegram_channel_settings().telegram_channel_id
    return configured.strip() or "@run4221"


def channel_message_key(
    event_id: str,
    registration_window_id: int | None,
    message_type: str,
    target_chat_id: str,
    *,
    occurrence: int = 0,
) -> str:
    value = f"{event_id}|{registration_window_id or ''}|{message_type}|{target_chat_id}"
    if occurrence:
        value = f"{value}|occurrence:{occurrence}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def list_channel_messages(
    *,
    event_id: str | None = None,
    status: str | None = None,
    limit: int = 100,
    database_url: str | None = None,
) -> tuple[ChannelMessageRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        query = select(models.ChannelMessage)
        if event_id is not None:
            query = query.where(models.ChannelMessage.event_id == event_id)
        if status is not None:
            query = query.where(models.ChannelMessage.status == status)
        rows = session.scalars(
            query.order_by(models.ChannelMessage.created_at, models.ChannelMessage.id).limit(limit)
        ).all()
        return tuple(channel_message_to_record(row) for row in rows)


def count_channel_messages(
    *, status: str, database_url: str | None = None
) -> int:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        return int(
            session.scalar(
                select(func.count(models.ChannelMessage.id)).where(
                    models.ChannelMessage.status == status
                )
            )
            or 0
        )


def list_actionable_channel_messages(
    *,
    limit: int = 100,
    database_url: str | None = None,
) -> tuple[ChannelMessageRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        rows = session.scalars(
            select(models.ChannelMessage)
            .where(models.ChannelMessage.status.in_(ACTIONABLE_STATUSES))
            .order_by(models.ChannelMessage.created_at, models.ChannelMessage.id)
            .limit(limit)
        ).all()
        return tuple(channel_message_to_record(row) for row in rows)


def count_actionable_channel_messages(*, database_url: str | None = None) -> int:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        return int(
            session.scalar(
                select(func.count(models.ChannelMessage.id)).where(
                    models.ChannelMessage.status.in_(ACTIONABLE_STATUSES)
                )
            )
            or 0
        )


def get_channel_message(
    message_id: int, *, database_url: str | None = None
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        row = session.get(models.ChannelMessage, message_id)
        return channel_message_to_record(row) if row is not None else None


def list_channel_message_reconciliations(
    message_id: int,
    *,
    database_url: str | None = None,
) -> tuple[ChannelMessageReconciliationRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        rows = session.scalars(
            select(models.ChannelMessageReconciliation)
            .where(models.ChannelMessageReconciliation.channel_message_id == message_id)
            .order_by(models.ChannelMessageReconciliation.id)
        ).all()
        return tuple(
            ChannelMessageReconciliationRecord(
                id=row.id,
                channel_message_id=row.channel_message_id,
                decision=row.decision,
                reviewer_user_id=row.reviewer_user_id,
                created_at=row.created_at,
            )
            for row in rows
        )


def approve_channel_message(
    message_id: int,
    *,
    reviewer_user_id: str,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)

    def operation(session: Session) -> ChannelMessageRecord | None:
        claimed = session.execute(
            update(models.ChannelMessage)
            .where(
                models.ChannelMessage.id == message_id,
                models.ChannelMessage.status == "pending_review",
            )
            .values(status="pending", approved_by_user_id=str(reviewer_user_id))
        )
        row = session.get(models.ChannelMessage, message_id)
        if claimed.rowcount != 1:
            return channel_message_to_record(row) if row is not None else None
        session.flush()
        return channel_message_to_record(row)

    return run_serialized_transaction(operation, database_url=database_url)


def retry_channel_message(
    message_id: int,
    *,
    reviewer_user_id: str,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    """Release one definitive failed delivery for a moderator-authorized retry."""

    ensure_database_schema(database_url)

    def operation(session: Session) -> ChannelMessageRecord | None:
        row = session.get(models.ChannelMessage, message_id)
        if row is None:
            return None
        if row.status == "failed":
            row.status = "pending"
            row.next_attempt_at = None
            row.failure_reason = None
            row.approved_by_user_id = str(reviewer_user_id)
            session.flush()
        return channel_message_to_record(row)

    return run_serialized_transaction(operation, database_url=database_url)


def reconcile_ambiguous_channel_message(
    message_id: int,
    *,
    reviewer_user_id: str,
    decision: str,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    """Record the human decision required after an unknown Telegram outcome."""

    if decision not in {"absent_retry", "published"}:
        raise ValueError(f"Unsupported reconciliation decision: {decision}")
    ensure_database_schema(database_url)

    def operation(session: Session) -> ChannelMessageRecord | None:
        row = session.get(models.ChannelMessage, message_id)
        if row is None:
            return None
        if row.status != "ambiguous":
            return channel_message_to_record(row)
        session.add(
            models.ChannelMessageReconciliation(
                channel_message_id=message_id,
                decision=decision,
                reviewer_user_id=str(reviewer_user_id),
            )
        )
        row.approved_by_user_id = str(reviewer_user_id)
        row.failure_reason = None
        row.next_attempt_at = None
        if decision == "absent_retry":
            row.status = "reconciled_absent"
        else:
            row.status = "published"
            row.published_at = datetime.now(UTC)
        session.flush()
        return channel_message_to_record(row)

    return run_serialized_transaction(operation, database_url=database_url)


def cancel_channel_message(
    message_id: int, *, database_url: str | None = None
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        row = session.get(models.ChannelMessage, message_id)
        if row is None:
            return None
        if row.status in CANCELLABLE_STATUSES:
            row.status = "cancelled"
            row.failure_reason = None
            session.flush()
        return channel_message_to_record(row)


def prepare_event_announcement_in_session(
    session: Session,
    event: TrackedEvent,
) -> ChannelMessageRecord:
    return _upsert_message(
        session,
        event=event,
        registration_window_id=_current_window_id(session, event.id),
        message_type="event_announced",
        status="pending_review",
    )


def prepare_registration_news_in_session(
    session: Session,
    event: TrackedEvent,
    *,
    current_fields: dict[str, object],
    proposed_fields: dict[str, object],
) -> ChannelMessageRecord | None:
    message_type = _registration_message_type(current_fields, proposed_fields)
    if message_type is None:
        return None
    update_lines = _registration_update_lines(
        message_type,
        current_fields=current_fields,
        proposed_fields=proposed_fields,
    )
    if message_type == "registration_updated" and not update_lines:
        return None
    window_id = _current_window_id(session, event.id)
    _cancel_superseded_registration_drafts(
        session,
        event_id=event.id,
        registration_window_id=window_id,
        keep_message_type=message_type,
    )
    return _upsert_message(
        session,
        event=event,
        registration_window_id=window_id,
        message_type=message_type,
        status="pending_review",
        update_lines=update_lines,
    )


def prepare_event_correction(
    event_id: str,
    *,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    """Prepare an explicit moderator-requested correction from approved event data."""

    ensure_database_schema(database_url)
    normalized_id = normalize_event_id(event_id)
    with session_scope(database_url) as session:
        model = session.scalar(
            select(models.Event)
            .where(models.Event.id == normalized_id, models.Event.removed_at.is_(None))
            .options(
                selectinload(models.Event.regions),
                selectinload(models.Event.collections),
                selectinload(models.Event.legacy_ids),
                selectinload(models.Event.search_keywords),
                selectinload(models.Event.registration_windows),
            )
        )
        if model is None:
            return None
        return _upsert_message(
            session,
            event=_event_to_domain(model),
            registration_window_id=_current_window_id(session, normalized_id),
            message_type="correction",
            status="pending_review",
        )


def cancel_event_messages_in_session(session: Session, event_id: str) -> None:
    session.execute(
        update(models.ChannelMessage)
        .where(
            models.ChannelMessage.event_id == event_id,
            models.ChannelMessage.status.in_(CANCELLABLE_STATUSES),
        )
        .values(status="cancelled", next_attempt_at=None)
    )
    _mark_in_flight_event_messages_ambiguous(session, event_id)


def cancel_event_schedules_in_session(session: Session, event_id: str) -> None:
    session.execute(
        update(models.ChannelMessage)
        .where(
            models.ChannelMessage.event_id == event_id,
            models.ChannelMessage.message_type.in_(SCHEDULED_MESSAGE_TYPES),
            models.ChannelMessage.status.in_(CANCELLABLE_STATUSES),
        )
        .values(status="cancelled", next_attempt_at=None)
    )
    _mark_in_flight_event_messages_ambiguous(
        session,
        event_id,
        message_types=SCHEDULED_MESSAGE_TYPES,
    )


def current_registration_window_id_in_session(
    session: Session,
    event_id: str,
) -> int | None:
    return _current_window_id(session, event_id)


def rebind_event_schedules_in_session(
    session: Session,
    event: TrackedEvent,
    *,
    previous_window_id: int | None,
) -> None:
    """Keep generic event edits from creating a new set of reminder records."""

    current_window_id = _current_window_id(session, event.id)
    if current_window_id == previous_window_id:
        return
    chat_id = target_channel_id()
    rows = session.scalars(
        select(models.ChannelMessage).where(
            models.ChannelMessage.event_id == event.id,
            models.ChannelMessage.registration_window_id == previous_window_id,
            models.ChannelMessage.message_type.in_(SCHEDULED_MESSAGE_TYPES),
            models.ChannelMessage.status == "scheduled",
        )
    ).all()
    for row in rows:
        row.registration_window_id = current_window_id
        row.target_chat_id = chat_id
        row.idempotency_key = channel_message_key(
            event.id,
            current_window_id,
            row.message_type,
            chat_id,
        )


def sync_event_schedules_in_session(
    session: Session,
    event: TrackedEvent,
    *,
    now: datetime | None = None,
) -> tuple[ChannelMessageRecord, ...]:
    now = _as_utc(now or datetime.now(UTC))
    window_id = _current_window_id(session, event.id)
    desired = _schedule_times(event)
    chat_id = target_channel_id()
    result: list[ChannelMessageRecord] = []
    for message_type in SCHEDULED_MESSAGE_TYPES:
        key = channel_message_key(event.id, window_id, message_type, chat_id)
        row = session.scalar(
            select(models.ChannelMessage).where(models.ChannelMessage.idempotency_key == key)
        )
        scheduled_for = desired.get(message_type)
        if scheduled_for is None:
            if (
                row is not None
                and row.scheduled_for is not None
                and row.status in {"scheduled", "cancelled"}
            ):
                row.status = "cancelled"
            continue
        if row is None:
            if scheduled_for <= now:
                continue
            record = _upsert_message(
                session,
                event=event,
                registration_window_id=window_id,
                message_type=message_type,
                status="scheduled",
                scheduled_for=scheduled_for,
            )
            result.append(record)
            continue
        if row.status == "cancelled" and scheduled_for > now:
            row.status = "scheduled"
        if row.status == "scheduled":
            row.scheduled_for = scheduled_for
            row.text = _bounded_text(render_channel_message(message_type, event))
            row.source_url = event.registration_url or event.official_url
        session.flush()
        result.append(channel_message_to_record(row))
    return tuple(result)


def sync_channel_schedules(
    *,
    database_url: str | None = None,
    now: datetime | None = None,
) -> tuple[ChannelMessageRecord, ...]:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        events = session.scalars(
            select(models.Event)
            .where(models.Event.removed_at.is_(None))
            .options(
                selectinload(models.Event.regions),
                selectinload(models.Event.collections),
                selectinload(models.Event.legacy_ids),
                selectinload(models.Event.search_keywords),
                selectinload(models.Event.registration_windows),
            )
        ).all()
        records: list[ChannelMessageRecord] = []
        for model in events:
            # One unschedulable event must not poison the sweep for every other event.
            try:
                with session.begin_nested():
                    records.extend(
                        sync_event_schedules_in_session(session, _event_to_domain(model), now=now)
                    )
            except Exception:
                logger.exception("Channel schedule sync failed for event %s", model.id)
        return tuple(records)


def claim_channel_message(
    message_id: int,
    *,
    database_url: str | None = None,
    now: datetime | None = None,
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)
    claim_time = _as_utc(now or datetime.now(UTC))

    def operation(session: Session) -> ChannelMessageRecord | None:
        row = session.get(models.ChannelMessage, message_id)
        if row is None or row.status == "published":
            return channel_message_to_record(row) if row is not None else None
        if row.status == "publishing":
            return None
        eligible = row.status in {"pending", "reconciled_absent"} or (
            row.status == "scheduled"
            and row.scheduled_for is not None
            and _as_utc(row.scheduled_for) <= claim_time
        ) or (
            row.status == "failed"
            and row.attempt_count < 3
            and (row.next_attempt_at is None or _as_utc(row.next_attempt_at) <= claim_time)
        )
        if not eligible:
            return channel_message_to_record(row)
        row.status = "publishing"
        row.attempt_count += 1
        session.flush()
        return channel_message_to_record(row)

    return run_serialized_transaction(operation, database_url=database_url)


def mark_channel_message_published(
    message_id: int,
    *,
    telegram_message_id: int,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    return _finish_message(
        message_id,
        status="published",
        telegram_message_id=telegram_message_id,
        database_url=database_url,
    )


def mark_channel_message_failed(
    message_id: int,
    *,
    reason: str,
    ambiguous: bool,
    database_url: str | None = None,
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        row = session.get(models.ChannelMessage, message_id)
        if row is None:
            return None
        if row.status != "publishing":
            return channel_message_to_record(row)
        row.status = "ambiguous" if ambiguous else "failed"
        row.failure_reason = reason[:1000]
        row.next_attempt_at = None if ambiguous else datetime.now(UTC) + timedelta(minutes=5)
        session.flush()
        return channel_message_to_record(row)


def list_ready_channel_message_ids(
    *,
    database_url: str | None = None,
    now: datetime | None = None,
    limit: int = 10,
) -> tuple[int, ...]:
    ensure_database_schema(database_url)
    current = _as_utc(now or datetime.now(UTC))
    with session_scope(database_url) as session:
        return tuple(
            session.scalars(
            select(models.ChannelMessage.id)
            .where(
                (models.ChannelMessage.status == "pending")
                | (models.ChannelMessage.status == "reconciled_absent")
                | (
                    (models.ChannelMessage.status == "scheduled")
                    & (models.ChannelMessage.scheduled_for <= current)
                )
                | (
                    (models.ChannelMessage.status == "failed")
                    & (models.ChannelMessage.attempt_count < 3)
                    & (models.ChannelMessage.next_attempt_at <= current)
                )
            )
            .order_by(models.ChannelMessage.scheduled_for, models.ChannelMessage.id)
            .limit(limit)
            ).all()
        )


def recover_stale_publishing_messages(
    *,
    database_url: str | None = None,
    now: datetime | None = None,
    stale_after: timedelta = timedelta(minutes=10),
) -> int:
    """Fail closed when a process died after claiming a Telegram send."""

    ensure_database_schema(database_url)
    cutoff = _as_utc(now or datetime.now(UTC)) - stale_after
    with session_scope(database_url) as session:
        recovered = session.execute(
            update(models.ChannelMessage)
            .where(
                models.ChannelMessage.status == "publishing",
                models.ChannelMessage.updated_at <= cutoff,
            )
            .values(
                status="ambiguous",
                failure_reason="Publisher stopped while delivery outcome was unknown.",
                next_attempt_at=None,
            )
        )
        return recovered.rowcount


def _finish_message(
    message_id: int,
    *,
    status: str,
    telegram_message_id: int,
    database_url: str | None,
) -> ChannelMessageRecord | None:
    ensure_database_schema(database_url)
    with session_scope(database_url) as session:
        row = session.get(models.ChannelMessage, message_id)
        if row is None:
            return None
        if row.status in {"publishing", "ambiguous"}:
            row.status = status
            row.telegram_message_id = telegram_message_id
            row.published_at = datetime.now(UTC)
            row.next_attempt_at = None
            session.flush()
        return channel_message_to_record(row)


def _upsert_message(
    session: Session,
    *,
    event: TrackedEvent,
    registration_window_id: int | None,
    message_type: str,
    status: str,
    scheduled_for: datetime | None = None,
    update_lines: tuple[str, ...] = (),
) -> ChannelMessageRecord:
    if message_type not in CHANNEL_MESSAGE_TYPES:
        raise ValueError(f"Unsupported channel message type: {message_type}")
    chat_id = target_channel_id()
    if message_type in REPEATABLE_NEWS_TYPES:
        row, key = _repeatable_news_row_and_key(
            session,
            event_id=event.id,
            registration_window_id=registration_window_id,
            message_type=message_type,
            target_chat_id=chat_id,
        )
    else:
        key = channel_message_key(event.id, registration_window_id, message_type, chat_id)
        row = session.scalar(
            select(models.ChannelMessage).where(models.ChannelMessage.idempotency_key == key)
        )
    if row is None:
        row = models.ChannelMessage(
            event_id=event.id,
            registration_window_id=registration_window_id,
            message_type=message_type,
            target_chat_id=chat_id,
            idempotency_key=key,
            status=status,
            text=_bounded_text(
                render_channel_message(message_type, event, update_lines=update_lines)
            ),
            source_url=event.registration_url or event.official_url,
            scheduled_for=scheduled_for,
        )
        session.add(row)
        session.flush()
    elif status == "pending_review" and row.status in REFRESHABLE_DRAFT_STATUSES:
        row.status = "pending_review"
        row.scheduled_for = None
        row.text = _bounded_text(
            render_channel_message(message_type, event, update_lines=update_lines)
        )
        row.source_url = event.registration_url or event.official_url
        session.flush()
    return channel_message_to_record(row)


def _repeatable_news_row_and_key(
    session: Session,
    *,
    event_id: str,
    registration_window_id: int | None,
    message_type: str,
    target_chat_id: str,
) -> tuple[models.ChannelMessage | None, str]:
    """Refresh an open repeatable draft, or mint a new occurrence after it published.

    Repeatable news (updates and corrections) may legitimately reach the channel more
    than once per registration window, so a published row must not suppress the next
    approved occurrence. The occurrence index is immutable once a row exists.
    """

    same_window = (
        models.ChannelMessage.registration_window_id.is_(None)
        if registration_window_id is None
        else models.ChannelMessage.registration_window_id == registration_window_id
    )
    identity = (
        models.ChannelMessage.event_id == event_id,
        same_window,
        models.ChannelMessage.message_type == message_type,
        models.ChannelMessage.target_chat_id == target_chat_id,
    )
    latest = session.scalar(
        select(models.ChannelMessage)
        .where(*identity)
        .order_by(models.ChannelMessage.id.desc())
        .limit(1)
    )
    if latest is not None and latest.status in REFRESHABLE_DRAFT_STATUSES:
        return latest, latest.idempotency_key
    occurrence = int(
        session.scalar(select(func.count(models.ChannelMessage.id)).where(*identity)) or 0
    )
    key = channel_message_key(
        event_id,
        registration_window_id,
        message_type,
        target_chat_id,
        occurrence=occurrence,
    )
    return None, key


def _cancel_superseded_registration_drafts(
    session: Session,
    *,
    event_id: str,
    registration_window_id: int | None,
    keep_message_type: str,
) -> None:
    same_window = (
        models.ChannelMessage.registration_window_id.is_(None)
        if registration_window_id is None
        else models.ChannelMessage.registration_window_id == registration_window_id
    )
    session.execute(
        update(models.ChannelMessage)
        .where(
            models.ChannelMessage.event_id == event_id,
            models.ChannelMessage.message_type.in_(REGISTRATION_NEWS_TYPES),
            not_(same_window),
            models.ChannelMessage.status.in_(CANCELLABLE_STATUSES),
        )
        .values(status="cancelled", next_attempt_at=None)
    )
    session.execute(
        update(models.ChannelMessage)
        .where(
            models.ChannelMessage.event_id == event_id,
            models.ChannelMessage.message_type.in_(REGISTRATION_NEWS_TYPES),
            same_window,
            models.ChannelMessage.message_type != keep_message_type,
            models.ChannelMessage.status.in_(
                {"pending_review", "pending", "failed", "reconciled_absent"}
            ),
        )
        .values(status="cancelled", next_attempt_at=None)
    )


def _mark_in_flight_event_messages_ambiguous(
    session: Session,
    event_id: str,
    *,
    message_types: tuple[str, ...] | None = None,
) -> None:
    query = update(models.ChannelMessage).where(
        models.ChannelMessage.event_id == event_id,
        models.ChannelMessage.status == "publishing",
    )
    if message_types is not None:
        query = query.where(models.ChannelMessage.message_type.in_(message_types))
    session.execute(
        query.values(
            status="ambiguous",
            failure_reason="Event changed while Telegram delivery outcome was unknown.",
            next_attempt_at=None,
        )
    )


def _registration_message_type(
    current_fields: dict[str, object], proposed_fields: dict[str, object]
) -> str | None:
    old_status = str(current_fields.get("registration_status") or "unknown")
    new_status = str(proposed_fields.get("registration_status") or old_status)
    if new_status != old_status:
        return {
            "open": "registration_open",
            "closed": "registration_closed",
            "sold_out": "sold_out",
            "waitlist": "waitlist",
        }.get(new_status)
    old_open = current_fields.get("registration_open_at")
    new_open = proposed_fields.get("registration_open_at")
    if "registration_open_at" in proposed_fields and new_open != old_open:
        return (
            "registration_date_discovered" if new_open and not old_open else "registration_updated"
        )
    public_fields = {"registration_close_at", "registration_url", "event_date"}
    if any(
        field in proposed_fields and proposed_fields.get(field) != current_fields.get(field)
        for field in public_fields
    ):
        return "registration_updated"
    return None


def _registration_update_lines(
    message_type: str,
    *,
    current_fields: dict[str, object],
    proposed_fields: dict[str, object],
) -> tuple[str, ...]:
    if message_type != "registration_updated":
        return ()
    labels = {
        "registration_open_at": "Registration opens",
        "registration_close_at": "Registration closes",
        "event_date": "Event date",
    }
    lines: list[str] = []
    for field, label in labels.items():
        if field not in proposed_fields:
            continue
        new_value = proposed_fields.get(field)
        if new_value == current_fields.get(field):
            continue
        if new_value:
            lines.append(f"{label}: {format_public_value(str(new_value))}")
        else:
            lines.append(f"{label}: no date currently announced")
    if "registration_url" in proposed_fields and proposed_fields.get(
        "registration_url"
    ) != current_fields.get("registration_url"):
        new_url = proposed_fields.get("registration_url")
        lines.append(
            "Registration page updated" if new_url else "Registration page is no longer available"
        )
    return tuple(lines)


def _schedule_times(event: TrackedEvent) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    if event.registration_status in {"unknown", "announced"}:
        opened = _parse_event_time(event.registration_open_at, event.timezone, close=False)
        if opened is not None:
            result["opens_tomorrow"] = _previous_local_day(opened, event.timezone)
            result["opens_today"] = opened
    if event.registration_status not in {"closed", "sold_out"}:
        closed = _parse_event_time(event.registration_close_at, event.timezone, close=True)
        if closed is not None:
            reminder = _registration_close_reminder_time(
                event.registration_close_at,
                closed,
                event.timezone,
            )
            if reminder is not None:
                result["closes_tomorrow"] = reminder
            result["registration_closed"] = closed
    return result


def _previous_local_day(value: datetime, timezone_name: str) -> datetime:
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = UTC
    return (value.astimezone(zone) - timedelta(days=1)).astimezone(UTC)


def _registration_close_reminder_time(
    raw_value: str | None,
    closed: datetime,
    timezone_name: str,
) -> datetime | None:
    if not raw_value:
        return None
    if len(raw_value) != 10:
        return _previous_local_day(closed, timezone_name)
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = UTC
    try:
        reminder_day = date.fromisoformat(raw_value) - timedelta(days=1)
    except ValueError:
        return None
    return datetime.combine(reminder_day, time(9), zone).astimezone(UTC)


def _parse_event_time(value: str | None, timezone_name: str, *, close: bool) -> datetime | None:
    if not value:
        return None
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        zone = UTC
    try:
        if len(value) == 10:
            day = date.fromisoformat(value)
            if close:
                day += timedelta(days=1)
                return datetime.combine(day, time(0, 5), zone).astimezone(UTC)
            return datetime.combine(day, time(9), zone).astimezone(UTC)
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=zone)
    return parsed.astimezone(UTC)


def _current_window_id(session: Session, event_id: str) -> int | None:
    event = session.get(models.Event, event_id)
    if event is None:
        return None
    query = select(models.RegistrationWindow).where(
        models.RegistrationWindow.event_id == event_id
    )
    if event.current_edition_id is not None:
        edition_window = session.scalar(
            query.where(
                models.RegistrationWindow.event_edition_id == event.current_edition_id
            ).order_by(models.RegistrationWindow.id.desc())
        )
        if edition_window is not None:
            return edition_window.id
    window = session.scalar(query.order_by(models.RegistrationWindow.id.desc()))
    return window.id if window is not None else None


def _event_to_domain(event: models.Event) -> TrackedEvent:
    window = None
    windows = list(event.registration_windows or ())
    if event.current_edition_id is not None:
        matches = [w for w in windows if w.event_edition_id == event.current_edition_id]
        if matches:
            window = max(matches, key=lambda item: item.id)
    if window is None and windows:
        window = max(windows, key=lambda item: item.id)
    return TrackedEvent(
        id=event.id,
        public_id=event.public_id,
        legacy_ids=tuple(item.legacy_id for item in event.legacy_ids),
        search_keywords=tuple(item.keyword for item in event.search_keywords),
        name=event.canonical_name,
        city=event.city,
        country=event.country,
        timezone=event.timezone,
        distances=tuple(event.distances or ()),
        regions=tuple(item.region_tag for item in event.regions),
        collections=tuple(item.collection_slug for item in event.collections),
        event_date=event.next_event_date,
        registration_status=event.registration_status,
        official_url=event.official_url,
        registration_url=event.registration_url,
        registration_open_at=window.registration_open_at if window else None,
        registration_open_precision=window.registration_open_precision if window else "unknown",
        registration_close_at=window.registration_close_at if window else None,
    )


def channel_message_to_record(row: models.ChannelMessage) -> ChannelMessageRecord:
    return ChannelMessageRecord(
        id=row.id,
        event_id=row.event_id,
        registration_window_id=row.registration_window_id,
        message_type=row.message_type,
        target_chat_id=row.target_chat_id,
        idempotency_key=row.idempotency_key,
        status=row.status,
        text=row.text,
        source_url=row.source_url,
        scheduled_for=row.scheduled_for,
        approved_by_user_id=row.approved_by_user_id,
        attempt_count=row.attempt_count,
        next_attempt_at=row.next_attempt_at,
        telegram_message_id=row.telegram_message_id,
        failure_reason=row.failure_reason,
        published_at=row.published_at,
    )


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _bounded_text(text: str) -> str:
    if len(text) > 4096:
        raise ValueError("Telegram channel message exceeds 4096 characters.")
    return text
