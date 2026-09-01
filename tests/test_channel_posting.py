from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiogram.exceptions import TelegramBadRequest, TelegramServerError

from run4221.config import get_telegram_channel_settings
from run4221.db.bootstrap import initialize_database
from run4221.db.repository import (
    EventCreate,
    EventUpdate,
    ProposedEventUpdateCreate,
    add_event,
    approve_proposed_event_update,
    archive_event,
    create_proposed_event_update,
    restore_event,
    update_event,
)
from run4221.posting import ledger
from run4221.posting.ledger import (
    approve_channel_message,
    claim_channel_message,
    get_channel_message,
    list_channel_message_reconciliations,
    list_channel_messages,
    mark_channel_message_published,
    prepare_event_correction,
    reconcile_ambiguous_channel_message,
    recover_stale_publishing_messages,
    sync_channel_schedules,
)
from run4221.posting.publisher import publish_channel_message
from run4221.posting.scheduler import run_channel_publisher_cycle
from run4221.posting.templates import render_channel_message


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'channel-posting.sqlite3'}"


def event_payload(**changes) -> EventCreate:
    values = {
        "public_id": "berlin.42",
        "name": "Berlin Marathon",
        "city": "Berlin",
        "country": "Germany",
        "timezone": "Europe/Berlin",
        "distances": ("marathon",),
        "regions": ("global", "eu", "de"),
        "official_url": "https://example.com/berlin",
        "registration_url": "https://example.com/berlin/register",
        "event_date": "2099-09-27",
        "registration_status": "announced",
        "registration_open_at": "2099-09-10T10:00:00+02:00",
        "registration_open_precision": "datetime",
        "registration_close_at": "2099-09-20T18:00:00+02:00",
    }
    values.update(changes)
    return EventCreate(**values)


class FakeBot:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.sent: list[dict[str, object]] = []

    async def send_message(self, **kwargs):
        self.sent.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(message_id=123)


def test_event_creation_prepares_news_and_scheduled_reminders(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)

    event = add_event(event_payload(), database_url=url)

    messages = list_channel_messages(event_id=event.id, database_url=url)
    assert [(message.message_type, message.status) for message in messages] == [
        ("event_announced", "pending_review"),
        ("opens_tomorrow", "scheduled"),
        ("opens_today", "scheduled"),
        ("closes_tomorrow", "scheduled"),
        ("registration_closed", "scheduled"),
    ]
    assert all(message.telegram_message_id is None for message in messages)
    assert all("https://example.com/berlin" in message.text for message in messages)


def test_approved_registration_update_prepares_one_reviewable_news_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(registration_status="unknown"), database_url=url)
    proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "unknown"},
            proposed_fields={"registration_status": "open"},
            evidence=("Official page says registration is open.",),
            confidence=0.99,
            change_summary="Registration is open.",
        ),
        database_url=url,
    )

    first = approve_proposed_event_update(proposal.id, reviewer_user_id="42", database_url=url)
    second = approve_proposed_event_update(proposal.id, reviewer_user_id="42", database_url=url)

    assert first is not None
    assert first.channel_message is not None
    assert first.channel_message.message_type == "registration_open"
    assert first.channel_message.status == "pending_review"
    assert second is None
    registration_news = [
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "registration_open"
    ]
    assert len(registration_news) == 1


def test_archive_is_silent_and_only_cancels_unpublished_messages(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)

    archived = archive_event(event.id, database_url=url)

    assert archived is not None
    messages = list_channel_messages(event_id=event.id, database_url=url)
    assert {message.message_type for message in messages} == {
        "event_announced",
        "opens_tomorrow",
        "opens_today",
        "closes_tomorrow",
        "registration_closed",
    }
    assert {message.status for message in messages} == {"cancelled"}

    restored = restore_event(event.id, database_url=url)
    assert restored is not None
    restored_messages = list_channel_messages(event_id=event.id, database_url=url)
    assert next(
        message for message in restored_messages if message.message_type == "event_announced"
    ).status == "cancelled"
    assert {
        message.status
        for message in restored_messages
        if message.message_type
        in {"opens_tomorrow", "opens_today", "closes_tomorrow", "registration_closed"}
    } == {"scheduled"}


def test_generic_edit_does_not_create_channel_records(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    before = list_channel_messages(event_id=event.id, database_url=url)

    updated = update_event(
        event.id,
        EventUpdate(
            name=event.name,
            city="Potsdam",
            country=event.country,
            timezone=event.timezone,
            distances=event.distances,
            regions=event.regions,
            official_url=event.official_url,
            registration_url=event.registration_url,
            event_date=event.event_date,
            registration_status=event.registration_status,
            registration_open_at=event.registration_open_at,
            registration_open_precision=event.registration_open_precision,
            registration_close_at=event.registration_close_at,
        ),
        database_url=url,
    )

    assert updated is not None
    after = list_channel_messages(event_id=event.id, database_url=url)
    assert [message.id for message in after] == [message.id for message in before]
    assert next(message for message in after if message.message_type == "opens_today").text != next(
        message for message in before if message.message_type == "opens_today"
    ).text


def test_moderator_approval_publishes_once_and_records_telegram_id(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approved = approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    assert approved is not None
    assert approved.status == "pending"
    bot = FakeBot()

    first = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))
    second = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))

    assert first is not None
    assert first.status == "published"
    assert first.telegram_message_id == 123
    assert second is not None
    assert second.status == "published"
    assert len(bot.sent) == 1
    assert bot.sent[0]["chat_id"] == "@run4221"


def test_schedule_sync_is_idempotent(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    now = datetime(2099, 9, 1, tzinfo=UTC)

    sync_channel_schedules(database_url=url, now=now)
    sync_channel_schedules(database_url=url, now=now)

    messages = list_channel_messages(event_id=event.id, database_url=url)
    assert sum(message.message_type == "opens_tomorrow" for message in messages) == 1
    assert sum(message.message_type == "opens_today" for message in messages) == 1
    assert sum(message.message_type == "closes_tomorrow" for message in messages) == 1
    assert sum(message.message_type == "registration_closed" for message in messages) == 1


def test_detected_closure_replaces_future_close_reminder_with_review_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "announced"},
            proposed_fields={"registration_status": "closed"},
            evidence=("Official registration is closed.",),
            confidence=0.99,
        ),
        database_url=url,
    )

    result = approve_proposed_event_update(
        proposal.id,
        reviewer_user_id="42",
        database_url=url,
    )

    assert result is not None
    assert result.channel_message is not None
    assert result.channel_message.message_type == "registration_closed"
    close_messages = [
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "registration_closed"
    ]
    assert len(close_messages) == 1
    assert close_messages[0].status == "pending_review"
    assert close_messages[0].scheduled_for is None


def test_ambiguous_delivery_is_never_retried_automatically(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    bot = FakeBot(error=TimeoutError("delivery outcome unknown"))

    first = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))
    second = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))

    assert first is not None
    assert first.status == "ambiguous"
    assert second is not None
    assert second.status == "ambiguous"
    assert len(bot.sent) == 1

    unreconciled = asyncio.run(publish_channel_message(FakeBot(), draft.id, database_url=url))
    assert unreconciled is not None
    assert unreconciled.status == "ambiguous"

    released = reconcile_ambiguous_channel_message(
        draft.id,
        reviewer_user_id="99",
        decision="absent_retry",
        database_url=url,
    )
    assert released is not None
    assert released.status == "reconciled_absent"
    actions = list_channel_message_reconciliations(draft.id, database_url=url)
    assert [(action.decision, action.reviewer_user_id) for action in actions] == [
        ("absent_retry", "99")
    ]
    success_bot = FakeBot()
    reconciled = asyncio.run(
        publish_channel_message(success_bot, draft.id, database_url=url)
    )
    assert reconciled is not None
    assert reconciled.status == "published"
    assert len(success_bot.sent) == 1


def test_ambiguous_delivery_can_be_reconciled_as_already_published(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    asyncio.run(
        publish_channel_message(
            FakeBot(error=TimeoutError("delivery outcome unknown")),
            draft.id,
            database_url=url,
        )
    )

    reconciled = reconcile_ambiguous_channel_message(
        draft.id,
        reviewer_user_id="99",
        decision="published",
        database_url=url,
    )

    assert reconciled is not None
    assert reconciled.status == "published"
    assert reconciled.telegram_message_id is None
    actions = list_channel_message_reconciliations(draft.id, database_url=url)
    assert [(action.decision, action.reviewer_user_id) for action in actions] == [
        ("published", "99")
    ]


def test_stale_publish_claim_fails_closed_as_ambiguous(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    claimed = claim_channel_message(draft.id, database_url=url)
    assert claimed is not None
    assert claimed.status == "publishing"

    recovered = recover_stale_publishing_messages(
        database_url=url,
        now=datetime.now(UTC) + timedelta(minutes=11),
    )

    assert recovered == 1
    stored = get_channel_message(draft.id, database_url=url)
    assert stored is not None
    assert stored.status == "ambiguous"


def test_due_reminder_survives_sync_and_publishes_once_in_cycle(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    bot = FakeBot()
    due = datetime(2099, 9, 9, 8, tzinfo=UTC)

    asyncio.run(run_channel_publisher_cycle(bot, database_url=url, now=due))
    asyncio.run(run_channel_publisher_cycle(bot, database_url=url, now=due))

    tomorrow = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "opens_tomorrow"
    )
    assert tomorrow.status == "published"
    assert len(bot.sent) == 1


def test_concurrent_publisher_cannot_send_one_claim_twice(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)

    class BlockingBot(FakeBot):
        def __init__(self) -> None:
            super().__init__()
            self.started = asyncio.Event()
            self.release = asyncio.Event()

        async def send_message(self, **kwargs):
            self.sent.append(kwargs)
            self.started.set()
            await self.release.wait()
            return SimpleNamespace(message_id=123)

    async def publish_concurrently() -> tuple[object, object]:
        bot = BlockingBot()
        first_task = asyncio.create_task(
            publish_channel_message(bot, draft.id, database_url=url)
        )
        await bot.started.wait()
        second = await publish_channel_message(bot, draft.id, database_url=url)
        bot.release.set()
        first = await first_task
        assert len(bot.sent) == 1
        return first, second

    first, second = asyncio.run(publish_concurrently())

    assert first is not None
    assert first.status == "published"
    assert second is None


def test_archive_preserves_unknown_send_and_late_success_is_recorded(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    claimed = claim_channel_message(draft.id, database_url=url)
    assert claimed is not None
    assert claimed.status == "publishing"

    archive_event(event.id, database_url=url)

    unknown = get_channel_message(draft.id, database_url=url)
    assert unknown is not None
    assert unknown.status == "ambiguous"
    completed = mark_channel_message_published(
        draft.id,
        telegram_message_id=456,
        database_url=url,
    )
    assert completed is not None
    assert completed.status == "published"
    assert completed.telegram_message_id == 456


def test_new_registration_fact_supersedes_older_pending_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(registration_status="unknown"), database_url=url)
    opened = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "unknown"},
            proposed_fields={"registration_status": "open"},
            evidence=("Official registration is open.",),
            confidence=0.99,
        ),
        database_url=url,
    )
    opened_result = approve_proposed_event_update(
        opened.id,
        reviewer_user_id="42",
        database_url=url,
    )
    assert opened_result is not None
    assert opened_result.channel_message is not None
    approve_channel_message(
        opened_result.channel_message.id,
        reviewer_user_id="42",
        database_url=url,
    )
    closed = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "open"},
            proposed_fields={
                "registration_status": "closed",
                "event_date": "2100-09-26",
            },
            evidence=("Official registration is closed.",),
            confidence=0.99,
        ),
        database_url=url,
    )

    approve_proposed_event_update(closed.id, reviewer_user_id="42", database_url=url)

    messages = list_channel_messages(event_id=event.id, database_url=url)
    assert next(
        message for message in messages if message.message_type == "registration_open"
    ).status == "cancelled"
    close_messages = [
        message for message in messages if message.message_type == "registration_closed"
    ]
    assert {message.status for message in close_messages} == {"cancelled", "pending_review"}


def test_explicit_correction_prepares_reviewable_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)

    correction = prepare_event_correction(event.id, database_url=url)

    assert correction is not None
    assert correction.message_type == "correction"
    assert correction.status == "pending_review"
    assert "<b>⚠️ Event update</b>" in correction.text
    assert "Official event page" in correction.text


def test_file_backed_channel_setting_is_used_for_ledger_and_delivery(
    tmp_path,
    monkeypatch,
) -> None:
    env_dir = tmp_path / "runtime"
    env_dir.mkdir()
    (env_dir / ".env").write_text("TELEGRAM_CHANNEL_ID=@run4221_test\n")
    monkeypatch.chdir(env_dir)
    monkeypatch.delenv("TELEGRAM_CHANNEL_ID", raising=False)
    get_telegram_channel_settings.cache_clear()
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    bot = FakeBot()

    try:
        asyncio.run(publish_channel_message(bot, draft.id, database_url=url))
    finally:
        get_telegram_channel_settings.cache_clear()

    assert draft.target_chat_id == "@run4221_test"
    assert bot.sent[0]["chat_id"] == "@run4221_test"


def test_channel_template_renders_event_card_before_concrete_update() -> None:
    event = SimpleNamespace(
        name="<b>Berlin & Friends</b>",
        distance_label="Half marathon",
        event_date="2026-10-25",
        location="Berlin & Brandenburg, Germany",
        official_url="https://example.com/?a=1&b=2",
        registration_url="https://example.com/register?a=1&b=2",
        registration_open_at="2026-10-15T10:00:00+02:00",
        registration_close_at="2099-09-20T18:00:00+02:00",
        regions=("global", "eu", "de"),
    )

    text = render_channel_message("opens_tomorrow", event)

    assert "<b>Berlin & Friends</b>" not in text
    assert "&lt;b&gt;Berlin &amp; Friends&lt;/b&gt;" in text
    assert "https://example.com/?a=1&amp;b=2" in text
    assert "#GLOBAL" not in text
    assert "#EU" not in text
    assert "#DE" not in text
    assert text.startswith(
        "<b>&lt;b&gt;Berlin &amp; Friends&lt;/b&gt;</b>\n"
        "Half marathon\n"
        "Event date: 2026-10-25\n"
        "Berlin &amp; Brandenburg, Germany\n\n"
        "<b>🔔 Registration opens tomorrow</b>\n"
        "2026-10-15 10:00:00 +02:00"
    )


def test_registration_update_names_the_changed_value(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={
                "registration_close_at": "2099-09-20T18:00:00+02:00",
            },
            proposed_fields={
                "registration_close_at": "2099-09-21T20:00:00+02:00",
            },
            evidence=("Official page lists the extended deadline.",),
            confidence=0.99,
        ),
        database_url=url,
    )

    result = approve_proposed_event_update(
        proposal.id,
        reviewer_user_id="42",
        database_url=url,
    )

    assert result is not None
    assert result.channel_message is not None
    assert result.channel_message.message_type == "registration_updated"
    assert "<b>⚠️ Registration update</b>" in result.channel_message.text
    assert "Registration closes: 2099-09-21 20:00:00 +02:00" in result.channel_message.text


def test_unchanged_registration_fields_do_not_create_update_news(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_url": event.registration_url},
            proposed_fields={"registration_url": event.registration_url},
            evidence=("Official registration link is unchanged.",),
            confidence=0.99,
        ),
        database_url=url,
    )

    result = approve_proposed_event_update(
        proposal.id,
        reviewer_user_id="42",
        database_url=url,
    )

    assert result is not None
    assert result.channel_message is None


def test_date_only_registration_close_schedules_tomorrow_reminder_at_local_morning(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(
        event_payload(registration_close_at="2099-09-20"),
        database_url=url,
    )

    reminder = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "closes_tomorrow"
    )

    assert reminder.scheduled_for == datetime(2099, 9, 19, 7)
    assert "<b>🔔 Registration closes tomorrow</b>" in reminder.text
    assert "2099-09-20" in reminder.text


def test_second_approved_update_after_publish_creates_new_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    first_proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_close_at": "2099-09-20T18:00:00+02:00"},
            proposed_fields={"registration_close_at": "2099-09-21T18:00:00+02:00"},
            evidence=("Official page extends the registration deadline.",),
            confidence=0.99,
        ),
        database_url=url,
    )
    first_result = approve_proposed_event_update(
        first_proposal.id, reviewer_user_id="42", database_url=url
    )
    assert first_result is not None
    assert first_result.channel_message is not None
    assert first_result.channel_message.message_type == "registration_updated"
    approve_channel_message(
        first_result.channel_message.id, reviewer_user_id="42", database_url=url
    )
    published = asyncio.run(
        publish_channel_message(FakeBot(), first_result.channel_message.id, database_url=url)
    )
    assert published is not None
    assert published.status == "published"

    second_proposal = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_close_at": "2099-09-21T18:00:00+02:00"},
            proposed_fields={"registration_close_at": "2099-09-22T18:00:00+02:00"},
            evidence=("Official page extends the registration deadline again.",),
            confidence=0.99,
        ),
        database_url=url,
    )
    second_result = approve_proposed_event_update(
        second_proposal.id, reviewer_user_id="42", database_url=url
    )

    assert second_result is not None
    assert second_result.channel_message is not None
    assert second_result.channel_message.message_type == "registration_updated"
    assert second_result.channel_message.status == "pending_review"
    assert second_result.channel_message.id != first_result.channel_message.id
    update_news = [
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "registration_updated"
    ]
    assert [(message.status,) for message in update_news] == [
        ("published",),
        ("pending_review",),
    ]
    assert len({message.idempotency_key for message in update_news}) == 2


def test_repeated_correction_after_publish_creates_new_draft(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)

    first = prepare_event_correction(event.id, database_url=url)
    assert first is not None
    approve_channel_message(first.id, reviewer_user_id="42", database_url=url)
    published = asyncio.run(publish_channel_message(FakeBot(), first.id, database_url=url))
    assert published is not None
    assert published.status == "published"

    second = prepare_event_correction(event.id, database_url=url)
    assert second is not None
    assert second.id != first.id
    assert second.status == "pending_review"

    refreshed = prepare_event_correction(event.id, database_url=url)
    assert refreshed is not None
    assert refreshed.id == second.id
    corrections = [
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "correction"
    ]
    assert len(corrections) == 2


def test_server_error_delivery_is_ambiguous_and_not_retried(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    bot = FakeBot(error=TelegramServerError(method=None, message="Bad Gateway"))

    first = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))

    assert first is not None
    assert first.status == "ambiguous"
    assert first.next_attempt_at is None

    success_bot = FakeBot()
    second = asyncio.run(publish_channel_message(success_bot, draft.id, database_url=url))
    assert second is not None
    assert second.status == "ambiguous"
    assert success_bot.sent == []


def test_definitive_delivery_failure_schedules_bounded_retry(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)
    bot = FakeBot(error=TelegramBadRequest(method=None, message="chat not found"))

    failed = asyncio.run(publish_channel_message(bot, draft.id, database_url=url))

    assert failed is not None
    assert failed.status == "failed"
    assert failed.next_attempt_at is not None
    assert failed.attempt_count == 1


def test_publisher_cycle_delivers_when_schedule_sync_fails(tmp_path, monkeypatch) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    event = add_event(event_payload(), database_url=url)
    draft = next(
        message
        for message in list_channel_messages(event_id=event.id, database_url=url)
        if message.message_type == "event_announced"
    )
    approve_channel_message(draft.id, reviewer_user_id="42", database_url=url)

    def broken_sync(**kwargs):
        raise RuntimeError("schedule sweep failed")

    def broken_recovery(**kwargs):
        raise RuntimeError("stale recovery failed")

    monkeypatch.setattr("run4221.posting.scheduler.sync_channel_schedules", broken_sync)
    monkeypatch.setattr(
        "run4221.posting.scheduler.recover_stale_publishing_messages", broken_recovery
    )
    bot = FakeBot()

    asyncio.run(run_channel_publisher_cycle(bot, database_url=url))

    stored = get_channel_message(draft.id, database_url=url)
    assert stored is not None
    assert stored.status == "published"
    assert len(bot.sent) == 1


def test_schedule_sweep_survives_poison_event(tmp_path, monkeypatch) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    poison = add_event(event_payload(), database_url=url)
    healthy = add_event(
        event_payload(public_id="hamburg.42", name="Hamburg Marathon", city="Hamburg"),
        database_url=url,
    )
    original = ledger.sync_event_schedules_in_session

    def poisoned_sync(session, event, *, now=None):
        if event.id == poison.id:
            raise ValueError("Telegram channel message exceeds 4096 characters.")
        return original(session, event, now=now)

    monkeypatch.setattr(ledger, "sync_event_schedules_in_session", poisoned_sync)

    records = sync_channel_schedules(database_url=url)

    assert records
    assert {record.event_id for record in records} == {healthy.id}
