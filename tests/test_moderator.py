import asyncio
from types import SimpleNamespace

import pytest

from run4221.bot import moderator
from run4221.bot.auth import is_moderator_account, is_moderator_id, is_moderator_username
from run4221.bot.moderator import (
    ARCHIVE_LIST_MAX_LIMIT,
    archive_event_callback,
    archive_event_confirmation_keyboard,
    archive_list_callback,
    archive_show_callback,
    archived_event_card_keyboard,
    archived_event_detail_keyboard,
    delete_event_callback,
    delete_event_confirm_callback,
    delete_event_preview_keyboard,
    format_archive_event_confirmation,
    format_archived_event_card,
    format_archived_event_detail,
    format_archived_events,
    format_delete_event_confirmation,
    format_delete_event_final_confirmation,
    format_draft_field_line,
    format_draft_summary,
    format_edit_event_prompt,
    format_edit_field_error,
    format_evidence_for_display,
    format_existing_url_warning,
    format_field_value,
    format_moderator_status,
    format_proposed_update_card,
    format_proposed_update_detail,
    format_proposed_update_list,
    format_suggestion_card,
    format_suggestion_detail,
    format_suggestion_queue,
    format_update_partial_confirmation,
    format_update_partial_selection,
    format_update_review_confirmation,
    is_archive_event_confirmation,
    is_delete_event_confirmation,
    parse_archive_limit,
    parse_distances,
    parse_edit_field,
    parse_edit_value,
    parse_optional_date,
    parse_queue_number,
    parse_regions,
    parse_suggestion_limit,
    parse_suggestion_record_id,
    parse_update_limit,
    parse_update_show_callback_payload,
    proposed_update_apply_confirm_callback,
    proposed_update_card_callback,
    proposed_update_confirmation_keyboard,
    proposed_update_detail_keyboard,
    proposed_update_list_callback,
    proposed_update_list_keyboard,
    proposed_update_partial_apply_callback,
    proposed_update_partial_callback,
    proposed_update_partial_confirm_callback,
    proposed_update_partial_confirmation_keyboard,
    proposed_update_partial_keyboard,
    proposed_update_partial_toggle_callback,
    proposed_update_reject_confirm_callback,
    proposed_update_show_callback,
    proposed_update_show_keyboard,
    restore_event_callback,
    restore_event_confirm_callback,
    suggestion_add_callback,
    suggestion_card_callback,
    suggestion_detail_keyboard,
    suggestion_list_callback,
    suggestion_queue_keyboard,
    suggestion_remove_callback,
    suggestion_show_callback,
    suggestion_show_keyboard,
    supported_distance_codes,
    todo_keyboard,
)
from run4221.config import parse_moderator_accounts
from run4221.db.repository import (
    EVENT_SUGGESTION_MAX_PENDING_TOTAL,
    EventWriteError,
    ProposedEventUpdatePartialApplyResult,
    ProposedEventUpdateRecord,
    normalize_public_id,
)
from run4221.events import DISTANCE_CODE_TO_KEY, DISTANCE_LABELS, TrackedEvent


class FakeMessage:
    def __init__(self, text: str | None = None) -> None:
        self.text = text
        self.answers: list[str] = []
        self.answer_kwargs: list[dict] = []
        self.edits: list[str] = []
        self.edit_kwargs: list[dict] = []

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append(text)
        self.answer_kwargs.append(kwargs)

    async def edit_text(self, text: str, **kwargs) -> None:
        self.edits.append(text)
        self.edit_kwargs.append(kwargs)


class FakeState:
    def __init__(self) -> None:
        self.state = "active"
        self.cleared = False

    async def get_state(self):
        return self.state

    async def clear(self) -> None:
        self.state = None
        self.cleared = True


class FakeCallback:
    def __init__(self, data: str, message: FakeMessage | None = None) -> None:
        self.data = data
        self.message = message or FakeMessage()
        self.from_user = SimpleNamespace(id=42, username="in4lio")
        self.answers: list[tuple[str | None, dict]] = []

    async def answer(self, text: str | None = None, **kwargs) -> None:
        self.answers.append((text, kwargs))


def sample_event(
    public_id: str = "berlin.42",
    name: str = "Berlin Marathon",
) -> TrackedEvent:
    return TrackedEvent(
        id=public_id,
        public_id=public_id,
        legacy_ids=(),
        search_keywords=(),
        name=name,
        city="Berlin",
        country="Germany",
        timezone="Europe/Berlin",
        distances=("marathon",),
        regions=("global", "eu", "de"),
        collections=("major",),
        event_date="2026-09-27",
        registration_status="unknown",
        official_url="https://example.com/berlin",
        registration_url=None,
    )


def test_parse_moderator_accounts_splits_ids_and_usernames() -> None:
    assert parse_moderator_accounts("42,@in4lio, In4lio, 7") == ((42, 7), ("in4lio",))


def test_moderator_auth_checks_unified_accounts() -> None:
    accounts = ((1, 42), ("in4lio",))

    assert is_moderator_account(42, None, accounts)
    assert is_moderator_account(None, "@In4lio", accounts)
    assert is_moderator_account(7, "in4lio", accounts)
    assert not is_moderator_account(7, "someone_else", accounts)


def test_moderator_auth_checks_configured_ids() -> None:
    assert is_moderator_id(42, (1, 42))
    assert not is_moderator_id(7, (1, 42))
    assert not is_moderator_id(None, (1, 42))


def test_moderator_auth_checks_configured_usernames() -> None:
    assert is_moderator_username("in4lio", ("in4lio",))
    assert is_moderator_username("@In4lio", ("in4lio",))
    assert not is_moderator_username("someone_else", ("in4lio",))
    assert not is_moderator_username(None, ("in4lio",))


def test_moderator_parsers_accept_add_event_inputs() -> None:
    assert parse_distances("42") == ("marathon",)
    assert parse_distances("21") == ("half_marathon",)
    assert parse_distances("42,21") == ("marathon", "half_marathon")
    assert parse_regions("global, eu, ch") == ("global", "eu", "ch")
    assert parse_optional_date("2027-04-18") == "2027-04-18"
    assert parse_optional_date("-") is None


def test_distance_draft_field_displays_input_codes() -> None:
    assert format_field_value("distances", ("marathon", "half_marathon")) == "42,21"
    assert format_field_value("regions", ("global", "eu")) == "global,eu"


def test_draft_field_line_underlines_proposed_text_value() -> None:
    assert (
        format_draft_field_line("name", "TCS Amsterdam Marathon")
        == "<b>Draft</b>: <u>TCS Amsterdam Marathon</u>"
    )


def test_draft_field_line_keeps_public_id_copyable() -> None:
    assert format_draft_field_line("public_id", "amsterdam.42") == (
        "<b>Draft</b>: <code>amsterdam.42</code>"
    )


def test_distance_draft_confirmation_uses_distance_buttons_without_help_text() -> None:
    message = FakeMessage()

    asyncio.run(
        moderator.ask_field_confirmation(
            message,
            "distances",
            ("marathon", "half_marathon"),
        )
    )

    assert message.answers == [
        "<b>💬 Distance</b>\n"
        "<b>Draft</b>: <u>42,21</u>\n"
        "Reply ok to keep it, or send the corrected value."
    ]
    keyboard = message.answer_kwargs[0]["reply_markup"].keyboard
    assert [button.text for button in keyboard[0]] == ["ok"]
    assert [button.text for button in keyboard[1]] == ["42", "21", "42,21"]
    assert [button.text for button in keyboard[2]] == ["Cancel"]


def test_distance_edit_prompt_uses_distance_buttons_without_supported_text() -> None:
    message = FakeMessage()

    asyncio.run(moderator.ask_edit_value(message, "distances", ("marathon",)))

    assert "Supported now" not in message.answers[0]
    assert "42=Marathon" not in message.answers[0]
    assert "Public ID cannot be changed" in message.answers[0]
    keyboard = message.answer_kwargs[0]["reply_markup"].keyboard
    assert [button.text for button in keyboard[0]] == ["42", "21", "42,21"]
    assert [button.text for button in keyboard[1]] == ["Cancel"]


def test_existing_url_warning_bolds_warning_label() -> None:
    warning = format_existing_url_warning(
        (SimpleNamespace(name="Berlin Marathon", public_id="berlin.42"),)
    )

    assert warning.startswith("⚠️ <b>Warning</b>:")


def test_suggestion_queue_formats_add_event_hint() -> None:
    queue = format_suggestion_queue(
        (
            SimpleNamespace(
                id=7,
                event_name="Baden Marathon",
                url="https://www.badenmarathon.de/",
                event_date="2026-09-20",
                location="Karlsruhe, Germany",
                region_tags=("eu", "de"),
                distances=("marathon",),
                submitter_username="runner",
                submitter_display_name=None,
                submitter_user_id="42",
                note="Please track it.",
            ),
        )
    )

    assert "<b>Suggestion #7</b>" in queue
    assert "<b>Name</b>: Baden Marathon" in queue
    assert "<b>Distances</b>: <u>42</u>" in queue
    assert "<b>URL</b>" not in queue
    assert "<b>From</b>" not in queue
    assert "<b>Note</b>" not in queue
    assert "/add_event 1" not in queue
    assert "/reject_suggestion 1" not in queue


def test_suggestion_queue_keyboard_formats_action_buttons() -> None:
    keyboard = suggestion_queue_keyboard(
        (
            SimpleNamespace(
                id=7,
                event_name="Baden Marathon",
            ),
        )
    )

    button = keyboard.inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == suggestion_show_callback(7)


def test_suggestion_show_keyboard_formats_single_button() -> None:
    suggestion = SimpleNamespace(id=7, event_name="Baden Marathon")

    keyboard = suggestion_show_keyboard(suggestion, limit=5)

    button = keyboard.inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == suggestion_show_callback(7, list_limit=5)


def test_suggestion_detail_keyboard_has_apply_reject_and_back_buttons() -> None:
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
    )
    keyboard = suggestion_detail_keyboard(suggestion, sequence=3, list_limit=5)

    apply, reject = keyboard.inline_keyboard[0]
    back = keyboard.inline_keyboard[1][0]
    assert apply.text == "Apply"
    assert apply.callback_data == suggestion_add_callback(7, sequence=3, list_limit=5)
    assert reject.text == "Reject"
    assert reject.callback_data == suggestion_remove_callback(7, sequence=3, list_limit=5)
    assert back.text == "Back"
    assert back.callback_data == suggestion_card_callback(7, list_limit=5)


def test_show_suggestion_callback_edits_list_message(monkeypatch) -> None:
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username="runner",
        submitter_display_name=None,
        submitter_user_id="42",
        note="Please track it.",
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "get_event_suggestion", lambda *args, **kwargs: suggestion)

    message = FakeMessage()
    callback = FakeCallback(suggestion_show_callback(7, list_limit=5), message)
    asyncio.run(moderator.handle_show_suggestion_callback(callback))

    assert message.edits == [format_suggestion_detail(suggestion)]
    assert message.edits[0].startswith("<b>✨ Suggestion #7</b>\n\n")
    keyboard = message.edit_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Apply"
    assert keyboard.inline_keyboard[1][0].text == "Back"


def test_reject_suggestion_callback_confirms_then_replaces_result(monkeypatch) -> None:
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username="runner",
        submitter_display_name=None,
        submitter_user_id="42",
        note=None,
    )
    removed_ids = []

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "get_event_suggestion", lambda *args, **kwargs: suggestion)
    monkeypatch.setattr(
        moderator,
        "update_event_suggestion_status",
        lambda suggestion_id, status: removed_ids.append((suggestion_id, status)),
    )

    message = FakeMessage()
    callback = FakeCallback(suggestion_remove_callback(7, sequence=2, list_limit=5), message)
    asyncio.run(moderator.handle_reject_suggestion_callback(callback))

    assert "<b>Confirm suggestion rejection</b>" in message.edits[0]
    assert message.edit_kwargs[0]["reply_markup"].inline_keyboard[0][0].text == "Confirm"

    confirm_callback = FakeCallback(
        moderator.suggestion_remove_confirm_callback(7, sequence=2, list_limit=5),
        message,
    )
    asyncio.run(moderator.handle_reject_suggestion_confirm_callback(confirm_callback))

    assert removed_ids == [(7, "removed")]
    assert message.edits[-1] == "Rejected suggestion <code>#7</code>: <b>Baden Marathon</b>."
    assert message.edit_kwargs[-1]["reply_markup"] is None


def test_suggestion_queue_uses_stable_suggestion_ids() -> None:
    queue = format_suggestion_queue(
        (
            SimpleNamespace(
                id=7,
                event_name="Baden Marathon",
                url="https://www.badenmarathon.de/",
                distances=("marathon",),
                submitter_username="runner",
                submitter_display_name=None,
                submitter_user_id="42",
                note=None,
            ),
        ),
        start=3,
        title="Pending suggestion",
    )

    assert "<b>✨ Pending suggestion</b>" in queue
    assert "<b>Suggestion #7</b>" in queue
    assert "/add_event 3" not in queue
    assert "/reject_suggestion 3" not in queue


def test_proposed_update_list_formats_action_hints() -> None:
    queue = format_proposed_update_list(
        (
            ProposedEventUpdateRecord(
                id=3,
                event_id="barcelona.42",
                update_type="registration_window",
                current_fields={"registration_status": "unknown"},
                proposed_fields={"registration_status": "open"},
                evidence=("Registration is open.",),
                confidence=0.91,
                status="pending",
                change_summary="Registration update proposed: registration_status.",
            ),
        )
    )

    assert "<b>Update #3</b>" in queue
    assert "<b>Update ID</b>" not in queue
    assert "<b>Event ID</b>: <code>barcelona.42</code>" in queue
    assert "<b>Type</b>: registration_window" in queue
    assert "<b>Confidence</b>: 0.91" in queue
    assert "<b>Fields</b>: registration_status" in queue
    assert "Registration update proposed" not in queue
    assert "/show_update 1" not in queue
    assert "/apply_update 1" not in queue
    assert "/reject_update 1" not in queue


def test_proposed_update_list_keyboard_shows_update_buttons() -> None:
    keyboard = proposed_update_list_keyboard(
        (
            ProposedEventUpdateRecord(
                id=3,
                event_id="barcelona.42",
                update_type="registration_window",
                current_fields={},
                proposed_fields={},
                evidence=(),
                confidence=0.91,
                status="pending",
                change_summary=None,
            ),
        ),
        limit=5,
    )

    button = keyboard.inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == proposed_update_show_callback(3, list_limit=5)


def test_proposed_update_show_keyboard_formats_single_button() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    keyboard = proposed_update_show_keyboard(update)
    button = keyboard.inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == proposed_update_show_callback(3)


def test_list_updates_sends_cards_like_event_list(monkeypatch) -> None:
    updates = (
        ProposedEventUpdateRecord(
            id=3,
            event_id="barcelona.42",
            update_type="registration_window",
            current_fields={},
            proposed_fields={},
            evidence=(),
            confidence=0.91,
            status="pending",
            change_summary="First update.",
        ),
        ProposedEventUpdateRecord(
            id=4,
            event_id="karlsruhe.42",
            update_type="registration_window",
            current_fields={},
            proposed_fields={},
            evidence=(),
            confidence=0.65,
            status="pending",
            change_summary="Second update.",
        ),
    )

    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "list_proposed_event_updates", lambda limit: updates)

    message = FakeMessage()
    asyncio.run(moderator.handle_list_updates(message, SimpleNamespace(args="")))

    assert message.answers == [
        "<b>✨ Pending updates</b>",
        format_proposed_update_card(updates[0]),
        format_proposed_update_card(updates[1]),
    ]
    first_button = message.answer_kwargs[1]["reply_markup"].inline_keyboard[0][0]
    second_button = message.answer_kwargs[2]["reply_markup"].inline_keyboard[0][0]
    assert first_button.text == "Show"
    assert second_button.text == "Show"


def test_list_suggestions_sends_cards_like_update_list(monkeypatch) -> None:
    suggestions = (
        SimpleNamespace(
            id=7,
            event_name="Baden Marathon",
            url="https://www.badenmarathon.de/",
            distances=("marathon",),
            submitter_username="runner",
            submitter_display_name=None,
            submitter_user_id="42",
            note=None,
        ),
        SimpleNamespace(
            id=8,
            event_name="Lisbon Half",
            url="https://www.lisbon-half.example/",
            distances=("half_marathon",),
            submitter_username=None,
            submitter_display_name="Vitaly",
            submitter_user_id="43",
            note="Watch this.",
        ),
    )

    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "list_event_suggestions", lambda limit: suggestions)

    message = FakeMessage()
    asyncio.run(moderator.handle_list_suggestions(message, SimpleNamespace(args="")))

    assert message.answers == [
        "<b>✨ Pending suggestion</b>",
        format_suggestion_card(suggestions[0]),
        format_suggestion_card(suggestions[1]),
    ]
    first_button = message.answer_kwargs[1]["reply_markup"].inline_keyboard[0][0]
    second_button = message.answer_kwargs[2]["reply_markup"].inline_keyboard[0][0]
    assert first_button.text == "Show"
    assert second_button.text == "Show"


def test_list_suggestions_rejects_count_parameter(monkeypatch) -> None:
    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)

    message = FakeMessage()
    asyncio.run(moderator.handle_list_suggestions(message, SimpleNamespace(args="5")))

    assert message.answers == ["Use /list_suggestions without a count."]


def test_next_update_shows_oldest_pending_update(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "list_proposed_event_updates", lambda limit: (update,))
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage()
    asyncio.run(moderator.handle_next_update(message))

    assert message.answers == [format_proposed_update_detail(update)]


def test_next_suggestion_shows_oldest_pending_suggestion(monkeypatch) -> None:
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username="runner",
        submitter_display_name=None,
        submitter_user_id="42",
        note="Please track it.",
    )

    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "list_event_suggestions", lambda limit: (suggestion,))
    monkeypatch.setattr(
        moderator,
        "find_pending_suggestion_by_record_id",
        lambda suggestion_id: suggestion,
    )

    message = FakeMessage()
    asyncio.run(moderator.handle_next_suggestion(message))

    assert message.answers == [format_suggestion_detail(suggestion)]


def test_apply_suggestion_starts_add_event_review(monkeypatch) -> None:
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username="runner",
        submitter_display_name=None,
        submitter_user_id="42",
        note=None,
    )
    captured = {}

    async def fake_require_moderator(message):
        return True

    async def fake_start(message, state, received_suggestion, *, label=None, announce=True):
        captured["suggestion"] = received_suggestion
        captured["label"] = label
        captured["announce"] = announce

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(
        moderator,
        "find_pending_suggestion_by_record_id",
        lambda suggestion_id: suggestion,
    )
    monkeypatch.setattr(moderator, "start_add_event_from_suggestion_record", fake_start)

    asyncio.run(
        moderator.handle_apply_suggestion(
            FakeMessage(),
            SimpleNamespace(args="#7"),
            FakeState(),
        )
    )

    assert captured == {
        "suggestion": suggestion,
        "label": "#7",
        "announce": True,
    }


def test_proposed_update_detail_keyboard_has_review_buttons() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )
    keyboard = proposed_update_detail_keyboard(update, list_limit=5)

    apply_button, partial, reject = keyboard.inline_keyboard[0]
    back = keyboard.inline_keyboard[1][0]
    assert apply_button.text == "Apply"
    assert apply_button.callback_data == "update:apply:3:5"
    assert partial.text == "Partial"
    assert partial.callback_data == proposed_update_partial_callback(3, list_limit=5)
    assert reject.text == "Reject"
    assert reject.callback_data == "update:reject:3:5"
    assert back.text == "Back"
    assert back.callback_data == proposed_update_card_callback(3, list_limit=5)


def test_proposed_update_partial_keyboard_toggles_fields() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={
            "registration_status": "unknown",
            "registration_url": None,
        },
        proposed_fields={
            "registration_status": "open",
            "registration_url": "https://example.com/register",
        },
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    keyboard = proposed_update_partial_keyboard(
        update,
        selected_fields=("registration_status",),
        list_limit=5,
    )

    status_button = keyboard.inline_keyboard[0][0]
    url_button = keyboard.inline_keyboard[1][0]
    confirm, back = keyboard.inline_keyboard[2]
    assert status_button.text == "✅ registration_status"
    assert status_button.callback_data == proposed_update_partial_toggle_callback(
        3,
        "registration_status",
        selected_fields=("registration_status",),
        changed_fields=("registration_status", "registration_url"),
        list_limit=5,
    )
    assert url_button.text == "⬜ registration_url"
    assert confirm.text == "Confirm"
    assert confirm.callback_data == proposed_update_partial_confirm_callback(
        3,
        selected_fields=("registration_status",),
        changed_fields=("registration_status", "registration_url"),
        list_limit=5,
    )
    assert back.text == "Back"


def test_update_partial_confirmation_keyboard_has_confirm_and_cancel() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={"registration_status": "unknown"},
        proposed_fields={"registration_status": "open"},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    keyboard = proposed_update_partial_confirmation_keyboard(
        update,
        selected_fields=("registration_status",),
        list_limit=5,
    )

    confirm, cancel = keyboard.inline_keyboard[0]
    assert confirm.text == "Confirm"
    assert confirm.callback_data == proposed_update_partial_apply_callback(
        3,
        selected_fields=("registration_status",),
        changed_fields=("registration_status",),
        list_limit=5,
    )
    assert cancel.text == "Cancel"
    assert cancel.callback_data == proposed_update_partial_callback(
        3,
        selected_fields=("registration_status",),
        changed_fields=("registration_status",),
        list_limit=5,
    )


def test_show_update_callback_edits_list_message(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage()
    callback = FakeCallback(proposed_update_show_callback(3, list_limit=5), message)
    asyncio.run(moderator.handle_show_update_callback(callback))

    assert message.edits == [format_proposed_update_detail(update)]
    keyboard = message.edit_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Apply"
    assert keyboard.inline_keyboard[0][1].text == "Partial"
    assert keyboard.inline_keyboard[1][0].text == "Back"


def test_partial_update_callback_edits_to_field_selection(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={
            "registration_status": "unknown",
            "registration_url": None,
        },
        proposed_fields={
            "registration_status": "open",
            "registration_url": "https://example.com/register",
        },
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage()
    callback = FakeCallback(proposed_update_partial_callback(3, list_limit=5), message)
    asyncio.run(moderator.handle_partial_update_callback(callback))

    assert message.edits == [format_update_partial_selection(update, selected_fields=())]
    keyboard = message.edit_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "⬜ registration_status"
    assert keyboard.inline_keyboard[-1][0].text == "Back"


def test_partial_update_confirm_callback_edits_to_confirmation(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={
            "registration_status": "unknown",
            "registration_url": None,
        },
        proposed_fields={
            "registration_status": "open",
            "registration_url": "https://example.com/register",
        },
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage()
    callback = FakeCallback(
        proposed_update_partial_confirm_callback(
            3,
            selected_fields=("registration_status",),
            changed_fields=("registration_status", "registration_url"),
            list_limit=5,
        ),
        message,
    )
    asyncio.run(moderator.handle_partial_update_confirm_callback(callback))

    assert message.edits == [
        format_update_partial_confirmation(
            update,
            selected_fields=("registration_status",),
        )
    ]
    keyboard = message.edit_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Confirm"
    assert keyboard.inline_keyboard[0][1].text == "Cancel"


def test_partial_update_apply_callback_replaces_dialog_with_result(monkeypatch) -> None:
    event = TrackedEvent(
        id="barcelona.42",
        public_id="barcelona.42",
        legacy_ids=(),
        search_keywords=(),
        name="Zurich Marató Barcelona",
        city="Barcelona",
        country="Spain",
        timezone="Europe/Madrid",
        distances=("marathon",),
        regions=("global", "eu", "es"),
        collections=(),
        event_date="2027-03-14",
        registration_status="open",
        official_url="https://zurichmaratobarcelona.es/en/",
        registration_url=None,
    )
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={
            "registration_status": "unknown",
            "registration_url": None,
        },
        proposed_fields={
            "registration_status": "open",
            "registration_url": "https://example.com/register",
        },
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )
    reviewed = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields=update.current_fields,
        proposed_fields=update.proposed_fields,
        evidence=(),
        confidence=0.91,
        status="applied_partial",
        change_summary=None,
    )
    follow_up = ProposedEventUpdateRecord(
        id=9,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={"registration_url": None},
        proposed_fields={"registration_url": "https://example.com/register"},
        evidence=("Created from partial apply of update #3.",),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)
    monkeypatch.setattr(
        moderator,
        "partial_apply_proposed_event_update",
        lambda update_id, selected_fields, reviewer_user_id: ProposedEventUpdatePartialApplyResult(
            update=reviewed,
            event=event,
            follow_up_update=follow_up,
            applied_fields=selected_fields,
            remaining_fields=("registration_url",),
        ),
    )

    message = FakeMessage()
    callback = FakeCallback(
        proposed_update_partial_apply_callback(
            3,
            selected_fields=("registration_status",),
            changed_fields=("registration_status", "registration_url"),
        ),
        message,
    )
    asyncio.run(moderator.handle_partial_update_apply_callback(callback))

    assert "Partially applied update <code>#3</code>" in message.edits[0]
    assert "<b>New pending update</b>: <code>#9</code>" in message.edits[0]
    assert "<b>Remaining fields</b>: registration_url" in message.edits[0]
    assert message.edit_kwargs[0]["reply_markup"] is None


def test_update_list_callback_edits_back_to_list(monkeypatch) -> None:
    updates = (
        ProposedEventUpdateRecord(
            id=3,
            event_id="barcelona.42",
            update_type="registration_window",
            current_fields={},
            proposed_fields={},
            evidence=(),
            confidence=0.91,
            status="pending",
            change_summary=None,
        ),
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "list_proposed_event_updates", lambda limit: updates)

    message = FakeMessage()
    callback = FakeCallback(proposed_update_list_callback(5), message)
    asyncio.run(moderator.handle_update_list_callback(callback))

    assert message.edits == ["<b>✨ Pending updates</b>"]
    assert message.edit_kwargs[0]["reply_markup"] is None
    assert message.answers == [format_proposed_update_card(updates[0])]
    keyboard = message.answer_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Show"


def test_suggestion_list_callback_edits_title_and_sends_cards(monkeypatch) -> None:
    suggestions = (
        SimpleNamespace(
            id=7,
            event_name="Baden Marathon",
            url="https://www.badenmarathon.de/",
            distances=("marathon",),
            submitter_username="runner",
            submitter_display_name=None,
            submitter_user_id="42",
            note=None,
        ),
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "list_event_suggestions", lambda limit: suggestions)

    message = FakeMessage()
    callback = FakeCallback(suggestion_list_callback(5), message)
    asyncio.run(moderator.handle_suggestion_list_callback(callback))

    assert message.edits == ["<b>✨ Pending suggestion</b>"]
    assert message.edit_kwargs[0]["reply_markup"] is None
    assert message.answers == [format_suggestion_card(suggestions[0])]
    keyboard = message.answer_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Show"


def test_update_card_callback_edits_back_to_summary_card(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage()
    callback = FakeCallback(proposed_update_card_callback(3, list_limit=5), message)
    asyncio.run(moderator.handle_update_card_callback(callback))

    assert message.edits == [format_proposed_update_card(update)]
    button = message.edit_kwargs[0]["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Show"


def test_proposed_update_confirmation_keyboard_has_confirm_and_cancel() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    apply_keyboard = proposed_update_confirmation_keyboard(update, action="apply", list_limit=5)
    apply_confirm, apply_cancel = apply_keyboard.inline_keyboard[0]
    assert apply_confirm.text == "Confirm"
    assert apply_confirm.callback_data == proposed_update_apply_confirm_callback(
        3,
        list_limit=5,
    )
    assert apply_cancel.text == "Cancel"
    assert apply_cancel.callback_data == proposed_update_show_callback(3, list_limit=5)

    reject_keyboard = proposed_update_confirmation_keyboard(update, action="reject", list_limit=5)
    reject_confirm, reject_cancel = reject_keyboard.inline_keyboard[0]
    assert reject_confirm.text == "Confirm"
    assert reject_confirm.callback_data == proposed_update_reject_confirm_callback(
        3,
        list_limit=5,
    )
    assert reject_cancel.text == "Cancel"
    assert reject_cancel.callback_data == proposed_update_show_callback(3, list_limit=5)


def test_apply_confirm_callback_replaces_dialog_with_result(monkeypatch) -> None:
    event = TrackedEvent(
        id="barcelona.42",
        public_id="barcelona.42",
        legacy_ids=(),
        search_keywords=(),
        name="Zurich Marató Barcelona",
        city="Barcelona",
        country="Spain",
        timezone="Europe/Madrid",
        distances=("marathon",),
        regions=("global", "eu", "es"),
        collections=(),
        event_date="2027-03-14",
        registration_status="open",
        official_url="https://zurichmaratobarcelona.es/en/",
        registration_url=None,
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(
        moderator,
        "approve_proposed_event_update",
        lambda update_id, reviewer_user_id: SimpleNamespace(event=event),
    )

    message = FakeMessage()
    callback = FakeCallback(proposed_update_apply_confirm_callback(3), message)
    asyncio.run(moderator.handle_apply_update_confirm_callback(callback))

    assert "Applied update <code>#3</code> to <b>Zurich Marató Barcelona</b>." in message.edits[0]
    assert "<b>Registration status</b>: open" in message.edits[0]
    assert message.edit_kwargs[0]["reply_markup"] is None


def test_update_review_confirmation_formats_action() -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    text = format_update_review_confirmation(update, action="apply")

    assert "<b>✨ Confirm apply #3</b>" in text
    assert "<b>✨ Confirm apply #3</b>\n\n<b>Event ID</b>" in text
    assert "<b>Update ID</b>" not in text
    assert "<b>Event ID</b>: <code>barcelona.42</code>" in text


def test_apply_update_id_asks_for_confirmation(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={},
        proposed_fields={},
        evidence=(),
        confidence=0.91,
        status="pending",
        change_summary=None,
    )

    def fail_if_applied(*args, **kwargs):
        raise AssertionError("apply should wait for confirmation")

    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)
    monkeypatch.setattr(moderator, "approve_proposed_event_update", fail_if_applied)

    message = FakeMessage()
    asyncio.run(moderator.apply_update_by_record_id_text(message, "#3"))

    assert message.answers == [
        format_update_review_confirmation(update, action="apply")
    ]
    confirm, back = message.answer_kwargs[0]["reply_markup"].inline_keyboard[0]
    assert confirm.text == "Confirm"
    assert back.text == "Cancel"


def test_show_update_state_removes_dialog_keyboard_before_detail(monkeypatch) -> None:
    update = ProposedEventUpdateRecord(
        id=7,
        event_id="berlin-marathon",
        update_type="registration_window",
        current_fields={"registration_status": "unknown"},
        proposed_fields={"registration_status": "open"},
        evidence=(),
        confidence=0.75,
        status="pending",
        change_summary=None,
    )

    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "find_pending_update_by_record_id", lambda update_id: update)

    message = FakeMessage(text="#7")
    state = FakeState()
    asyncio.run(moderator.handle_show_update_id(message, state))

    assert state.cleared
    assert message.answers == [
        moderator.INPUT_RECEIVED_MESSAGE,
        format_proposed_update_detail(update),
    ]
    assert message.answer_kwargs[0]["reply_markup"].remove_keyboard is True
    assert message.answer_kwargs[1]["reply_markup"].inline_keyboard[0][0].text == "Apply"


def test_show_update_state_keeps_dialog_keyboard_on_invalid_id(monkeypatch) -> None:
    async def fake_require_moderator(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)

    message = FakeMessage(text="not-an-update")
    state = FakeState()
    asyncio.run(moderator.handle_show_update_id(message, state))

    assert not state.cleared
    assert message.answers == [
        "<b>💬 Show update</b>\n"
        "Send an update ID.\n"
        "<b>Example</b>: <i>#1</i>"
    ]
    assert [button.text for button in message.answer_kwargs[0]["reply_markup"].keyboard[0]] == [
        "Cancel"
    ]


def test_guided_input_rejects_unexpected_commands(monkeypatch) -> None:
    async def fake_require_moderator(message):
        return True

    def fail_if_lookup_runs(*args, **kwargs):
        raise AssertionError("command should not be treated as an event ID")

    monkeypatch.setattr(moderator, "require_moderator", fake_require_moderator)
    monkeypatch.setattr(moderator, "find_database_event", fail_if_lookup_runs)

    message = FakeMessage(text="/list_suggestions")
    asyncio.run(moderator.handle_edit_event_id(message, FakeState()))

    assert message.answers == [
        "Only /cancel is accepted while this dialog is waiting for input."
    ]
    assert [button.text for button in message.answer_kwargs[0]["reply_markup"].keyboard[0]] == [
        "Cancel"
    ]


def test_moderator_todo_formats_counts() -> None:
    status = format_moderator_status(pending_updates=2, pending_suggestions=5)

    assert status.splitlines() == [
        "<b>✨ Todo</b>",
        "<b>Updates</b>: 2",
        "<b>Suggestion</b>: 5",
    ]


def test_todo_keyboard_links_to_pending_work() -> None:
    keyboard = todo_keyboard(pending_updates=2, pending_suggestions=5)

    assert keyboard is not None
    update_button = keyboard.inline_keyboard[0][0]
    suggestion_button = keyboard.inline_keyboard[1][0]
    assert update_button.text == "List update"
    assert update_button.callback_data == proposed_update_list_callback()
    assert suggestion_button.text == "List suggestion"
    assert suggestion_button.callback_data == suggestion_list_callback()


def test_manual_update_event_runs_registration_scan(monkeypatch) -> None:
    event = TrackedEvent(
        id="barcelona.42",
        public_id="barcelona.42",
        legacy_ids=(),
        search_keywords=(),
        name="Zurich Marató Barcelona",
        city="Barcelona",
        country="Spain",
        timezone="Europe/Madrid",
        distances=("marathon",),
        regions=("global", "eu", "es"),
        collections=(),
        event_date="2027-03-14",
        registration_status="unknown",
        official_url="https://zurichmaratobarcelona.es/en/",
        registration_url=None,
    )

    async def fake_update_registration_window(received_event):
        assert received_event is event
        return SimpleNamespace(
            event_id=event.id,
            registration_status="unknown",
            confidence=0.25,
            registration_open_at=None,
            registration_url=None,
            event_date=event.event_date,
            proposed_update_id=None,
            applied=False,
            evidence="Registration check skipped because no fetcher was configured.",
        )

    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    monkeypatch.setattr(
        moderator,
        "update_registration_window",
        fake_update_registration_window,
    )

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "barcelona.42"))

    assert message.answers[0] == "Running registration scan for <b>Zurich Marató Barcelona</b>..."
    assert "<b>Registration scan</b>" in message.answers[1]
    assert "No registration announcement detected yet." in message.answers[1]


def test_proposed_update_detail_formats_diff_and_evidence() -> None:
    detail = format_proposed_update_detail(
        ProposedEventUpdateRecord(
            id=3,
            event_id="barcelona.42",
            update_type="registration_window",
            current_fields={"registration_status": "unknown", "registration_url": None},
            proposed_fields={
                "registration_status": "open",
                "registration_url": "https://example.com/register",
            },
            evidence=("Fetched page snapshot with status 200.", "Registration is open."),
            confidence=0.91,
            status="pending",
            change_summary="Registration update proposed: registration_status.",
        ),
    )

    assert "<b>✨ Update #3</b>" in detail
    assert "<b>✨ Update #3</b>\n\n<b>Event ID</b>" in detail
    assert "<b>Update ID</b>" not in detail
    assert "<b>Status</b>" not in detail
    assert "Registration update proposed" not in detail
    assert "<b>What's changed</b>" in detail
    assert "- <b>registration_status</b>\n  <s>unknown</s>\n  open" in detail
    assert (
        "- <b>registration_url</b>\n"
        "  <s>unknown</s>\n"
        "  https://example.com/register"
    ) in detail
    assert "<b>Source check</b>" in detail
    assert "/apply_update 1" not in detail
    assert "/reject_update 1" not in detail


def test_parse_queue_number_accepts_visible_numbers() -> None:
    assert parse_queue_number("#7") == 7
    assert parse_queue_number("7") == 7
    assert parse_queue_number("0") is None
    assert parse_queue_number("https://example.com") is None


def test_parse_suggestion_record_id_accepts_stable_handles() -> None:
    assert parse_suggestion_record_id("#7") == 7
    assert parse_suggestion_record_id("7") == 7
    assert parse_suggestion_record_id("") is None
    assert parse_suggestion_record_id("abc") is None


def test_parse_update_show_callback_payload() -> None:
    assert parse_update_show_callback_payload("17") == (17, 10)
    assert parse_update_show_callback_payload("17:5") == (17, 5)
    assert parse_update_show_callback_payload("bad") == (None, 10)
    assert parse_update_show_callback_payload("0") == (None, 10)


def test_parse_suggestion_limit_defaults_and_caps() -> None:
    assert parse_suggestion_limit("") == 10
    assert parse_suggestion_limit("5") == 5
    assert parse_suggestion_limit("999") == EVENT_SUGGESTION_MAX_PENDING_TOTAL

    with pytest.raises(ValueError):
        parse_suggestion_limit("0")
    with pytest.raises(ValueError):
        parse_suggestion_limit("abc")


def test_parse_update_limit_defaults_and_caps() -> None:
    assert parse_update_limit("") == 10
    assert parse_update_limit("5") == 5
    assert parse_update_limit("999") == 30

    with pytest.raises(ValueError):
        parse_update_limit("0")
    with pytest.raises(ValueError):
        parse_update_limit("abc")


def test_edit_event_prompt_excludes_public_id_field() -> None:
    prompt = format_edit_event_prompt(
        SimpleNamespace(
            public_id="berlin.42",
            name="Berlin Marathon",
            event_date="2026-09-27",
            city="Berlin",
            country="Germany",
            timezone="Europe/Berlin",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url="https://example.com/berlin",
            registration_url=None,
            registration_status="unknown",
            registration_open_at=None,
            registration_open_precision="unknown",
            registration_close_at=None,
        )
    )

    assert "<b>ID</b>: <code>berlin.42</code>" in prompt
    assert "Public ID cannot be edited." in prompt
    assert "<code>public_id</code>" not in prompt
    assert "<code>name</code>" in prompt
    assert "<code>event_date</code>" in prompt
    assert "<code>registration_open_at</code>" in prompt
    assert "<code>registration_close_at</code>" in prompt


def test_edit_field_error_uses_separate_copyable_fields() -> None:
    error = format_edit_field_error()

    assert "Fields: <code>name, event_date" not in error
    assert "- <code>name</code>" in error
    assert "- <code>event_date</code>" in error
    assert "- <code>registration_url</code>" in error
    assert "- <code>registration_open_at</code>" in error


def test_archive_event_confirmation_names_exact_event() -> None:
    prompt = format_archive_event_confirmation(
        TrackedEvent(
            id="berlin.42",
            public_id="berlin.42",
            legacy_ids=(),
            search_keywords=(),
            name="Berlin Marathon",
            city="Berlin",
            country="Germany",
            timezone="Europe/Berlin",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            collections=("major",),
            event_date="2026-09-27",
            registration_status="unknown",
            official_url="https://example.com/berlin",
            registration_url=None,
        )
    )

    assert "Berlin Marathon" in prompt
    assert "<b>ID</b>: <code>berlin.42</code>" in prompt
    assert "<b>Confirm archive</b>" in prompt
    assert "Reply <code>archive</code> to confirm" not in prompt
    assert is_archive_event_confirmation("archive")
    assert not is_archive_event_confirmation("yes")


def test_archive_event_confirmation_keyboard_has_archive_and_cancel_buttons() -> None:
    keyboard = archive_event_confirmation_keyboard("berlin.42")

    archive, cancel = keyboard.inline_keyboard[0]
    assert archive.text == "Archive"
    assert archive.callback_data == archive_event_callback("berlin.42")
    assert cancel.text == "Cancel"
    assert cancel.callback_data == "panel:cancel"


def test_delete_event_confirmation_requires_event_id() -> None:
    prompt = format_delete_event_confirmation(
        TrackedEvent(
            id="berlin.42",
            public_id="berlin.42",
            legacy_ids=(),
            search_keywords=(),
            name="Berlin Marathon",
            city="Berlin",
            country="Germany",
            timezone="Europe/Berlin",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            collections=("major",),
            event_date="2026-09-27",
            registration_status="unknown",
            official_url="https://example.com/berlin",
            registration_url=None,
        )
    )

    assert "Berlin Marathon" in prompt
    assert "<b>Delete event</b>" in prompt
    assert "cannot be restored" in prompt
    assert "Reply <code>delete berlin.42</code> to confirm" not in prompt
    assert is_delete_event_confirmation("delete berlin.42", "berlin.42")
    assert is_delete_event_confirmation("DELETE BERLIN.42", "berlin.42")
    assert not is_delete_event_confirmation("delete", "berlin.42")


def test_delete_event_final_confirmation_and_keyboard() -> None:
    event = TrackedEvent(
        id="berlin.42",
        public_id="berlin.42",
        legacy_ids=(),
        search_keywords=(),
        name="Berlin Marathon",
        city="Berlin",
        country="Germany",
        timezone="Europe/Berlin",
        distances=("marathon",),
        regions=("global", "eu", "de"),
        collections=("major",),
        event_date="2026-09-27",
        registration_status="unknown",
        official_url="https://example.com/berlin",
        registration_url=None,
    )

    prompt = format_delete_event_final_confirmation(event)
    keyboard = delete_event_preview_keyboard("berlin.42")

    assert "<b>Confirm permanent deletion</b>" in prompt
    delete_button, cancel = keyboard.inline_keyboard[0]
    assert delete_button.text == "Delete"
    assert delete_button.callback_data == delete_event_callback("berlin.42")
    assert cancel.text == "Cancel"


def test_archived_events_omit_restore_commands() -> None:
    archived = format_archived_events(
        (
            SimpleNamespace(
                event=TrackedEvent(
                    id="berlin.42",
                    public_id="berlin.42",
                    legacy_ids=(),
                    search_keywords=(),
                    name="Berlin Marathon",
                    city="Berlin",
                    country="Germany",
                    timezone="Europe/Berlin",
                    distances=("marathon",),
                    regions=("global", "eu", "de"),
                    collections=("major",),
                    event_date="2026-09-27",
                    registration_status="unknown",
                    official_url="https://example.com/berlin",
                    registration_url=None,
                ),
                removed_at="2026-05-18T18:00:00+00:00",
            ),
        )
    )

    assert "<b>✨ Archived events</b>" in archived
    assert "Berlin Marathon" in archived
    assert "1. Berlin Marathon" not in archived
    assert "<b>ID</b>: <code>berlin.42</code>" in archived
    assert "/restore_event berlin.42" not in archived


def test_archived_event_card_keyboard_has_show_button() -> None:
    archived = (
        SimpleNamespace(
            event=TrackedEvent(
                id="berlin.42",
                public_id="berlin.42",
                legacy_ids=(),
                search_keywords=(),
                name="Berlin Marathon",
                city="Berlin",
                country="Germany",
                timezone="Europe/Berlin",
                distances=("marathon",),
                regions=("global", "eu", "de"),
                collections=("major",),
                event_date="2026-09-27",
                registration_status="unknown",
                official_url="https://example.com/berlin",
                registration_url=None,
            ),
            removed_at="2026-05-18T18:00:00+00:00",
        ),
    )

    keyboard = archived_event_card_keyboard(archived[0])
    button = keyboard.inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == archive_show_callback("berlin.42")


def test_archived_event_detail_keyboard_has_restore_delete_and_back() -> None:
    archived = SimpleNamespace(
        event=TrackedEvent(
            id="berlin.42",
            public_id="berlin.42",
            legacy_ids=(),
            search_keywords=(),
            name="Berlin Marathon",
            city="Berlin",
            country="Germany",
            timezone="Europe/Berlin",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            collections=("major",),
            event_date="2026-09-27",
            registration_status="unknown",
            official_url="https://example.com/berlin",
            registration_url=None,
        ),
        removed_at="2026-05-18T18:00:00+00:00",
    )

    keyboard = archived_event_detail_keyboard(archived, list_limit=5)
    restore, delete = keyboard.inline_keyboard[0]
    back = keyboard.inline_keyboard[1][0]
    assert restore.text == "Restore"
    assert restore.callback_data == restore_event_callback(
        "berlin.42",
        list_limit=5,
    )
    assert delete.text == "Delete"
    assert back.text == "Back"


def test_archive_list_callback_edits_title_and_sends_cards(monkeypatch) -> None:
    archived = (
        SimpleNamespace(
            event=sample_event(),
            removed_at="2026-05-18T18:00:00+00:00",
        ),
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "list_archived_events", lambda limit: archived)

    message = FakeMessage()
    callback = FakeCallback(archive_list_callback(5), message)
    asyncio.run(moderator.handle_archive_list_callback(callback))

    assert message.edits == ["<b>✨ Archived events</b>"]
    assert message.edit_kwargs[0]["reply_markup"] is None
    assert message.answers == [format_archived_event_card(archived[0])]
    button = message.answer_kwargs[0]["reply_markup"].inline_keyboard[0][0]
    assert button.text == "Show"
    assert button.callback_data == archive_show_callback("berlin.42", limit=5)


def test_archive_show_callback_edits_to_archived_detail(monkeypatch) -> None:
    archived = SimpleNamespace(
        event=sample_event(),
        removed_at="2026-05-18T18:00:00+00:00",
    )

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_archived_event", lambda public_id: archived)

    message = FakeMessage()
    callback = FakeCallback(archive_show_callback("berlin.42", limit=5), message)
    asyncio.run(moderator.handle_archive_show_callback(callback))

    assert message.edits == [format_archived_event_detail(archived)]
    assert "Archived event" not in message.edits[0]
    assert message.edits[0].startswith("<b>✨ Berlin Marathon</b>")
    keyboard = message.edit_kwargs[0]["reply_markup"]
    assert keyboard.inline_keyboard[0][0].text == "Restore"
    assert keyboard.inline_keyboard[1][0].text == "Back"


def test_restore_confirm_callback_replaces_dialog_with_result(monkeypatch) -> None:
    event = sample_event()

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "restore_event", lambda event_id: event)

    message = FakeMessage()
    callback = FakeCallback(restore_event_confirm_callback("berlin.42"), message)
    asyncio.run(moderator.handle_restore_event_confirm_callback(callback))

    assert "Restored <b>Berlin Marathon</b> to active tracking." in message.edits[0]
    assert "<b>ID</b>: <code>berlin.42</code>" in message.edits[0]
    assert message.edit_kwargs[0]["reply_markup"] is None


def test_delete_callbacks_replace_preview_confirm_and_result(monkeypatch) -> None:
    event = sample_event()

    async def fake_require_moderator_callback(callback):
        return True

    monkeypatch.setattr(
        moderator,
        "require_moderator_callback",
        fake_require_moderator_callback,
    )
    monkeypatch.setattr(moderator, "find_event_for_delete", lambda event_id: event)
    monkeypatch.setattr(moderator, "delete_event", lambda event_id: event)

    message = FakeMessage()
    callback = FakeCallback(delete_event_callback("berlin.42"), message)
    asyncio.run(moderator.handle_delete_event_callback(callback))

    assert "<b>Confirm permanent deletion</b>" in message.edits[0]
    confirm = message.edit_kwargs[0]["reply_markup"].inline_keyboard[0][0]
    assert confirm.text == "Confirm"
    assert confirm.callback_data == delete_event_confirm_callback("berlin.42")

    confirm_callback = FakeCallback(delete_event_confirm_callback("berlin.42"), message)
    asyncio.run(moderator.handle_delete_event_confirm_callback(confirm_callback))

    assert message.edits[-1] == "Deleted <b>Berlin Marathon</b> permanently."
    assert message.edit_kwargs[-1]["reply_markup"] is None


def test_parse_archive_limit_defaults_and_caps() -> None:
    assert parse_archive_limit("") == 10
    assert parse_archive_limit("5") == 5
    assert parse_archive_limit("999") == ARCHIVE_LIST_MAX_LIMIT

    with pytest.raises(ValueError):
        parse_archive_limit("0")
    with pytest.raises(ValueError):
        parse_archive_limit("abc")


def test_edit_event_parsers_reuse_add_event_validation() -> None:
    assert parse_edit_field("date") == "event_date"
    assert parse_edit_field("event_date") == "event_date"
    assert parse_edit_field("event-date") == "event_date"
    assert parse_edit_field("official_url") == "official_url"
    assert parse_edit_field("registration_url") == "registration_url"
    assert parse_edit_field("registration open") == "registration_open_at"
    assert parse_edit_field("registration_open_precision") == "registration_open_precision"
    assert parse_edit_field("registration close") == "registration_close_at"
    assert parse_edit_field("public_id") is None
    assert parse_edit_value("event_date", "-") is None
    assert parse_edit_value("distances", "42,21") == ("marathon", "half_marathon")
    assert parse_edit_value("regions", "global, eu, de") == ("global", "eu", "de")
    assert parse_edit_value("registration_url", "-") is None
    assert parse_edit_value("registration_status", "sold-out") == "sold_out"
    assert parse_edit_value("registration_open_at", "2026-10-01") == "2026-10-01"
    assert parse_edit_value("registration_open_at", "2026-10-01 09:00") == "2026-10-01 09:00"
    assert parse_edit_value("registration_open_at", "-") is None
    assert parse_edit_value("registration_open_precision", "date-only") == "date_only"

    with pytest.raises(ValueError):
        parse_edit_value("official_url", "example.com")


def test_draft_summary_bolds_headers() -> None:
    summary = format_draft_summary(
        SimpleNamespace(confidence=0.95, evidence="Fetched page snapshot with status 200.")
    )

    assert summary.startswith("<b>✨ Draft extracted from URL</b>")
    assert "<b>Confidence</b>: 0.95" in summary
    assert "<b>Source check</b>" in summary
    assert "Fetched page OK (status 200)." in summary


def test_evidence_display_splits_parameters_onto_lines() -> None:
    evidence = (
        "Fetched page snapshot with status 200. Text hash: dbd36b62c9c8. "
        "Stored snapshot: /Users/in4lio/Pro/run4221/data/page_snapshots/snapshot.json. "
        "Title: Badenmarathon. "
        '"Badenmarathon - Marathon | Halbmarathon" '
        '"42. Baden-Marathon in Karlsruhe - Sei dabei!" '
        '"Der 42. Baden-Marathon findet am 20. September 2026 statt." '
        '"Marathon" "Halbmarathon" "https://www.badenmarathon.de/" '
        "Extractor provider: openai."
    )

    assert format_evidence_for_display(evidence).splitlines() == [
        "<b>Source check</b>",
        "Fetched page OK (status 200).",
        "<b>Snapshot</b>: snapshot.json",
        "<b>Title</b>: Badenmarathon",
        "<b>Provider</b>: openai",
        "",
        "<b>Detected info</b>",
        '"Badenmarathon - Marathon | Halbmarathon"',
        '"42. Baden-Marathon in Karlsruhe - Sei dabei!"',
        '"Der 42. Baden-Marathon findet am 20. September 2026 statt."',
        '"Marathon"',
    ]


def test_evidence_display_limits_quoted_snippets() -> None:
    evidence = (
        '"Snippet 1." "Snippet 2." "Snippet 3." "Snippet 4." "Snippet 5." '
        "Extractor provider: openai."
    )

    lines = format_evidence_for_display(evidence).splitlines()

    assert lines == [
        "<b>Source check</b>",
        "<b>Provider</b>: openai",
        "",
        "<b>Detected info</b>",
        '"Snippet 1."',
        '"Snippet 2."',
        '"Snippet 3."',
        '"Snippet 4."',
    ]


def test_evidence_display_separates_detected_registration_fields() -> None:
    evidence = (
        "Fetched page snapshot with status 200. "
        "Stored snapshot: /Users/in4lio/Pro/run4221/data/page_snapshots/snapshot.json. "
        "Title: Registration for the BMW BERLIN-MARATHON. "
        "Detected registration status: open. "
        "Detected registration URL: "
        "https://www.bmw-berlin-marathon.com/en/registration/"
        "registration-information#page-content. "
        "Detected event date: 2026-09-27. "
        "Registration extractor provider: heuristic."
    )

    assert format_evidence_for_display(evidence).splitlines() == [
        "<b>Source check</b>",
        "Fetched page OK (status 200).",
        "<b>Snapshot</b>: snapshot.json",
        "<b>Title</b>: Registration for the BMW BERLIN-MARATHON",
        "<b>Provider</b>: heuristic",
        "",
        "<b>Detected info</b>",
        "<b>Registration status</b>: open",
        (
            "<b>Registration URL</b>: "
            "https://www.bmw-berlin-marathon.com/en/registration/"
            "registration-information#page-content"
        ),
        "<b>Event date</b>: 2026-09-27",
    ]


def test_evidence_display_marks_blocked_page_warning() -> None:
    evidence = (
        "Fetched page snapshot with status 403. "
        "Stored snapshot: snapshot.json. "
        "Title: Just a moment.... "
        "Page blocked: site protection challenge page (HTTP 403). "
        "Extractor provider: url-fallback."
    )

    assert "⚠️ <b>Warning</b>: site protection challenge page (HTTP 403)" in (
        format_evidence_for_display(evidence)
    )


def test_public_id_must_end_with_supported_distance() -> None:
    assert normalize_public_id("Zurich.42") == "zurich.42"
    with pytest.raises(EventWriteError):
        normalize_public_id("zurich.marathon")


def test_distance_codes_are_extensible(monkeypatch) -> None:
    monkeypatch.setitem(DISTANCE_CODE_TO_KEY, "10", "ten_k")
    monkeypatch.setitem(DISTANCE_LABELS, "ten_k", "10K")

    assert parse_distances("10") == ("ten_k",)
    assert normalize_public_id("berlin.10") == "berlin.10"
    assert ".10" in supported_distance_codes()
