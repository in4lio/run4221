import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from run4221.bot import moderator
from run4221.bot.auth import is_moderator_account, is_moderator_id, is_moderator_username
from run4221.bot.moderator import (
    ARCHIVE_LIST_MAX_LIMIT,
    PARSE_IN_PROGRESS_MESSAGE,
    AddEventStates,
    archive_event_callback,
    archive_event_confirmation_keyboard,
    archive_list_callback,
    archive_show_callback,
    archived_event_card_keyboard,
    archived_event_detail_keyboard,
    channel_draft_keyboard,
    channel_reconciliation_keyboard,
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
    format_event_added_confirmation,
    format_existing_url_warning,
    format_field_value,
    format_moderator_status,
    format_proposed_update_card,
    format_proposed_update_detail,
    format_proposed_update_list,
    format_refresh_outcome,
    format_stored_evidence,
    format_suggestion_card,
    format_suggestion_detail,
    format_suggestion_queue,
    format_suggestion_reject_confirmation,
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
from run4221.config import Settings, parse_moderator_accounts
from run4221.db.repository import (
    EVENT_SUGGESTION_MAX_PENDING_TOTAL,
    EventWriteError,
    ProposedEventUpdatePartialApplyResult,
    ProposedEventUpdateRecord,
    normalize_public_id,
)
from run4221.events import DISTANCE_CODE_TO_KEY, DISTANCE_LABELS, TrackedEvent
from run4221.researcher.engine import EngineConfigError, SourceNotFoundError
from run4221.researcher.schemas import (
    ArtifactReference,
    EventProfileDraft,
    ResearchRunStatus,
)
from run4221.researcher.service import ProfileJobResult, ResearchJobResult

RUN_ID = "2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc"


@pytest.fixture(autouse=True)
def reset_moderator_engine_state():
    def reset() -> None:
        moderator._engine = None
        moderator._parsing_chats.clear()
        moderator._refreshing_events.clear()
        moderator._background_refresh_tasks.clear()

    reset()
    yield
    reset()


class FakeMessage:
    def __init__(self, text: str | None = None, chat_id: int = 100) -> None:
        self.text = text
        self.chat = SimpleNamespace(id=chat_id)
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
        self.data: dict = {}
        self.cleared = False

    async def get_state(self):
        return self.state

    async def set_state(self, state=None) -> None:
        self.state = getattr(state, "state", state)

    async def get_data(self) -> dict:
        return dict(self.data)

    async def update_data(self, **kwargs) -> dict:
        self.data.update(kwargs)
        return dict(self.data)

    async def clear(self) -> None:
        self.state = None
        self.data = {}
        self.cleared = True


class FakeEngine:
    """Canned researcher engine: async profile/refresh_source with real dataclasses."""

    def __init__(
        self,
        *,
        profile_result: ProfileJobResult | None = None,
        refresh_result: ResearchJobResult | None = None,
        profile_error: Exception | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.profile_result = profile_result
        self.refresh_result = refresh_result
        self.profile_error = profile_error
        self.refresh_error = refresh_error
        self.profile_calls: list[str] = []
        self.refresh_calls: list[str] = []

    async def profile(self, url: str) -> ProfileJobResult:
        self.profile_calls.append(url)
        if self.profile_error is not None:
            raise self.profile_error
        assert self.profile_result is not None
        return self.profile_result

    async def refresh_source(self, event_id: str) -> ResearchJobResult:
        self.refresh_calls.append(event_id)
        if self.refresh_error is not None:
            raise self.refresh_error
        assert self.refresh_result is not None
        return self.refresh_result


def install_engine(monkeypatch, engine) -> None:
    monkeypatch.setattr(moderator, "get_engine", lambda: engine)


def artifact_reference(run_id: str = RUN_ID) -> ArtifactReference:
    return ArtifactReference(
        run_id=run_id,
        artifact_name="terminal.json",
        source_url="https://www.badenmarathon.de/",
        content_hash="a" * 64,
    )


def sample_profile_draft(**overrides) -> EventProfileDraft:
    values: dict = {
        "source_url": "https://www.badenmarathon.de/",
        "name": "Baden Marathon",
        "public_id": "karlsruhe.42",
        "city": "Karlsruhe",
        "country": "Germany",
        "timezone": "Etc/UTC",
        "event_date": "2026-09-20",
        "distances": ("marathon",),
        "regions": ("global", "eu", "de"),
        "official_url": "https://www.badenmarathon.de/",
        "registration_url": None,
        "registration_url_candidates": (
            {"url": "https://www.badenmarathon.de/anmeldung/marathon", "link_text": "Anmeldung"},
        ),
        "summary": "The official page confirms the 2026 Baden Marathon.",
        "confidence": 0.92,
    }
    values.update(overrides)
    return EventProfileDraft.model_validate(values)


def profile_result(
    draft: EventProfileDraft | None = None,
    *,
    run_id: str = RUN_ID,
    status: str = "succeeded",
    outcome: str = "profile_completed",
    detail: str | None = None,
    located: bool = False,
) -> ProfileJobResult:
    return ProfileJobResult(
        run_id=run_id,
        status=ResearchRunStatus(status=status, outcome=outcome, detail=detail),
        terminal_reference=artifact_reference(run_id),
        draft=draft,
        located=located,
    )


def sample_validation_error() -> ValidationError:
    try:
        EventProfileDraft.model_validate({})
    except ValidationError as error:
        return error
    raise AssertionError("EventProfileDraft.model_validate({}) must fail")


def refresh_result(
    *,
    run_id: str = RUN_ID,
    status: str = "succeeded",
    outcome: str = "no_change",
    detail: str | None = None,
    queue_reference: str | None = None,
    conflicting_update_id: int | None = None,
) -> ResearchJobResult:
    return ResearchJobResult(
        run_id=run_id,
        status=ResearchRunStatus(status=status, outcome=outcome, detail=detail),
        terminal_reference=artifact_reference(run_id),
        queue_reference=queue_reference,
        conflicting_update_id=conflicting_update_id,
    )


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


def researcher_evidence(
    *,
    summary: str = "Registration is open.",
    source_url: str = "https://example.com/register",
    artifact: str = "20260831T140000Z-page.json",
) -> tuple[str, ...]:
    run_id = "2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc"
    return (
        f"Researcher worker: {summary}",
        "Source check: stored approved event source.",
        f"researcher-decision:v1 run={run_id} artifact=prepared.json sha256={'a' * 64}",
        "researcher-evidence:v1 "
        f"run={run_id} artifact={artifact} sha256={'b' * 12} "
        f"source={source_url} captured_at=2026-08-31T14:00:00+00:00",
    )


def researcher_update(
    *,
    summary: str = "Registration is open.",
    source_url: str = "https://example.com/register",
) -> ProposedEventUpdateRecord:
    return ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={"registration_status": "unknown"},
        proposed_fields={"registration_status": "open"},
        evidence=researcher_evidence(summary=summary, source_url=source_url),
        confidence=0.91,
        status="pending",
        change_summary="Registration is open.",
    )


def test_parse_moderator_accounts_splits_ids_and_usernames() -> None:
    assert parse_moderator_accounts("42,@in4lio, In4lio, 7") == ((42, 7), ("in4lio",))


def test_settings_reject_mutable_moderator_usernames() -> None:
    with pytest.raises(ValueError, match="numeric user IDs"):
        Settings(
            _env_file=None,
            telegram_bot_token="test-token",
            telegram_moderator_accounts="@in4lio",
        )


def test_moderator_auth_checks_unified_accounts() -> None:
    accounts = ((1, 42), ("in4lio",))

    assert is_moderator_account(42, None, accounts)
    assert not is_moderator_account(None, "@In4lio", accounts)
    assert not is_moderator_account(7, "in4lio", accounts)
    assert not is_moderator_account(7, "someone_else", accounts)


def test_moderator_auth_checks_configured_ids() -> None:
    assert is_moderator_id(42, (1, 42))
    assert not is_moderator_id(7, (1, 42))
    assert not is_moderator_id(None, (1, 42))


def test_moderator_auth_never_authorizes_configured_usernames() -> None:
    assert not is_moderator_username("in4lio", ("in4lio",))
    assert not is_moderator_username("@In4lio", ("in4lio",))
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


def test_researcher_suggestion_detail_escapes_and_bounds_provenance() -> None:
    note = "\n".join(
        researcher_evidence(
            summary='Treat <b>this</b> & "that" as evidence.',
            source_url="https://example.com/race?<script>&source=calendar",
            artifact="/srv/run4221/private/evidence.json",
        )
    )
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username=None,
        submitter_display_name=None,
        submitter_user_id=None,
        note=note,
    )

    detail = format_suggestion_detail(suggestion)

    assert "<b>From</b>: Researcher worker" in detail
    assert 'Treat &lt;b&gt;this&lt;/b&gt; &amp; "that" as evidence.' in detail
    assert "<b>Source</b>: https://example.com/race?&lt;script&gt;&amp;source=calendar" in detail
    assert "<b>Captured</b>: 2026-08-31T14:00:00+00:00" in detail
    assert "<b>Run ID</b>: <code>2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc</code>" in detail
    assert "<b>Artifact</b>: evidence.json" in detail
    assert "<b>Hash</b>: <code>bbbbbbbbbbbb</code>" in detail
    assert "/srv/run4221" not in detail
    assert "researcher-evidence:v1" not in detail
    assert len(detail) <= 4096


def test_researcher_detail_renders_all_captured_sources() -> None:
    run_id = "2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc"
    evidence = researcher_evidence() + (
        "researcher-evidence:v1 "
        f"run={run_id} artifact=registration.json sha256={'c' * 64} "
        "source=https://example.com/registration "
        "captured_at=2026-08-31T14:01:00+00:00",
        "researcher-evidence:v1 "
        f"run={run_id} artifact=lottery.json sha256={'d' * 64} "
        "source=https://example.com/lottery "
        "captured_at=2026-08-31T14:02:00+00:00",
    )
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username=None,
        submitter_display_name=None,
        submitter_user_id=None,
        note="\n".join(evidence),
    )

    detail = format_suggestion_detail(suggestion)

    assert "<b>Source</b>: https://example.com/register" in detail
    assert "<b>Source 2</b>: https://example.com/registration" in detail
    assert "<b>Source 3</b>: https://example.com/lottery" in detail
    assert "<b>Artifact 2</b>: registration.json" in detail
    assert "<b>Artifact 3</b>: lottery.json" in detail
    assert "<b>Hash 2</b>: <code>cccccccccccc</code>" in detail
    assert "<b>Hash 3</b>: <code>dddddddddddd</code>" in detail
    assert len(detail) <= 4096


def test_subscriber_suggestion_detail_is_unchanged() -> None:
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

    assert format_suggestion_detail(suggestion) == (
        "<b>✨ Suggestion #7</b>\n\n"
        "<b>Name</b>: Baden Marathon\n"
        "<b>URL</b>: https://www.badenmarathon.de/\n"
        "<b>Distances</b>: <u>42</u>\n"
        "<b>From</b>: @runner\n"
        "<b>Note</b>: Please track it."
    )


def test_non_researcher_system_suggestion_is_not_mislabeled() -> None:
    suggestion = SimpleNamespace(
        id=8,
        event_name="Seeded Marathon",
        url="https://example.com/seeded",
        distances=("marathon",),
        submitter_username=None,
        submitter_display_name=None,
        submitter_user_id=None,
        note="Imported by the seed workflow.",
    )

    detail = format_suggestion_detail(suggestion)

    assert "<b>From</b>: unknown" in detail
    assert "Researcher worker" not in detail


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

    assert message.answers == [format_update_review_confirmation(update, action="apply")]
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
        "<b>💬 Show update</b>\nSend an update ID.\n<b>Example</b>: <i>#1</i>"
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

    assert message.answers == ["Only /cancel is accepted while this dialog is waiting for input."]
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


def test_todo_includes_channel_drafts_when_present() -> None:
    status = format_moderator_status(
        pending_updates=0,
        pending_suggestions=0,
        pending_channel_messages=2,
    )
    keyboard = todo_keyboard(
        pending_updates=0,
        pending_suggestions=0,
        pending_channel_messages=2,
    )

    assert "<b>Channel drafts</b>: 2" in status
    assert keyboard is not None
    assert keyboard.inline_keyboard[0][0].text == "Channel drafts"
    assert keyboard.inline_keyboard[0][0].callback_data == "channel:list"


def test_ambiguous_channel_draft_requires_explicit_reconciliation() -> None:
    keyboard = channel_draft_keyboard(17, status="ambiguous")

    assert len(keyboard.inline_keyboard) == 1
    assert keyboard.inline_keyboard[0][0].text == "Reconcile delivery"
    assert keyboard.inline_keyboard[0][0].callback_data == "channel:reconcile:17"

    reconciliation = channel_reconciliation_keyboard(17)
    assert [row[0].text for row in reconciliation.inline_keyboard] == [
        "Confirmed absent — retry",
        "Already published",
    ]
    assert [row[0].callback_data for row in reconciliation.inline_keyboard] == [
        "channel:retry_confirmed:17",
        "channel:mark_published:17",
    ]


def test_todo_reads_channel_queue_from_configured_database(monkeypatch) -> None:
    async def allow(_message):
        return True

    seen: list[str | None] = []

    def count_channel(*, database_url=None):
        seen.append(database_url)
        return 0

    monkeypatch.setattr(moderator, "require_moderator", allow)
    monkeypatch.setattr(
        moderator,
        "get_settings",
        lambda: SimpleNamespace(database_url="sqlite:///configured.sqlite3"),
    )
    monkeypatch.setattr(moderator, "count_proposed_event_updates", lambda **_kwargs: 0)
    monkeypatch.setattr(moderator, "count_event_suggestions", lambda **_kwargs: 0)
    monkeypatch.setattr(moderator, "count_actionable_channel_messages", count_channel)
    message = FakeMessage()

    asyncio.run(moderator.handle_todo(message))

    assert seen == ["sqlite:///configured.sqlite3"]


def test_manual_update_event_runs_engine_refresh(monkeypatch) -> None:
    event = sample_event("barcelona.42", "Zurich Marató Barcelona")
    engine = FakeEngine(
        refresh_result=refresh_result(
            status="succeeded",
            outcome="no_change",
            detail="The approved source still shows the same registration window.",
        )
    )
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "barcelona.42"))

    assert message.answers[0] == "Running registration check for <b>Zurich Marató Barcelona</b>..."
    assert engine.refresh_calls == ["barcelona.42"]
    assert "<b>Registration check</b>" in message.answers[1]
    assert "No material change detected." in message.answers[1]
    assert "The approved source still shows the same registration window." in message.answers[1]
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in message.answers[1]


def test_update_event_without_active_source_reports_not_found(monkeypatch) -> None:
    event = sample_event()
    engine = FakeEngine(
        refresh_error=SourceNotFoundError("No active research source for event: berlin.42")
    )
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "berlin.42"))

    assert engine.refresh_calls == ["berlin.42"]
    assert message.answers[-1] == (
        "I could not find an active source for event <code>berlin.42</code>."
    )
    assert not moderator._refreshing_events


def test_update_event_validation_error_is_not_reported_as_missing_source(
    monkeypatch,
) -> None:
    event = sample_event()
    engine = FakeEngine(refresh_error=sample_validation_error())
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "berlin.42"))

    failure = message.answers[-1]
    assert failure.startswith("Registration check failed.")
    assert "could not find an active source" not in failure
    assert not moderator._refreshing_events


def test_ae4_update_event_names_created_pending_update(monkeypatch) -> None:
    event = sample_event()
    engine = FakeEngine(
        refresh_result=refresh_result(
            status="succeeded",
            outcome="proposal_created",
            detail="Registration opens on 2026-05-01.",
            queue_reference="proposed_event_update:12",
        )
    )
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "berlin.42"))

    outcome = message.answers[-1]
    assert "Created pending update #12 for moderator review." in outcome
    assert "Registration opens on 2026-05-01." in outcome
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in outcome


def test_ae5_update_event_names_conflicting_pending_update(monkeypatch) -> None:
    event = sample_event()
    engine = FakeEngine(
        refresh_result=refresh_result(
            status="skipped",
            outcome="inconclusive",
            detail="A conflicting pending proposal already exists.",
            conflicting_update_id=7,
        )
    )
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "berlin.42"))

    outcome = message.answers[-1]
    assert "Update #7 is already pending for this event." in outcome
    assert "A conflicting pending proposal already exists." in outcome


def test_repeated_update_event_rejects_second_run(monkeypatch) -> None:
    event = sample_event()
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        def __init__(self) -> None:
            self.refresh_calls: list[str] = []

        async def refresh_source(self, event_id: str) -> ResearchJobResult:
            self.refresh_calls.append(event_id)
            started.set()
            await release.wait()
            return refresh_result()

    engine = BlockingEngine()
    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    install_engine(monkeypatch, engine)

    first = FakeMessage()
    second = FakeMessage()

    async def scenario() -> None:
        task = asyncio.create_task(
            moderator.update_event_registration_by_id(first, "berlin.42")
        )
        await started.wait()
        await moderator.update_event_registration_by_id(second, "berlin.42")
        release.set()
        await task

    asyncio.run(scenario())

    assert engine.refresh_calls == ["berlin.42"]
    assert second.answers == [
        "A registration check is already running for <b>Berlin Marathon</b>."
    ]
    assert "<b>Registration check</b>" in first.answers[-1]
    assert not moderator._refreshing_events


def test_ae3_update_event_reports_engine_config_error(monkeypatch) -> None:
    event = sample_event()

    def broken_engine():
        raise EngineConfigError("Researcher settings are invalid or incomplete.")

    monkeypatch.setattr(moderator, "find_database_event", lambda event_id: event)
    monkeypatch.setattr(moderator, "get_engine", broken_engine)

    message = FakeMessage()
    asyncio.run(moderator.update_event_registration_by_id(message, "berlin.42"))

    assert message.answers == [
        "Researcher engine is not configured: Researcher settings are invalid or incomplete."
    ]


def test_ae1_add_event_url_feeds_engine_draft_into_field_confirmation(monkeypatch) -> None:
    engine = FakeEngine(profile_result=profile_result(sample_profile_draft()))
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()
    asyncio.run(
        moderator.start_add_event_from_url(message, state, "https://www.badenmarathon.de/")
    )

    assert engine.profile_calls == ["https://www.badenmarathon.de/"]
    assert message.answers[0] == "Parsing the event page..."
    summary = message.answers[1]
    assert "<b>✨ Draft extracted from URL</b>" in summary
    assert "<b>Confidence</b>: 0.92" in summary
    assert "<b>Timezone</b>: Europe/Berlin (derived)" in summary
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in summary
    # The draft feeds the guided field confirmation.
    assert state.state == AddEventStates.name.state
    assert state.data["name"] == "Baden Marathon"
    assert state.data["public_id"] == "karlsruhe.42"
    assert state.data["timezone"] == "Europe/Berlin"
    assert state.data["registration_status"] == "unknown"
    assert state.data["registration_url_candidates"] == (
        ("https://www.badenmarathon.de/anmeldung/marathon", "Anmeldung"),
    )
    assert "<b>💬 Event name</b>" in message.answers[2]
    assert "<b>Draft</b>: <u>Baden Marathon</u>" in message.answers[2]
    assert not moderator._parsing_chats


def test_draft_to_state_normalizes_display_text_to_manual_vocabulary() -> None:
    # The engine draft may carry display text; the host maps it through the
    # exact parsers the manual input path uses and drops what they reject.
    draft = sample_profile_draft(
        public_id="Baden Marathon 42",
        distances=("Marathon (42.195 km)",),
        regions=("Europe",),
    )

    state_data = moderator.draft_to_state(draft)

    assert state_data["distances"] == ("marathon",)
    assert state_data["regions"] is None
    assert state_data["public_id"] is None


def test_draft_to_state_keeps_canonical_vocabulary_untouched() -> None:
    state_data = moderator.draft_to_state(sample_profile_draft())

    assert state_data["public_id"] == "karlsruhe.42"
    assert state_data["distances"] == ("marathon",)
    assert state_data["regions"] == ("global", "eu", "de")


def test_draft_to_state_substitutes_captured_page_for_missing_official_url() -> None:
    draft = sample_profile_draft(official_url=None)

    state_data = moderator.draft_to_state(draft)

    assert state_data["official_url"] == "https://www.badenmarathon.de/"


def test_add_event_url_without_draft_continues_with_manual_entry(monkeypatch) -> None:
    engine = FakeEngine(
        profile_result=profile_result(
            None,
            status="skipped",
            outcome="inconclusive",
            detail="Profile page was unusable: site protection challenge page.",
        )
    )
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()
    asyncio.run(
        moderator.start_add_event_from_url(message, state, "https://www.badenmarathon.de/")
    )

    failure = message.answers[1]
    assert "<b>Event page parsing</b>" in failure
    assert "The captured evidence did not support a validated change." in failure
    assert "Profile page was unusable: site protection challenge page." in failure
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in failure
    assert "Continuing with manual entry - fill each field." in message.answers[2]
    # The guided flow continues with an empty manual state: the URL is kept
    # and every draft field waits for the moderator; nothing is invented.
    assert state.state == AddEventStates.name.state
    assert state.data["source_url"] == "https://www.badenmarathon.de/"
    assert state.data["name"] is None
    assert state.data["city"] is None
    assert state.data["official_url"] is None
    assert state.data["registration_url"] is None
    assert state.data["distances"] is None
    assert state.data["registration_url_candidates"] == ()
    assert "<b>💬 Event name</b>" in message.answers[3]
    assert not moderator._parsing_chats


def test_manual_entry_after_failed_parse_still_cancels(monkeypatch) -> None:
    engine = FakeEngine(
        profile_result=profile_result(
            None,
            status="failed",
            outcome="inconclusive",
            detail="Profile page capture failed (PageFetchError).",
        )
    )
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()

    async def scenario() -> None:
        await moderator.start_add_event_from_url(
            message, state, "https://www.badenmarathon.de/"
        )
        assert state.state == AddEventStates.name.state
        await moderator.handle_cancel(FakeMessage(text="/cancel"), state)

    asyncio.run(scenario())

    assert state.state is None
    assert state.data == {}
    assert not moderator._parsing_chats


def test_concurrent_add_event_starts_run_exactly_one_parse(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        def __init__(self) -> None:
            self.profile_calls: list[str] = []

        async def profile(self, url: str) -> ProfileJobResult:
            self.profile_calls.append(url)
            started.set()
            await release.wait()
            return profile_result(sample_profile_draft())

    async def yielding_warn(message, url) -> None:
        # Force a real suspension point before the engine call so a racing
        # second start would sneak past a non-atomic check-then-add guard.
        await asyncio.sleep(0)

    engine = BlockingEngine()
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "warn_existing_url_events", yielding_warn)

    first_message, second_message = FakeMessage(), FakeMessage()
    first_state, second_state = FakeState(), FakeState()

    async def scenario() -> None:
        first = asyncio.create_task(
            moderator.start_add_event_from_url(
                first_message, first_state, "https://www.badenmarathon.de/"
            )
        )
        second = asyncio.create_task(
            moderator.start_add_event_from_url(
                second_message, second_state, "https://www.badenmarathon.de/"
            )
        )
        await started.wait()
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(scenario())

    assert engine.profile_calls == ["https://www.badenmarathon.de/"]
    assert second_message.answers == [PARSE_IN_PROGRESS_MESSAGE]
    assert first_state.state == AddEventStates.name.state
    assert not moderator._parsing_chats


def test_distance_step_drops_article_registration_fallback(monkeypatch) -> None:
    async def allow(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", allow)
    state = FakeState()
    state.state = AddEventStates.distance.state
    state.data = {
        "name": "Baden Marathon",
        "registration_url": "https://baden.example/news/2026/03/05/race-day-recap",
        "registration_url_candidates": (),
    }

    message = FakeMessage(text="42,21")
    asyncio.run(moderator.handle_add_event_distance(message, state))

    # A dated-article draft URL is dropped even for a multi-distance event;
    # the moderator fills the field instead of inheriting a guess.
    assert state.data["distances"] == ("marathon", "half_marathon")
    assert state.data["registration_url"] is None


def test_distance_step_keeps_clean_registration_fallback(monkeypatch) -> None:
    async def allow(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", allow)
    state = FakeState()
    state.state = AddEventStates.distance.state
    state.data = {
        "name": "Baden Marathon",
        "registration_url": "https://baden.example/anmeldung",
        "registration_url_candidates": (),
    }

    message = FakeMessage(text="42,21")
    asyncio.run(moderator.handle_add_event_distance(message, state))

    assert state.data["registration_url"] == "https://baden.example/anmeldung"


def test_ae6_cancel_during_parse_discards_draft_without_resurrection(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        def __init__(self) -> None:
            self.profile_calls: list[str] = []

        async def profile(self, url: str) -> ProfileJobResult:
            self.profile_calls.append(url)
            started.set()
            await release.wait()
            return profile_result(sample_profile_draft())

    engine = BlockingEngine()
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()

    async def scenario() -> None:
        task = asyncio.create_task(
            moderator.start_add_event_from_url(
                message, state, "https://www.badenmarathon.de/"
            )
        )
        await started.wait()
        await moderator.handle_cancel(FakeMessage(text="/cancel"), state)
        release.set()
        await task

    asyncio.run(scenario())

    assert state.state is None
    assert state.data == {}
    assert all("Draft extracted from URL" not in text for text in message.answers)
    assert not moderator._parsing_chats


def test_ae6_second_url_and_add_event_are_rejected_during_parse(monkeypatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        def __init__(self) -> None:
            self.profile_calls: list[str] = []

        async def profile(self, url: str) -> ProfileJobResult:
            self.profile_calls.append(url)
            started.set()
            await release.wait()
            return profile_result(sample_profile_draft())

    async def allow(message):
        return True

    engine = BlockingEngine()
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())
    monkeypatch.setattr(moderator, "require_moderator", allow)

    message = FakeMessage()
    state = FakeState()
    second_url_message = FakeMessage(text="https://two.example/")
    add_event_message = FakeMessage(text="/add_event https://two.example/")

    async def scenario() -> None:
        task = asyncio.create_task(
            moderator.start_add_event_from_url(
                message, state, "https://www.badenmarathon.de/"
            )
        )
        await started.wait()
        # A second URL routed by the parsing-state filter is rejected.
        await moderator.handle_add_event_parsing_input(second_url_message)
        # /add_event does not restart the flow while the parse is in flight.
        await moderator.handle_add_event(
            add_event_message,
            state,
            SimpleNamespace(args="https://two.example/"),
        )
        # A direct second start (e.g. suggestion Apply) is also rejected.
        await moderator.start_add_event_from_url(
            FakeMessage(), state, "https://three.example/"
        )
        release.set()
        await task

    asyncio.run(scenario())

    assert engine.profile_calls == ["https://www.badenmarathon.de/"]
    assert "Still parsing the event page." in second_url_message.answers[0]
    assert add_event_message.answers == [PARSE_IN_PROGRESS_MESSAGE]
    # The original parse still completed into the guided flow.
    assert state.state == AddEventStates.name.state
    assert state.data["name"] == "Baden Marathon"


def test_ae6_cancel_then_restart_is_blocked_and_stale_token_discards_draft(
    monkeypatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingEngine:
        def __init__(self) -> None:
            self.profile_calls: list[str] = []

        async def profile(self, url: str) -> ProfileJobResult:
            self.profile_calls.append(url)
            started.set()
            await release.wait()
            return profile_result(sample_profile_draft(name="URL One Marathon"))

    async def allow(message):
        return True

    engine = BlockingEngine()
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())
    monkeypatch.setattr(moderator, "require_moderator", allow)

    message = FakeMessage()
    state = FakeState()
    restart_message = FakeMessage(text="/add_event https://two.example/")

    async def scenario() -> None:
        task = asyncio.create_task(
            moderator.start_add_event_from_url(
                message, state, "https://one.example/"
            )
        )
        await started.wait()
        await moderator.handle_cancel(FakeMessage(text="/cancel"), state)
        # Restart right after /cancel: the in-flight marker still blocks it.
        await moderator.handle_add_event(
            restart_message,
            state,
            SimpleNamespace(args="https://two.example/"),
        )
        # Simulate a second flow owning the parsing state with its own token:
        # URL one's stale token must never hand its draft to this flow.
        state.state = AddEventStates.parsing.state
        state.data = {"parse_token": "another-flows-token"}
        release.set()
        await task

    asyncio.run(scenario())

    assert engine.profile_calls == ["https://one.example/"]
    assert restart_message.answers == [PARSE_IN_PROGRESS_MESSAGE]
    # URL one's draft was discarded: the other flow's state stayed untouched.
    assert state.state == AddEventStates.parsing.state
    assert state.data == {"parse_token": "another-flows-token"}
    assert all("URL One Marathon" not in text for text in message.answers)
    assert not moderator._parsing_chats


def test_ae3_add_flow_reports_engine_config_error_and_leaves_flow_clean(
    monkeypatch,
) -> None:
    def broken_engine():
        raise EngineConfigError("Researcher prompt could not be loaded.")

    monkeypatch.setattr(moderator, "get_engine", broken_engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()
    asyncio.run(
        moderator.start_add_event_from_url(message, state, "https://example.com/race")
    )

    assert message.answers == [
        "Researcher engine is not configured: Researcher prompt could not be loaded."
    ]
    assert state.state == "active"
    assert state.data == {}
    assert not moderator._parsing_chats


def test_ae3_non_ai_commands_serve_without_engine(monkeypatch) -> None:
    def broken_engine():
        raise EngineConfigError("Researcher settings are invalid or incomplete.")

    async def allow(message):
        return True

    monkeypatch.setattr(moderator, "get_engine", broken_engine)
    monkeypatch.setattr(moderator, "require_moderator", allow)
    monkeypatch.setattr(moderator, "list_proposed_event_updates", lambda limit: ())

    message = FakeMessage()
    asyncio.run(moderator.handle_list_updates(message, SimpleNamespace(args="")))

    assert message.answers == ["No pending updates."]


def prepared_add_event_state() -> FakeState:
    state = FakeState()
    state.state = AddEventStates.registration_close_at.state
    state.data = {
        "source_url": "https://example.com/berlin",
        "name": "Berlin Marathon",
        "public_id": "berlin.42",
        "city": "Berlin",
        "country": "Germany",
        "timezone": "Europe/Berlin",
        "event_date": "2026-09-27",
        "distances": ("marathon",),
        "regions": ("global", "eu", "de"),
        "official_url": "https://example.com/berlin",
        "registration_url": None,
        "registration_status": "unknown",
        "registration_open_at": None,
        "registration_open_precision": "unknown",
    }
    return state


def test_ae7_create_event_fires_single_background_refresh(monkeypatch) -> None:
    event = sample_event()
    engine = FakeEngine(
        refresh_result=refresh_result(
            status="succeeded",
            outcome="proposal_created",
            detail="Registration opens on 2026-05-01.",
            queue_reference="proposed_event_update:9",
        )
    )
    added_events = []

    async def allow(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", allow)
    monkeypatch.setattr(
        moderator,
        "add_event",
        lambda event_create: added_events.append(event_create) or event,
    )
    monkeypatch.setattr(moderator, "list_channel_messages", lambda **kwargs: ())
    install_engine(monkeypatch, engine)

    message = FakeMessage(text="-")
    state = prepared_add_event_state()

    async def scenario() -> None:
        await moderator.handle_add_event_registration_close_at(message, state)
        tasks = tuple(moderator._background_refresh_tasks)
        assert len(tasks) == 1
        await asyncio.gather(*tasks)

    asyncio.run(scenario())

    assert len(added_events) == 1
    assert engine.refresh_calls == ["berlin.42"]
    assert "Event added." in message.answers[0]
    assert any(
        "Started the first registration check in the background." in text
        for text in message.answers
    )
    follow_up = message.answers[-1]
    assert "<b>Registration check</b>" in follow_up
    assert "Created pending update #9 for moderator review." in follow_up
    assert not moderator._refreshing_events
    assert not moderator._background_refresh_tasks


def test_ae7_background_refresh_failure_keeps_created_event(monkeypatch) -> None:
    event = sample_event()
    engine = FakeEngine(refresh_error=RuntimeError("provider exploded"))
    added_events = []

    async def allow(message):
        return True

    monkeypatch.setattr(moderator, "require_moderator", allow)
    monkeypatch.setattr(
        moderator,
        "add_event",
        lambda event_create: added_events.append(event_create) or event,
    )
    monkeypatch.setattr(moderator, "list_channel_messages", lambda **kwargs: ())
    install_engine(monkeypatch, engine)

    message = FakeMessage(text="-")
    state = prepared_add_event_state()

    async def scenario() -> None:
        await moderator.handle_add_event_registration_close_at(message, state)
        await asyncio.gather(*tuple(moderator._background_refresh_tasks))

    asyncio.run(scenario())

    assert len(added_events) == 1
    assert engine.refresh_calls == ["berlin.42"]
    follow_up = message.answers[-1]
    assert "The background registration check failed." in follow_up
    assert "provider exploded" in follow_up
    assert not moderator._refreshing_events


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
    assert ("- <b>registration_url</b>\n  <s>unknown</s>\n  https://example.com/register") in detail
    # AE10: legacy rows keep their stored evidence lines verbatim.
    assert "<b>Evidence</b>" in detail
    assert "Fetched page snapshot with status 200." in detail
    assert "Registration is open." in detail
    assert "/apply_update 1" not in detail
    assert "/reject_update 1" not in detail


def test_proposed_update_detail_formats_explicit_field_clear() -> None:
    detail = format_proposed_update_detail(
        ProposedEventUpdateRecord(
            id=4,
            event_id="berlin.42",
            update_type="registration_window",
            current_fields={"registration_url": "https://example.com/kids-and-youth/mini-marathon"},
            proposed_fields={"registration_url": None},
            evidence=("The saved URL belongs to a child event.",),
            confidence=0.99,
            status="pending",
            change_summary="Clear mismatched registration URL.",
        )
    )

    assert "- <b>registration_url</b>" in detail
    assert "<s>https://example.com/kids-and-youth/mini-marathon</s>" in detail
    assert "<i>clear</i>" in detail


def test_researcher_update_detail_is_bounded_and_keeps_actions() -> None:
    long_url = "https://example.com/register?" + ("x=<tag>&" * 500)
    update = ProposedEventUpdateRecord(
        id=3,
        event_id="barcelona.42",
        update_type="registration_window",
        current_fields={"registration_url": "https://example.com/old"},
        proposed_fields={"registration_url": long_url},
        evidence=researcher_evidence(
            summary="<script>approve it</script> " + ("hostile & evidence " * 200),
            source_url=long_url,
        ),
        confidence=0.91,
        status="pending",
        change_summary="Registration URL changed.",
    )

    detail = format_proposed_update_detail(update)
    keyboard = proposed_update_detail_keyboard(update)

    assert len(detail) <= 4096
    assert "<script>" not in detail
    assert "&lt;script&gt;approve it&lt;/script&gt;" in detail
    assert [button.text for button in keyboard.inline_keyboard[0]] == [
        "Apply",
        "Partial",
        "Reject",
    ]


def test_researcher_update_detail_shows_validated_field_support_and_conflicts() -> None:
    update = researcher_update()
    update = replace(
        update,
        evidence=update.evidence
        + (
            "Researcher field support: registration_status <- "
            "page_snapshot-status.json#bbbbbbbbbbbb",
            "Researcher conflict: event_date <- "
            "page_snapshot-overview.json#cccccccccccc, "
            "page_snapshot-status.json#bbbbbbbbbbbb | "
            "Overview and registration page use different date wording.",
        ),
    )

    detail = format_proposed_update_detail(update)

    assert (
        "<b>Field support</b>: registration_status &lt;- "
        "page_snapshot-status.json#bbbbbbbbbbbb"
    ) in detail
    assert (
        "<b>Conflict</b>: event_date &lt;- "
        "page_snapshot-overview.json#cccccccccccc, "
        "page_snapshot-status.json#bbbbbbbbbbbb | "
        "Overview and registration page use different date wording."
    ) in detail


def test_researcher_update_detail_bounds_complete_maximal_provenance() -> None:
    run_id = "2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc"
    fields = (
        "registration_status",
        "registration_open_at",
        "registration_open_precision",
        "registration_close_at",
        "registration_url",
        "event_date",
    )
    evidence = list(
        researcher_evidence(
            summary="Official evidence " * 100,
            source_url="https://example.com/register?" + "source=official&" * 100,
        )
    )
    evidence.extend(
        "researcher-evidence:v1 "
        f"run={run_id} artifact=evidence-{index}.json sha256={'c' * 64} "
        f"source=https://example.com/evidence/{index}?{'x=1&' * 100} "
        f"captured_at=2026-08-31T14:0{index}:00+00:00"
        for index in (1, 2)
    )
    evidence.extend(
        f"Researcher field support: {field} <- E{index}.json#{str(index) * 12}"
        for index, field in enumerate(fields, start=1)
    )
    evidence.extend(
        f"Researcher conflict: unrelated-{index} <- E1.json#111111111111, "
        f"E2.json#222222222222 | {'conflicting context ' * 40}"
        for index in range(4)
    )
    update = ProposedEventUpdateRecord(
        id=99,
        event_id="badenmarathon.42",
        update_type="registration_window",
        current_fields={
            "registration_status": "unknown",
            "registration_open_at": "2026-01-01",
            "registration_open_precision": "unknown",
            "registration_close_at": "2026-02-01",
            "registration_url": "https://example.com/old?" + "old=1&" * 100,
            "event_date": "2026-09-20",
        },
        proposed_fields={
            "registration_status": "closed",
            "registration_open_at": "2026-05-01",
            "registration_open_precision": "date_only",
            "registration_close_at": "2026-11-01",
            "registration_url": "https://example.com/new?" + "new=1&" * 100,
            "event_date": "2027-09-19",
        },
        evidence=tuple(evidence),
        confidence=0.97,
        status="pending",
        change_summary="Registration fields were refreshed from official sources.",
    )

    detail = format_proposed_update_detail(update)

    assert len(detail) <= 4_096
    assert detail.count("<blockquote>") == detail.count("</blockquote>") == 1
    assert detail.count("<code>") == detail.count("</code>")
    assert detail.count("<b>Field support</b>") == len(fields)
    for index, field in enumerate(fields, start=1):
        assert f"{field} &lt;- E{index}.json#{str(index) * 12}" in detail
    assert f"<b>Run ID</b>: <code>{run_id}</code>" in detail


def test_researcher_provenance_is_repeated_in_all_confirmations() -> None:
    update = researcher_update()
    suggestion = SimpleNamespace(
        id=7,
        event_name="Baden Marathon",
        url="https://www.badenmarathon.de/",
        distances=("marathon",),
        submitter_username=None,
        submitter_display_name=None,
        submitter_user_id=None,
        note="\n".join(researcher_evidence()),
    )

    confirmations = (
        format_update_review_confirmation(update, action="apply"),
        format_update_review_confirmation(update, action="reject"),
        format_update_partial_confirmation(
            update,
            selected_fields=("registration_status",),
        ),
        format_suggestion_reject_confirmation(suggestion, label="#7"),
        format_event_added_confirmation(
            from_suggestion=True,
            suggestion_note=suggestion.note,
        ),
    )

    for confirmation in confirmations:
        assert "<blockquote><b>Source check</b>" in confirmation
        assert "<b>Evidence</b>: Registration is open." in confirmation
        assert "<b>Source</b>: https://example.com/register" in confirmation
        assert "<b>Captured</b>: 2026-08-31T14:00:00+00:00" in confirmation
        assert "<b>Run ID</b>: <code>2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc</code>" in confirmation
        assert "<b>Artifact</b>: 20260831T140000Z-page.json" in confirmation
        assert "<b>Hash</b>: <code>bbbbbbbbbbbb</code>" in confirmation
        assert len(confirmation) <= 4096

    assert format_event_added_confirmation(from_suggestion=False) == "Event added."
    assert (
        format_event_added_confirmation(
            from_suggestion=True,
            suggestion_note="Please track it.",
        )
        == "Event added. The source suggestion was removed from the pending queue."
    )


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


def test_draft_summary_renders_typed_profile_fields() -> None:
    draft = sample_profile_draft()

    summary = format_draft_summary(draft, run_id=RUN_ID, timezone="Europe/Berlin")

    assert summary.startswith("<b>✨ Draft extracted from URL</b>")
    assert "<b>Confidence</b>: 0.92" in summary
    assert "<b>Name</b>: Baden Marathon" in summary
    assert "<b>Captured page</b>: https://www.badenmarathon.de/" in summary
    assert "<b>Timezone</b>: Europe/Berlin (derived)" in summary
    assert "<b>Summary</b>: The official page confirms the 2026 Baden Marathon." in summary
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in summary
    assert "I will ask you to confirm or correct each field." in summary
    assert "Source check" not in summary


def test_draft_summary_omits_derived_label_for_model_timezone() -> None:
    draft = sample_profile_draft(timezone="Europe/Berlin")

    summary = format_draft_summary(draft, run_id=RUN_ID, timezone="Europe/Berlin")

    assert "(derived)" not in summary


def test_draft_summary_names_locate_provenance_only_when_located() -> None:
    draft = sample_profile_draft()

    located = format_draft_summary(
        draft,
        run_id=RUN_ID,
        timezone="Europe/Berlin",
        located=True,
    )
    direct = format_draft_summary(draft, run_id=RUN_ID, timezone="Europe/Berlin")

    assert (
        "Located via web search: www.badenmarathon.de "
        "— verify this is the official site." in located
    )
    assert "Located via web search" not in direct


def test_add_event_flow_surfaces_locate_provenance(monkeypatch) -> None:
    engine = FakeEngine(
        profile_result=profile_result(sample_profile_draft(), located=True)
    )
    install_engine(monkeypatch, engine)
    monkeypatch.setattr(moderator, "list_events_by_url", lambda url: ())

    message = FakeMessage()
    state = FakeState()
    asyncio.run(
        moderator.start_add_event_from_url(
            message, state, "https://news.example/marathon-report"
        )
    )

    summary = message.answers[1]
    assert "<b>✨ Draft extracted from URL</b>" in summary
    assert "Located via web search: www.badenmarathon.de" in summary


def test_refresh_outcome_maps_provider_failures_to_limits_wording() -> None:
    outcome = format_refresh_outcome(
        refresh_result(
            status="failed",
            outcome="inconclusive",
            detail="OpenAI rate limit was reached.",
        )
    )

    assert "check its configuration or usage limits" in outcome
    assert "OpenAI rate limit was reached." in outcome
    assert f"<b>Run ID</b>: <code>{RUN_ID}</code>" in outcome


def test_refresh_outcome_free_text_failure_detail_stays_generic() -> None:
    # Only the typed provider error details map to the limits wording; free
    # detail text that merely mentions a timeout must not.
    outcome = format_refresh_outcome(
        refresh_result(
            status="failed",
            outcome="inconclusive",
            detail="Page fetch timed out",
        )
    )

    assert "The check failed before reaching a decision." in outcome
    assert "configuration or usage limits" not in outcome


def test_refresh_outcome_maps_bare_provider_error_codes_to_limits_wording() -> None:
    outcome = format_refresh_outcome(
        refresh_result(status="failed", outcome="inconclusive", detail="rate_limit")
    )

    assert "check its configuration or usage limits" in outcome


def test_refresh_outcome_maps_queue_full_and_budget_caps() -> None:
    queue_full = format_refresh_outcome(
        refresh_result(
            status="capped",
            outcome="inconclusive",
            detail="The researcher proposal queue is full.",
        )
    )
    budget = format_refresh_outcome(
        refresh_result(
            status="capped",
            outcome="inconclusive",
            detail="Refresh continuation budget was exhausted.",
        )
    )

    assert "The moderation queue is full. Review pending updates, then retry." in queue_full
    assert "The check stopped at its run budget." in budget


def test_refresh_outcome_keeps_skip_detail_verbatim_and_escaped() -> None:
    outcome = format_refresh_outcome(
        refresh_result(
            status="skipped",
            outcome="inconclusive",
            detail="Approved source was unusable: <script> challenge page.",
        )
    )

    assert "The captured evidence did not support a validated change." in outcome
    assert "Approved source was unusable: &lt;script&gt; challenge page." in outcome
    assert "<script>" not in outcome


def test_stored_evidence_renders_lines_verbatim() -> None:
    rendered = format_stored_evidence(
        (
            "Fetched page snapshot with status 200.",
            'Registration is <b>open</b> & "confirmed".',
        )
    )

    assert rendered.splitlines() == [
        "<b>Evidence</b>",
        "Fetched page snapshot with status 200.",
        'Registration is &lt;b&gt;open&lt;/b&gt; &amp; "confirmed".',
    ]


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
