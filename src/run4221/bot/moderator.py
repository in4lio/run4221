from __future__ import annotations

import re
from datetime import date, datetime
from html import escape

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from run4221.ai.event_extractor import (
    EventDraft,
    extract_event_draft_from_url,
    select_registration_url_for_distances,
)
from run4221.ai.provider_factory import ExtractorProviderConfigError, get_extractor_provider
from run4221.ai.registration_window import (
    RegistrationWindowUpdateResult,
    update_registration_window,
)
from run4221.bot.auth import is_moderator_account, require_moderator
from run4221.bot.formatting import (
    bounded_html_escape,
    format_bounded_field_line,
    format_event_detail,
    format_field_line,
    format_major_title,
    format_researcher_source_check,
    parse_researcher_provenance,
)
from run4221.bot.keyboards import (
    dialog_keyboard,
    distance_dialog_keyboard,
    is_cancel_text,
    remove_dialog_keyboard,
)
from run4221.bot.prompts import waiting_prompt
from run4221.config import get_settings
from run4221.db.repository import (
    EVENT_SUGGESTION_MAX_PENDING_TOTAL,
    REGISTRATION_OPEN_PRECISIONS,
    REGISTRATION_STATUSES,
    EventCreate,
    EventUpdate,
    EventWriteError,
    ProposedEventUpdateRecord,
    add_event,
    add_event_from_suggestion,
    approve_proposed_event_update,
    archive_event,
    count_event_suggestions,
    count_proposed_event_updates,
    delete_event,
    find_archived_event,
    get_event_suggestion,
    get_proposed_event_update,
    list_archived_events,
    list_event_suggestions,
    list_events_by_url,
    list_proposed_event_updates,
    partial_apply_proposed_event_update,
    reject_proposed_event_update,
    restore_event,
    update_event,
    update_event_suggestion_status,
)
from run4221.db.repository import find_event as find_database_event
from run4221.events import (
    DISTANCE_CODE_TO_KEY,
    DISTANCE_KEY_TO_CODE,
    DISTANCE_LABELS,
    REGION_LABELS,
    TrackedEvent,
    normalize_event_id,
)

router = Router(name="moderator")

ARCHIVE_LIST_DEFAULT_LIMIT = 10
ARCHIVE_LIST_MAX_LIMIT = 30
UPDATE_LIST_DEFAULT_LIMIT = 10
UPDATE_LIST_MAX_LIMIT = 30
UPDATE_LIST_CALLBACK_PREFIX = "update:list:"
UPDATE_CARD_CALLBACK_PREFIX = "update:card:"
UPDATE_SHOW_CALLBACK_PREFIX = "update:show:"
UPDATE_APPLY_CALLBACK_PREFIX = "update:apply:"
UPDATE_REJECT_CALLBACK_PREFIX = "update:reject:"
UPDATE_PARTIAL_CALLBACK_PREFIX = "update:partial:"
UPDATE_PARTIAL_TOGGLE_CALLBACK_PREFIX = "update:partial_toggle:"
UPDATE_PARTIAL_CONFIRM_CALLBACK_PREFIX = "update:partial_confirm:"
UPDATE_PARTIAL_APPLY_CALLBACK_PREFIX = "update:partial_apply:"
UPDATE_APPLY_CONFIRM_CALLBACK_PREFIX = "update:confirm_apply:"
UPDATE_REJECT_CONFIRM_CALLBACK_PREFIX = "update:confirm_reject:"
UPDATE_REVIEW_CANCEL_CALLBACK = "update:review_cancel"
INPUT_RECEIVED_MESSAGE = "Got it."
PANEL_CANCEL_CALLBACK = "panel:cancel"
SUGGESTION_LIST_DEFAULT_LIMIT = 10
ARCHIVE_EVENT_CALLBACK_PREFIX = "event:archive:"
ARCHIVE_LIST_CALLBACK_PREFIX = "archive:list:"
ARCHIVE_SHOW_CALLBACK_PREFIX = "archive:show:"
RESTORE_EVENT_CALLBACK_PREFIX = "event:restore:"
RESTORE_CONFIRM_CALLBACK_PREFIX = "event:confirm_restore:"
DELETE_EVENT_PREVIEW_CALLBACK_PREFIX = "event:delete_preview:"
DELETE_EVENT_CALLBACK_PREFIX = "event:delete:"
DELETE_EVENT_CONFIRM_CALLBACK_PREFIX = "event:confirm_delete:"
SUGGESTION_ADD_CALLBACK_PREFIX = "suggestion:add:"
SUGGESTION_CARD_CALLBACK_PREFIX = "suggestion:card:"
SUGGESTION_LIST_CALLBACK_PREFIX = "suggestion:list:"
SUGGESTION_SHOW_CALLBACK_PREFIX = "suggestion:show:"
SUGGESTION_REMOVE_CALLBACK_PREFIX = "suggestion:remove:"
SUGGESTION_REMOVE_CONFIRM_CALLBACK_PREFIX = "suggestion:confirm_remove:"


class AddEventStates(StatesGroup):
    source_url = State()
    name = State()
    public_id = State()
    city = State()
    country = State()
    timezone = State()
    event_date = State()
    distance = State()
    regions = State()
    official_url = State()
    registration_url = State()
    registration_status = State()
    registration_open_at = State()
    registration_open_precision = State()
    registration_close_at = State()


class ArchiveEventStates(StatesGroup):
    event_id = State()
    confirm = State()


class DeleteEventStates(StatesGroup):
    event_id = State()
    confirm = State()


class ApplySuggestionStates(StatesGroup):
    suggestion_id = State()


class RejectSuggestionStates(StatesGroup):
    suggestion_id = State()


class ShowSuggestionStates(StatesGroup):
    suggestion_id = State()


class RestoreEventStates(StatesGroup):
    event_id = State()


class EditEventStates(StatesGroup):
    event_id = State()
    field = State()
    value = State()


class UpdateEventStates(StatesGroup):
    event_id = State()


class ShowUpdateStates(StatesGroup):
    update_number = State()


class ApplyUpdateStates(StatesGroup):
    update_number = State()


class RejectUpdateStates(StatesGroup):
    update_number = State()


FIELD_LABELS = {
    "name": "Event name",
    "public_id": "Public ID",
    "city": "City",
    "country": "Country",
    "timezone": "Timezone",
    "event_date": "Event date",
    "distances": "Distance",
    "regions": "Region tags",
    "official_url": "Official URL",
    "registration_url": "Registration URL",
    "registration_status": "Registration status",
    "registration_open_at": "Registration opens",
    "registration_open_precision": "Registration open precision",
    "registration_close_at": "Registration closes",
}

EDITABLE_FIELDS = (
    "name",
    "event_date",
    "city",
    "country",
    "timezone",
    "distances",
    "regions",
    "official_url",
    "registration_url",
    "registration_status",
    "registration_open_at",
    "registration_open_precision",
    "registration_close_at",
)

EDIT_FIELD_ALIASES = {
    "name": "name",
    "event name": "name",
    "date": "event_date",
    "event date": "event_date",
    "city": "city",
    "country": "country",
    "timezone": "timezone",
    "time zone": "timezone",
    "distance": "distances",
    "distances": "distances",
    "region": "regions",
    "regions": "regions",
    "tags": "regions",
    "official url": "official_url",
    "official_url": "official_url",
    "url": "official_url",
    "registration url": "registration_url",
    "registration_url": "registration_url",
    "registration": "registration_url",
    "registration status": "registration_status",
    "registration_status": "registration_status",
    "status": "registration_status",
    "registration opens": "registration_open_at",
    "registration open": "registration_open_at",
    "registration_open_at": "registration_open_at",
    "registration_open": "registration_open_at",
    "opens": "registration_open_at",
    "open at": "registration_open_at",
    "open date": "registration_open_at",
    "registration open precision": "registration_open_precision",
    "registration_open_precision": "registration_open_precision",
    "open precision": "registration_open_precision",
    "precision": "registration_open_precision",
    "registration closes": "registration_close_at",
    "registration close": "registration_close_at",
    "registration_close_at": "registration_close_at",
    "registration_close": "registration_close_at",
    "closes": "registration_close_at",
    "close at": "registration_close_at",
    "close date": "registration_close_at",
}


@router.message(Command("cancel"))
async def handle_cancel(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Nothing to cancel.", reply_markup=remove_dialog_keyboard())
        return

    await state.clear()
    await message.answer("Cancelled.", reply_markup=remove_dialog_keyboard())


@router.message(lambda message: is_cancel_text(message.text))
async def handle_cancel_button(message: Message, state: FSMContext) -> None:
    await handle_cancel(message, state)


@router.message(Command("add_event"))
async def handle_add_event(
    message: Message,
    state: FSMContext,
    command: CommandObject | None = None,
) -> None:
    if not await require_moderator(message):
        return

    await state.clear()
    source = (command.args or "").strip() if command else ""
    if source:
        if not valid_url(source):
            await message.answer(
                "Please send a full URL starting with http:// or https://."
            )
            return
        await start_add_event_from_url(message, state, source)
        return

    await state.set_state(AddEventStates.source_url)
    await message.answer(
        waiting_prompt(
            "Add event",
            "official event URL",
            example="https://example[.]com/race",
        ),
        reply_markup=dialog_keyboard(),
    )


@router.message(AddEventStates.source_url)
async def handle_add_event_source_url(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    source = text_value(message)
    if not valid_url(source):
        await message.answer("Please send a full URL starting with http:// or https://.")
        return

    await start_add_event_from_url(message, state, source)


async def start_add_event_from_suggestion(
    message: Message,
    state: FSMContext,
    suggestion_number: int,
) -> None:
    suggestion = pending_suggestion_by_number(suggestion_number)
    if suggestion is None:
        await message.answer(
            f"I could not find pending suggestion <code>{suggestion_number}</code>."
        )
        return
    if not suggestion.url:
        await message.answer(
            f"Suggestion <code>{suggestion_number}</code> has no URL. "
            "Reject it or handle it manually."
        )
        return

    await start_add_event_from_suggestion_record(
        message,
        state,
        suggestion,
        label=str(suggestion_number),
    )


async def start_add_event_from_suggestion_record(
    message: Message,
    state: FSMContext,
    suggestion,
    *,
    label: str | None = None,
    announce: bool = True,
) -> None:
    if not suggestion.url:
        await message.answer("This suggestion has no URL. Reject it or handle it manually.")
        return

    if announce:
        label_text = f" <code>{escape(label)}</code>" if label else ""
        await message.answer(
            f"Starting review for suggestion{label_text}: "
            f"<b>{escape(suggestion.event_name)}</b>."
        )
    await start_add_event_from_url(
        message,
        state,
        suggestion.url,
        source_suggestion_id=suggestion.id,
        source_suggestion_note=_researcher_suggestion_evidence(suggestion),
    )


async def start_add_event_from_url(
    message: Message,
    state: FSMContext,
    url: str,
    *,
    source_suggestion_id: int | None = None,
    source_suggestion_note: str | None = None,
) -> None:
    await warn_existing_url_events(message, url)

    try:
        extractor_provider = get_extractor_provider(get_settings())
    except ExtractorProviderConfigError as error:
        await message.answer(f"Extractor provider is not configured: {escape(str(error))}")
        return

    draft = await extract_event_draft_from_url(url, extractor_provider=extractor_provider)
    state_data = draft_to_state(draft)
    if source_suggestion_id is not None:
        state_data["source_suggestion_id"] = source_suggestion_id
        state_data["source_suggestion_note"] = source_suggestion_note
    await state.update_data(**state_data)
    await message.answer(format_draft_summary(draft))
    await state.set_state(AddEventStates.name)
    await ask_field_confirmation(message, "name", draft.name)


@router.message(AddEventStates.name)
async def handle_add_event_name(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    value = confirmed_value(text_value(message), data.get("name"))
    if not value:
        await message.answer("Please send the event name.")
        return

    await state.update_data(name=value)
    await state.set_state(AddEventStates.public_id)
    await ask_field_confirmation(message, "public_id", (await state.get_data()).get("public_id"))


@router.message(AddEventStates.public_id)
async def handle_add_event_public_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    value = normalize_event_id(str(confirmed_value(text_value(message), data.get("public_id"))))
    if public_id_distance_code(value) not in DISTANCE_CODE_TO_KEY:
        await message.answer(
            "Use public ID format <code>&lt;place&gt;.&lt;distance&gt;</code> "
            f"ending in one of: <code>{escape(supported_distance_codes())}</code>, "
            "for example <i>zurich.42</i>."
        )
        return

    existing_event = find_database_event(value)
    if existing_event is not None:
        await message.answer(
            format_existing_id_warning(existing_event),
            reply_markup=dialog_keyboard(),
        )
        return

    await state.update_data(public_id=value)
    await state.set_state(AddEventStates.city)
    await ask_field_confirmation(message, "city", (await state.get_data()).get("city"))


@router.message(AddEventStates.city)
async def handle_add_event_city(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    value = confirmed_value(text_value(message), data.get("city"))
    if not value:
        await message.answer("Please send the city.")
        return

    await state.update_data(city=value)
    await state.set_state(AddEventStates.country)
    await ask_field_confirmation(message, "country", (await state.get_data()).get("country"))


@router.message(AddEventStates.country)
async def handle_add_event_country(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    value = confirmed_value(text_value(message), data.get("country"))
    if not value:
        await message.answer("Please send the country.")
        return

    await state.update_data(country=value)
    await state.set_state(AddEventStates.timezone)
    await ask_field_confirmation(message, "timezone", (await state.get_data()).get("timezone"))


@router.message(AddEventStates.timezone)
async def handle_add_event_timezone(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    value = str(confirmed_value(text_value(message), data.get("timezone")) or "")
    if "/" not in value:
        await message.answer(
            "Please send an IANA timezone, for example <i>Europe/Berlin</i>."
        )
        return

    await state.update_data(timezone=value)
    await state.set_state(AddEventStates.event_date)
    await ask_field_confirmation(message, "event_date", (await state.get_data()).get("event_date"))


@router.message(AddEventStates.event_date)
async def handle_add_event_date(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("event_date"))
        event_date = parse_optional_date(str(raw_value or "-"))
    except ValueError:
        await message.answer("Use YYYY-MM-DD, or <code>-</code> if unknown.")
        return

    await state.update_data(event_date=event_date)
    await state.set_state(AddEventStates.distance)
    await ask_field_confirmation(message, "distances", (await state.get_data()).get("distances"))


@router.message(AddEventStates.distance)
async def handle_add_event_distance(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("distances"))
        distances = (
            tuple(raw_value)
            if isinstance(raw_value, tuple | list)
            else parse_distances(str(raw_value or ""))
        )
    except ValueError as error:
        await message.answer(str(error))
        return

    registration_url = select_registration_url_for_distances(
        tuple(data.get("registration_url_candidates") or ()),
        distances,
        fallback=optional_string(data.get("registration_url")),
    )
    await state.update_data(distances=distances, registration_url=registration_url)
    await state.set_state(AddEventStates.regions)
    await ask_field_confirmation(message, "regions", (await state.get_data()).get("regions"))


@router.message(AddEventStates.regions)
async def handle_add_event_regions(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("regions"))
        regions = (
            tuple(raw_value)
            if isinstance(raw_value, tuple | list)
            else parse_regions(str(raw_value or ""))
        )
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(regions=regions)
    await state.set_state(AddEventStates.official_url)
    await ask_field_confirmation(
        message,
        "official_url",
        (await state.get_data()).get("official_url"),
    )


@router.message(AddEventStates.official_url)
async def handle_add_event_official_url(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    url = str(confirmed_value(text_value(message), data.get("official_url")) or "")
    if not valid_url(url):
        await message.answer("Please send a full URL starting with http:// or https://.")
        return

    await warn_existing_url_events(message, url)

    await state.update_data(official_url=url)
    await state.set_state(AddEventStates.registration_url)
    await ask_field_confirmation(
        message,
        "registration_url",
        (await state.get_data()).get("registration_url"),
    )


@router.message(AddEventStates.registration_url)
async def handle_add_event_registration_url(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    raw_value = confirmed_value(text_value(message), data.get("registration_url"))
    registration_url = parse_optional_url(str(raw_value or "-"))
    if registration_url is not None and not valid_url(registration_url):
        await message.answer(
            "Please send a full URL starting with http:// or https://, or <code>-</code>."
        )
        return
    if registration_url is not None:
        await warn_existing_url_events(message, registration_url)

    await state.update_data(registration_url=registration_url)
    await state.set_state(AddEventStates.registration_status)
    await ask_field_confirmation(
        message,
        "registration_status",
        (await state.get_data()).get("registration_status"),
    )


@router.message(AddEventStates.registration_status)
async def handle_add_event_registration_status(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("registration_status"))
        registration_status = parse_registration_status(str(raw_value or "unknown"))
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(registration_status=registration_status)
    await state.set_state(AddEventStates.registration_open_at)
    await ask_field_confirmation(
        message,
        "registration_open_at",
        (await state.get_data()).get("registration_open_at"),
    )


@router.message(AddEventStates.registration_open_at)
async def handle_add_event_registration_open_at(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("registration_open_at"))
        registration_open_at = parse_optional_registration_time(str(raw_value or "-"))
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(registration_open_at=registration_open_at)
    await state.set_state(AddEventStates.registration_open_precision)
    await ask_field_confirmation(
        message,
        "registration_open_precision",
        (await state.get_data()).get("registration_open_precision"),
    )


@router.message(AddEventStates.registration_open_precision)
async def handle_add_event_registration_open_precision(
    message: Message,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(
            text_value(message),
            data.get("registration_open_precision"),
        )
        registration_open_precision = parse_registration_open_precision(
            str(raw_value or "unknown")
        )
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(registration_open_precision=registration_open_precision)
    await state.set_state(AddEventStates.registration_close_at)
    await ask_field_confirmation(
        message,
        "registration_close_at",
        (await state.get_data()).get("registration_close_at"),
    )


@router.message(AddEventStates.registration_close_at)
async def handle_add_event_registration_close_at(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    try:
        data = await state.get_data()
        raw_value = confirmed_value(text_value(message), data.get("registration_close_at"))
        registration_close_at = parse_optional_registration_time(str(raw_value or "-"))
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(registration_close_at=registration_close_at)
    data = await state.get_data()
    try:
        event_create = EventCreate(
            public_id=data["public_id"],
            name=data["name"],
            city=data["city"],
            country=data["country"],
            timezone=data["timezone"],
            event_date=data["event_date"],
            distances=data["distances"],
            regions=data["regions"],
            official_url=data["official_url"],
            registration_url=optional_string(data.get("registration_url")),
            registration_status=str(data["registration_status"]),
            registration_open_at=optional_string(data.get("registration_open_at")),
            registration_open_precision=str(data["registration_open_precision"]),
            registration_close_at=registration_close_at,
        )
        source_suggestion_id = data.get("source_suggestion_id")
        if source_suggestion_id is None:
            event = add_event(event_create)
        else:
            event = add_event_from_suggestion(event_create, int(source_suggestion_id))
    except EventWriteError as error:
        await state.clear()
        await message.answer(
            f"Could not add event: {escape(str(error))}\nStart again with /add_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    await state.clear()
    await message.answer(
        format_event_added_confirmation(
            from_suggestion=source_suggestion_id is not None,
            suggestion_note=optional_string(data.get("source_suggestion_note")),
        ),
        reply_markup=remove_dialog_keyboard(),
    )
    await message.answer(format_event_detail(event))
    await message.answer("Running first registration scan...")
    try:
        registration_update = await update_registration_window(event)
    except Exception as error:
        await message.answer(
            "First registration scan failed. Event was added and can be checked later.\n"
            f"Error: {escape(str(error))}"
        )
        return

    await message.answer(format_registration_update_result(registration_update))


@router.message(Command("edit_event"))
async def handle_edit_event(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    await state.clear()
    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(EditEventStates.event_id)
        await message.answer(
            waiting_prompt("Edit event", "event ID", example="berlin.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_edit_event(message, state, event_id)


@router.message(EditEventStates.event_id)
async def handle_edit_event_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    event_id = text_value(message)
    if not event_id:
        await message.answer(
            waiting_prompt("Edit event", "event ID", example="berlin.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_edit_event(message, state, event_id)


async def start_edit_event(message: Message, state: FSMContext, event_id: str) -> None:
    event = find_database_event(event_id)
    if event is None:
        await message.answer(
            f"I could not find event <code>{escape(event_id.strip())}</code>.",
            reply_markup=dialog_keyboard(),
        )
        return

    await state.update_data(edit_event_id=event.public_id, **event_to_edit_state(event))
    await state.set_state(EditEventStates.field)
    await message.answer(format_edit_event_prompt(event), reply_markup=dialog_keyboard())


@router.message(EditEventStates.field)
async def handle_edit_event_field(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    raw_field = text_value(message)
    field = parse_edit_field(raw_field)
    if field is None:
        await message.answer(format_edit_field_error())
        return

    data = await state.get_data()
    await state.update_data(edit_field=field)
    await state.set_state(EditEventStates.value)
    await ask_edit_value(message, field, data.get(field))


@router.message(EditEventStates.value)
async def handle_edit_event_value(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    field = str(data["edit_field"])
    try:
        new_value = parse_edit_value(field, text_value(message))
    except ValueError as error:
        await message.answer(str(error))
        return

    values = {key: data.get(key) for key in EDITABLE_FIELDS}
    values[field] = new_value

    if field == "official_url":
        await warn_existing_url_events(message, str(new_value))
    elif field == "registration_url" and new_value is not None:
        await warn_existing_url_events(message, str(new_value))

    try:
        event = update_event(
            str(data["edit_event_id"]),
            EventUpdate(
                name=str(values["name"]),
                city=str(values["city"]),
                country=str(values["country"]),
                timezone=str(values["timezone"]),
                event_date=optional_string(values["event_date"]),
                distances=tuple(values["distances"] or ()),
                regions=tuple(values["regions"] or ()),
                official_url=str(values["official_url"]),
                registration_url=optional_string(values["registration_url"]),
                registration_status=str(values["registration_status"]),
                registration_open_at=optional_string(values["registration_open_at"]),
                registration_open_precision=str(values["registration_open_precision"]),
                registration_close_at=optional_string(values["registration_close_at"]),
            ),
        )
    except EventWriteError as error:
        await message.answer(f"Could not update event: {escape(str(error))}")
        return

    if event is None:
        await state.clear()
        await message.answer(
            "Event no longer exists. Start again with /edit_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    await state.clear()
    await message.answer("Event updated.", reply_markup=remove_dialog_keyboard())
    await message.answer(format_event_detail(event))


@router.message(Command("update_event"))
async def handle_update_event(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(UpdateEventStates.event_id)
        await message.answer(
            waiting_prompt("Update event", "event ID", example="berlin.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await update_event_registration_by_id(message, event_id)


@router.message(UpdateEventStates.event_id)
async def handle_update_event_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    event_id = text_value(message)
    if not event_id:
        await message.answer(
            waiting_prompt("Update event", "event ID", example="berlin.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await update_event_registration_by_id(message, event_id, cleanup_keyboard=True)
    await state.clear()


async def update_event_registration_by_id(
    message: Message,
    event_id: str,
    *,
    cleanup_keyboard: bool = False,
) -> None:
    event = find_database_event(event_id)
    if event is None:
        await message.answer(
            f"I could not find event <code>{escape(event_id.strip())}</code>.",
            reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
        )
        return

    await message.answer(
        f"Running registration scan for <b>{escape(event.name)}</b>...",
        reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
    )
    try:
        registration_update = await update_registration_window(event)
    except Exception as error:
        await message.answer(
            "Registration scan failed. Try again later.\n"
            f"Error: {escape(str(error))}"
        )
        return

    await message.answer(format_registration_update_result(registration_update))


@router.message(Command("archive_event"))
async def handle_archive_event(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(ArchiveEventStates.event_id)
        await message.answer(
            waiting_prompt("Archive event", "event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_archive_event_confirmation(message, state, event_id)


@router.message(ArchiveEventStates.event_id)
async def handle_archive_event_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    event_id = text_value(message)
    if not event_id:
        await message.answer(
            waiting_prompt("Archive event", "event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_archive_event_confirmation(message, state, event_id)


async def start_archive_event_confirmation(
    message: Message,
    state: FSMContext,
    event_id: str,
) -> None:
    event = find_database_event(event_id)
    if event is None:
        await message.answer(f"I could not find event <code>{escape(event_id.strip())}</code>.")
        return

    await state.clear()
    await message.answer(
        format_archive_event_confirmation(event),
        reply_markup=archive_event_confirmation_keyboard(event.public_id),
    )


@router.message(ArchiveEventStates.confirm)
async def handle_archive_event_confirmation(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    event_id = str(data.get("archive_event_id") or "")
    if not event_id:
        await state.clear()
        await message.answer(
            "Archive flow lost the event ID. Start again with /archive_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    if not is_archive_event_confirmation(text_value(message)):
        await message.answer(
            "No changes made.",
            reply_markup=archive_event_confirmation_keyboard(event_id),
        )
        return

    await archive_event_by_id(message, state, event_id)


async def archive_event_by_id(message: Message, state: FSMContext, event_id: str) -> None:
    event = archive_event(event_id)
    await state.clear()
    if event is None:
        await message.answer(
            "Event no longer exists. Start again with /archive_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    await message.answer(
        f"Archived <b>{escape(event.name)}</b> from active tracking.",
        reply_markup=remove_dialog_keyboard(),
    )


async def archive_event_by_id_in_place(message: Message, event_id: str) -> None:
    event = archive_event(event_id)
    if event is None:
        await message.edit_text(
            "Event no longer exists. Start again with /archive_event.",
            reply_markup=None,
        )
        return

    await message.edit_text(
        f"Archived <b>{escape(event.name)}</b> from active tracking.",
        reply_markup=None,
    )


@router.callback_query(F.data.startswith(ARCHIVE_EVENT_CALLBACK_PREFIX))
async def handle_archive_event_callback(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id = (callback.data or "").removeprefix(ARCHIVE_EVENT_CALLBACK_PREFIX)
    if not event_id:
        await callback.answer("Archive button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await state.clear()
    await archive_event_by_id_in_place(callback.message, event_id)


@router.message(Command("list_archive"))
async def handle_list_archive(message: Message, command: CommandObject) -> None:
    if not await require_moderator(message):
        return

    try:
        limit = parse_archive_limit(command.args or "")
    except ValueError as error:
        await message.answer(str(error))
        return

    archived_events = list_archived_events(limit=limit)
    if not archived_events:
        await message.answer("Archive is empty.")
        return

    await send_archived_event_cards(message, archived_events, limit=limit)


@router.callback_query(F.data.startswith(ARCHIVE_LIST_CALLBACK_PREFIX))
async def handle_archive_list_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    try:
        limit = parse_archive_limit(
            (callback.data or "").removeprefix(ARCHIVE_LIST_CALLBACK_PREFIX)
        )
    except ValueError:
        limit = ARCHIVE_LIST_DEFAULT_LIMIT
    archived_events = list_archived_events(limit=limit)
    await callback.answer()
    await replace_with_archived_event_cards(callback.message, archived_events, limit=limit)


@router.callback_query(F.data.startswith(ARCHIVE_SHOW_CALLBACK_PREFIX))
async def handle_archive_show_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    public_id, list_limit = parse_archive_show_callback_payload(
        (callback.data or "").removeprefix(ARCHIVE_SHOW_CALLBACK_PREFIX)
    )
    if not public_id:
        await callback.answer("Archive button is invalid.", show_alert=True)
        return

    archived = find_archived_event(public_id)
    if archived is None:
        await callback.answer("Archived event not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_archived_event_detail(archived),
        reply_markup=archived_event_detail_keyboard(
            archived,
            list_limit=list_limit,
        ),
    )


@router.message(Command("restore_event"))
async def handle_restore_event(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(RestoreEventStates.event_id)
        await message.answer(
            waiting_prompt("Restore event", "archived event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await restore_event_by_id(message, event_id)


@router.message(RestoreEventStates.event_id)
async def handle_restore_event_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    event_id = text_value(message)
    if not event_id:
        await message.answer(
            waiting_prompt("Restore event", "archived event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await restore_event_by_id(message, event_id, cleanup_keyboard=True)
    await state.clear()


async def restore_event_by_id(
    message: Message,
    event_id: str,
    *,
    cleanup_keyboard: bool = False,
) -> None:
    event = restore_event(event_id)
    if event is None:
        await message.answer(
            f"I could not find archived event <code>{escape(event_id.strip())}</code>.",
            reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
        )
        return

    await message.answer(
        f"Restored <b>{escape(event.name)}</b> to active tracking.",
        reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
    )
    await message.answer(format_event_detail(event))


async def restore_event_by_id_in_place(message: Message, event_id: str) -> None:
    event = restore_event(event_id)
    if event is None:
        await message.edit_text(
            f"I could not find archived event <code>{escape(event_id.strip())}</code>.",
            reply_markup=None,
        )
        return

    await message.edit_text(
        "\n\n".join(
            [
                f"Restored <b>{escape(event.name)}</b> to active tracking.",
                format_event_detail(event),
            ]
        ),
        reply_markup=None,
    )


@router.callback_query(F.data.startswith(RESTORE_EVENT_CALLBACK_PREFIX))
async def handle_restore_event_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id, list_limit = parse_restore_callback_payload(
        (callback.data or "").removeprefix(RESTORE_EVENT_CALLBACK_PREFIX)
    )
    if not event_id:
        await callback.answer("Restore button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    archived = find_archived_event(event_id)
    if archived is None:
        await callback.answer("Archived event not found.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_restore_event_confirmation(archived.event),
        reply_markup=restore_event_confirmation_keyboard(
            archived.event.public_id,
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(RESTORE_CONFIRM_CALLBACK_PREFIX))
async def handle_restore_event_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id = (callback.data or "").removeprefix(RESTORE_CONFIRM_CALLBACK_PREFIX)
    if not event_id:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await restore_event_by_id_in_place(callback.message, event_id)


@router.message(Command("delete_event"))
async def handle_delete_event(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(DeleteEventStates.event_id)
        await message.answer(
            waiting_prompt("Delete event", "event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_delete_event_confirmation(message, state, event_id)


@router.message(DeleteEventStates.event_id)
async def handle_delete_event_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    event_id = text_value(message)
    if not event_id:
        await message.answer(
            waiting_prompt("Delete event", "event ID", example="zurich.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await start_delete_event_confirmation(message, state, event_id)


async def start_delete_event_confirmation(
    message: Message,
    state: FSMContext,
    event_id: str,
) -> None:
    event = find_event_for_delete(event_id)
    if event is None:
        await message.answer(f"I could not find event <code>{escape(event_id.strip())}</code>.")
        return

    await state.clear()
    await message.answer(
        format_delete_event_confirmation(event),
        reply_markup=delete_event_preview_keyboard(event.public_id),
    )


def find_event_for_delete(event_id: str):
    event = find_database_event(event_id)
    if event is not None:
        return event

    archived = find_archived_event(event_id)
    return archived.event if archived is not None else None


async def show_delete_event_preview_in_place(message: Message, event_id: str) -> None:
    event = find_event_for_delete(event_id)
    if event is None:
        await message.edit_text(
            f"I could not find event <code>{escape(event_id.strip())}</code>.",
            reply_markup=None,
        )
        return

    await message.edit_text(
        format_delete_event_confirmation(event),
        reply_markup=delete_event_preview_keyboard(event.public_id),
    )


async def show_delete_event_final_confirmation_in_place(
    message: Message,
    event_id: str,
) -> None:
    event = find_event_for_delete(event_id)
    if event is None:
        await message.edit_text(
            f"I could not find event <code>{escape(event_id.strip())}</code>.",
            reply_markup=None,
        )
        return

    await message.edit_text(
        format_delete_event_final_confirmation(event),
        reply_markup=delete_event_confirmation_keyboard(event.public_id),
    )


async def delete_event_by_id_in_place(message: Message, event_id: str) -> None:
    event = delete_event(event_id)
    if event is None:
        await message.edit_text(
            "Event no longer exists. Start again with /delete_event.",
            reply_markup=None,
        )
        return

    await message.edit_text(
        f"Deleted <b>{escape(event.name)}</b> permanently.",
        reply_markup=None,
    )


@router.callback_query(F.data.startswith(DELETE_EVENT_PREVIEW_CALLBACK_PREFIX))
async def handle_delete_event_preview_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id = (callback.data or "").removeprefix(DELETE_EVENT_PREVIEW_CALLBACK_PREFIX)
    if not event_id:
        await callback.answer("Delete button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await show_delete_event_preview_in_place(callback.message, event_id)


@router.callback_query(F.data.startswith(DELETE_EVENT_CALLBACK_PREFIX))
async def handle_delete_event_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id = (callback.data or "").removeprefix(DELETE_EVENT_CALLBACK_PREFIX)
    if not event_id:
        await callback.answer("Delete button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await show_delete_event_final_confirmation_in_place(callback.message, event_id)


@router.callback_query(F.data.startswith(DELETE_EVENT_CONFIRM_CALLBACK_PREFIX))
async def handle_delete_event_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    event_id = (callback.data or "").removeprefix(DELETE_EVENT_CONFIRM_CALLBACK_PREFIX)
    if not event_id:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await delete_event_by_id_in_place(callback.message, event_id)


@router.message(DeleteEventStates.confirm)
async def handle_delete_event_confirmation(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    data = await state.get_data()
    event_id = str(data.get("delete_event_id") or "")
    if not event_id:
        await state.clear()
        await message.answer(
            "Delete flow lost the event ID. Start again with /delete_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    if not is_delete_event_confirmation(text_value(message), event_id):
        await message.answer(
            f"No changes made. Reply <code>delete {escape(event_id)}</code> "
            "to confirm, or cancel.",
            reply_markup=dialog_keyboard(),
        )
        return

    event = delete_event(event_id)
    await state.clear()
    if event is None:
        await message.answer(
            "Event no longer exists. Start again with /delete_event.",
            reply_markup=remove_dialog_keyboard(),
        )
        return

    await message.answer(
        f"Deleted <b>{escape(event.name)}</b> permanently.",
        reply_markup=remove_dialog_keyboard(),
    )


@router.message(Command("show_suggestion"))
async def handle_show_suggestion(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    suggestion_id = command.args or ""
    if not suggestion_id.strip():
        await state.set_state(ShowSuggestionStates.suggestion_id)
        await message.answer(
            waiting_prompt("Show suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await show_suggestion_by_record_id(message, suggestion_id)


@router.message(ShowSuggestionStates.suggestion_id)
async def handle_show_suggestion_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    suggestion_id = text_value(message)
    if not suggestion_id:
        await message.answer(
            waiting_prompt("Show suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    completed = await show_suggestion_by_record_id(
        message,
        suggestion_id,
        cleanup_keyboard=True,
        retry_keyboard=True,
    )
    if completed:
        await state.clear()


@router.message(Command("list_suggestions"))
async def handle_list_suggestions(message: Message, command: CommandObject) -> None:
    if not await require_moderator(message):
        return

    if (command.args or "").strip():
        await message.answer("Use /list_suggestions without a count.")
        return

    limit = SUGGESTION_LIST_DEFAULT_LIMIT
    suggestions = list_event_suggestions(limit=limit)
    if not suggestions:
        await message.answer("No pending suggestions.")
        return

    await send_suggestion_cards(message, suggestions, limit=limit)


@router.message(Command("next_update"))
async def handle_next_update(message: Message) -> None:
    if not await require_moderator(message):
        return

    updates = list_proposed_event_updates(limit=1)
    if not updates:
        await message.answer("No pending updates.")
        return

    await show_update_by_record_id(message, format_update_handle(updates[0].id))


@router.message(Command("next_suggestion"))
async def handle_next_suggestion(message: Message) -> None:
    if not await require_moderator(message):
        return

    suggestions = list_event_suggestions(limit=1)
    if not suggestions:
        await message.answer("No pending suggestions.")
        return

    await show_suggestion_by_record_id(message, format_suggestion_handle(suggestions[0].id))


@router.message(Command("apply_suggestion"))
async def handle_apply_suggestion(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    await state.clear()
    suggestion_id = command.args or ""
    if not suggestion_id.strip():
        await state.set_state(ApplySuggestionStates.suggestion_id)
        await message.answer(
            waiting_prompt("Apply suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await apply_suggestion_by_record_id(message, state, suggestion_id)


@router.message(ApplySuggestionStates.suggestion_id)
async def handle_apply_suggestion_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    suggestion_id = text_value(message)
    if not suggestion_id:
        await message.answer(
            waiting_prompt("Apply suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    completed = await apply_suggestion_by_record_id(
        message,
        state,
        suggestion_id,
        cleanup_keyboard=True,
        retry_keyboard=True,
    )
    if not completed:
        return


@router.message(Command("todo"))
async def handle_todo(message: Message) -> None:
    if not await require_moderator(message):
        return

    pending_updates = count_proposed_event_updates()
    pending_suggestions = count_event_suggestions()
    await message.answer(
        format_moderator_status(
            pending_updates=pending_updates,
            pending_suggestions=pending_suggestions,
        ),
        reply_markup=todo_keyboard(
            pending_updates=pending_updates,
            pending_suggestions=pending_suggestions,
        ),
    )


@router.message(Command("list_updates"))
async def handle_list_updates(message: Message, command: CommandObject) -> None:
    if not await require_moderator(message):
        return

    try:
        limit = parse_update_limit(command.args or "")
    except ValueError as error:
        await message.answer(str(error))
        return

    updates = list_proposed_event_updates(limit=limit)
    if not updates:
        await message.answer("No pending updates.")
        return

    await send_proposed_update_cards(message, updates, limit=limit)


@router.message(Command("show_update"))
async def handle_show_update(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    update_id = command.args or ""
    if not update_id.strip():
        await state.set_state(ShowUpdateStates.update_number)
        await message.answer(
            waiting_prompt("Show update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await show_update_by_record_id(message, update_id)


@router.message(ShowUpdateStates.update_number)
async def handle_show_update_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    update_id = text_value(message)
    if not update_id:
        await message.answer(
            waiting_prompt("Show update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    completed = await show_update_by_record_id(
        message,
        update_id,
        cleanup_keyboard=True,
        retry_keyboard=True,
    )
    if completed:
        await state.clear()


@router.message(Command("apply_update"))
async def handle_apply_update(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    update_id = command.args or ""
    if not update_id.strip():
        await state.set_state(ApplyUpdateStates.update_number)
        await message.answer(
            waiting_prompt("Apply update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await apply_update_by_record_id_text(message, update_id)


@router.message(ApplyUpdateStates.update_number)
async def handle_apply_update_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    update_id = text_value(message)
    if not update_id:
        await message.answer(
            waiting_prompt("Apply update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    completed = await apply_update_by_record_id_text(
        message,
        update_id,
        cleanup_keyboard=True,
        retry_keyboard=True,
    )
    if completed:
        await state.clear()


@router.message(Command("reject_update"))
async def handle_reject_update(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    update_id = command.args or ""
    if not update_id.strip():
        await state.set_state(RejectUpdateStates.update_number)
        await message.answer(
            waiting_prompt("Reject update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await reject_update_by_record_id_text(message, update_id)


@router.message(RejectUpdateStates.update_number)
async def handle_reject_update_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    update_id = text_value(message)
    if not update_id:
        await message.answer(
            waiting_prompt("Reject update", "update ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    completed = await reject_update_by_record_id_text(
        message,
        update_id,
        cleanup_keyboard=True,
        retry_keyboard=True,
    )
    if completed:
        await state.clear()


async def show_update_by_record_id(
    message: Message,
    value: str,
    *,
    cleanup_keyboard: bool = False,
    retry_keyboard: bool = False,
) -> bool:
    update_id = parse_update_record_id(value)
    if update_id is None:
        await message.answer(
            waiting_prompt("Show update", "update ID", example="#1"),
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await message.answer(
            f"I could not find pending update <code>{format_update_handle(update_id)}</code>.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    if cleanup_keyboard:
        await message.answer(INPUT_RECEIVED_MESSAGE, reply_markup=remove_dialog_keyboard())
    await message.answer(
        format_proposed_update_detail(update),
        reply_markup=proposed_update_detail_keyboard(update),
    )
    return True


async def show_suggestion_by_record_id(
    message: Message,
    value: str,
    *,
    cleanup_keyboard: bool = False,
    retry_keyboard: bool = False,
) -> bool:
    suggestion_id = parse_suggestion_record_id(value)
    if suggestion_id is None:
        await message.answer(
            waiting_prompt("Show suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    suggestion = find_pending_suggestion_by_record_id(suggestion_id)
    if suggestion is None:
        await message.answer(
            f"I could not find pending suggestion "
            f"<code>{format_suggestion_handle(suggestion_id)}</code>.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    if cleanup_keyboard:
        await message.answer(INPUT_RECEIVED_MESSAGE, reply_markup=remove_dialog_keyboard())
    await message.answer(
        format_suggestion_detail(suggestion),
        reply_markup=suggestion_detail_keyboard(suggestion),
    )
    return True


async def apply_suggestion_by_record_id(
    message: Message,
    state: FSMContext,
    value: str,
    *,
    cleanup_keyboard: bool = False,
    retry_keyboard: bool = False,
) -> bool:
    suggestion_id = parse_suggestion_record_id(value)
    if suggestion_id is None:
        await message.answer(
            waiting_prompt("Apply suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    suggestion = find_pending_suggestion_by_record_id(suggestion_id)
    if suggestion is None:
        await message.answer(
            f"I could not find pending suggestion "
            f"<code>{format_suggestion_handle(suggestion_id)}</code>.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False
    if not suggestion.url:
        await message.answer(
            "This suggestion has no URL. Reject it or handle it manually.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    if cleanup_keyboard:
        await message.answer(INPUT_RECEIVED_MESSAGE, reply_markup=remove_dialog_keyboard())
    await start_add_event_from_suggestion_record(
        message,
        state,
        suggestion,
        label=format_suggestion_handle(suggestion.id),
    )
    return True


async def apply_update_by_record_id_text(
    message: Message,
    value: str,
    *,
    cleanup_keyboard: bool = False,
    retry_keyboard: bool = False,
) -> bool:
    update_id = parse_update_record_id(value)
    if update_id is None:
        await message.answer(
            waiting_prompt("Apply update", "update ID", example="#1"),
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await message.answer(
            f"I could not find pending update <code>{format_update_handle(update_id)}</code>.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    await send_update_review_confirmation(
        message,
        update,
        action="apply",
        cleanup_keyboard=cleanup_keyboard,
    )
    return True


async def reject_update_by_record_id_text(
    message: Message,
    value: str,
    *,
    cleanup_keyboard: bool = False,
    retry_keyboard: bool = False,
) -> bool:
    update_id = parse_update_record_id(value)
    if update_id is None:
        await message.answer(
            waiting_prompt("Reject update", "update ID", example="#1"),
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await message.answer(
            f"I could not find pending update <code>{format_update_handle(update_id)}</code>.",
            reply_markup=dialog_keyboard() if retry_keyboard else None,
        )
        return False

    await send_update_review_confirmation(
        message,
        update,
        action="reject",
        cleanup_keyboard=cleanup_keyboard,
    )
    return True


async def send_update_review_confirmation(
    message: Message,
    update: ProposedEventUpdateRecord,
    *,
    action: str,
    cleanup_keyboard: bool = False,
) -> None:
    if cleanup_keyboard:
        await message.answer(INPUT_RECEIVED_MESSAGE, reply_markup=remove_dialog_keyboard())
    await message.answer(
        format_update_review_confirmation(update, action=action),
        reply_markup=proposed_update_confirmation_keyboard(
            update,
            action=action,
        ),
    )


async def send_proposed_update_cards(
    message: Message,
    updates: tuple[ProposedEventUpdateRecord, ...],
    *,
    limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> None:
    if not updates:
        await message.answer("No pending updates.")
        return

    await message.answer(format_major_title("Pending updates"))
    for update in updates:
        await message.answer(
            format_proposed_update_card(update),
            reply_markup=proposed_update_show_keyboard(update, limit=limit),
        )


async def send_suggestion_cards(
    message: Message,
    suggestions,
    *,
    limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> None:
    if not suggestions:
        await message.answer("No pending suggestions.")
        return

    await message.answer(format_major_title("Pending suggestion"))
    for suggestion in suggestions:
        await message.answer(
            format_suggestion_card(suggestion),
            reply_markup=suggestion_show_keyboard(suggestion, limit=limit),
        )


async def replace_with_proposed_update_cards(
    message: Message,
    updates: tuple[ProposedEventUpdateRecord, ...],
    *,
    limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> None:
    if not updates:
        await message.edit_text("No pending updates.", reply_markup=None)
        return

    await message.edit_text(format_major_title("Pending updates"), reply_markup=None)
    for update in updates:
        await message.answer(
            format_proposed_update_card(update),
            reply_markup=proposed_update_show_keyboard(update, limit=limit),
        )


async def replace_with_suggestion_cards(
    message: Message,
    suggestions,
    *,
    limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> None:
    if not suggestions:
        await message.edit_text("No pending suggestions.", reply_markup=None)
        return

    await message.edit_text(format_major_title("Pending suggestion"), reply_markup=None)
    for suggestion in suggestions:
        await message.answer(
            format_suggestion_card(suggestion),
            reply_markup=suggestion_show_keyboard(suggestion, limit=limit),
        )


async def send_archived_event_cards(
    message: Message,
    archived_events,
    *,
    limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> None:
    if not archived_events:
        await message.answer("Archive is empty.")
        return

    await message.answer(format_major_title("Archived events"))
    for archived in archived_events:
        await message.answer(
            format_archived_event_card(archived),
            reply_markup=archived_event_card_keyboard(
                archived,
                limit=limit,
            ),
        )


async def replace_with_archived_event_cards(
    message: Message,
    archived_events,
    *,
    limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> None:
    if not archived_events:
        await message.edit_text("Archive is empty.", reply_markup=None)
        return

    await message.edit_text(format_major_title("Archived events"), reply_markup=None)
    for archived in archived_events:
        await message.answer(
            format_archived_event_card(archived),
            reply_markup=archived_event_card_keyboard(
                archived,
                limit=limit,
            ),
        )


async def apply_update_by_record_id(
    message: Message,
    update_id: int,
    *,
    reviewer_user_id: str | None,
    label: str | None = None,
) -> None:
    try:
        result = approve_proposed_event_update(
            update_id,
            reviewer_user_id=reviewer_user_id,
        )
    except EventWriteError as error:
        await message.answer(f"Could not apply update: {escape(str(error))}")
        return

    if result is None:
        await message.answer("I could not find that pending update.")
        return

    label_text = f" <code>{escape(label or format_update_handle(update_id))}</code>"
    await message.answer(
        f"Applied update{label_text} to <b>{escape(result.event.name)}</b>."
    )
    await message.answer(format_event_detail(result.event))


async def apply_update_by_record_id_in_place(
    message: Message,
    update_id: int,
    *,
    reviewer_user_id: str | None,
    label: str | None = None,
) -> None:
    try:
        result = approve_proposed_event_update(
            update_id,
            reviewer_user_id=reviewer_user_id,
        )
    except EventWriteError as error:
        await message.edit_text(
            f"Could not apply update: {escape(str(error))}",
            reply_markup=None,
        )
        return

    if result is None:
        await message.edit_text("I could not find that pending update.", reply_markup=None)
        return

    label_text = f" <code>{escape(label or format_update_handle(update_id))}</code>"
    await message.edit_text(
        "\n\n".join(
            [
                f"Applied update{label_text} to <b>{escape(result.event.name)}</b>.",
                format_event_detail(result.event),
            ]
        ),
        reply_markup=None,
    )


async def partial_apply_update_by_record_id_in_place(
    message: Message,
    update_id: int,
    *,
    selected_fields: tuple[str, ...],
    reviewer_user_id: str | None,
) -> None:
    try:
        result = partial_apply_proposed_event_update(
            update_id,
            selected_fields=selected_fields,
            reviewer_user_id=reviewer_user_id,
        )
    except EventWriteError as error:
        await message.edit_text(
            f"Could not apply update: {escape(str(error))}",
            reply_markup=None,
        )
        return

    if result is None:
        await message.edit_text("I could not find that pending update.", reply_markup=None)
        return

    await message.edit_text(
        format_partial_apply_result(result),
        reply_markup=None,
    )


async def reject_update_by_record_id(
    message: Message,
    update_id: int,
    *,
    reviewer_user_id: str | None,
    label: str | None = None,
) -> None:
    update = reject_proposed_event_update(
        update_id,
        reviewer_user_id=reviewer_user_id,
    )
    if update is None:
        await message.answer("I could not find that pending update.")
        return

    label_text = f" <code>{escape(label or format_update_handle(update_id))}</code>"
    await message.answer(f"Rejected update{label_text}.")


async def reject_update_by_record_id_in_place(
    message: Message,
    update_id: int,
    *,
    reviewer_user_id: str | None,
    label: str | None = None,
) -> None:
    update = reject_proposed_event_update(
        update_id,
        reviewer_user_id=reviewer_user_id,
    )
    if update is None:
        await message.edit_text("I could not find that pending update.", reply_markup=None)
        return

    label_text = f" <code>{escape(label or format_update_handle(update_id))}</code>"
    await message.edit_text(
        f"Rejected update{label_text} for <code>{escape(update.event_id)}</code>.",
        reply_markup=None,
    )


@router.callback_query(F.data.startswith(SUGGESTION_ADD_CALLBACK_PREFIX))
async def handle_add_suggestion_callback(
    callback: CallbackQuery,
    state: FSMContext,
) -> None:
    if not await require_moderator_callback(callback):
        return

    suggestion_id, _sequence, _list_limit = parse_suggestion_action_callback_payload(
        (callback.data or "").removeprefix(SUGGESTION_ADD_CALLBACK_PREFIX)
    )
    if suggestion_id is None:
        await callback.answer("Suggestion button is invalid.", show_alert=True)
        return

    suggestion = get_event_suggestion(suggestion_id, status="pending")
    if suggestion is None:
        await callback.answer("Pending suggestion not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        f"Starting event review for <b>{escape(suggestion.event_name)}</b>...",
        reply_markup=None,
    )
    await start_add_event_from_suggestion_record(
        callback.message,
        state,
        suggestion,
        label=format_suggestion_handle(suggestion.id),
        announce=False,
    )


@router.callback_query(F.data.startswith(SUGGESTION_REMOVE_CALLBACK_PREFIX))
async def handle_reject_suggestion_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    suggestion_id, sequence, list_limit = parse_suggestion_action_callback_payload(
        (callback.data or "").removeprefix(SUGGESTION_REMOVE_CALLBACK_PREFIX)
    )
    if suggestion_id is None:
        await callback.answer("Suggestion button is invalid.", show_alert=True)
        return

    suggestion = get_event_suggestion(suggestion_id, status="pending")
    if suggestion is None:
        await callback.answer("Pending suggestion not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_suggestion_reject_confirmation(
            suggestion,
            label=format_suggestion_handle(suggestion.id),
        ),
        reply_markup=suggestion_reject_confirmation_keyboard(
            suggestion,
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(SUGGESTION_REMOVE_CONFIRM_CALLBACK_PREFIX))
async def handle_reject_suggestion_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    suggestion_id, _sequence, _list_limit = parse_suggestion_action_callback_payload(
        (callback.data or "").removeprefix(SUGGESTION_REMOVE_CONFIRM_CALLBACK_PREFIX)
    )
    if suggestion_id is None:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return

    suggestion = get_event_suggestion(suggestion_id, status="pending")
    if suggestion is None:
        await callback.answer("Pending suggestion not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update_event_suggestion_status(suggestion.id, "removed")
    await callback.answer()
    label_text = f" <code>{format_suggestion_handle(suggestion.id)}</code>"
    await callback.message.edit_text(
        f"Rejected suggestion{label_text}: <b>{escape(suggestion.event_name)}</b>.",
        reply_markup=None,
    )


@router.callback_query(F.data.startswith(SUGGESTION_SHOW_CALLBACK_PREFIX))
async def handle_show_suggestion_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    _sequence, suggestion_id, list_limit = parse_suggestion_show_callback_payload(
        (callback.data or "").removeprefix(SUGGESTION_SHOW_CALLBACK_PREFIX)
    )
    if suggestion_id is None:
        await callback.answer("Suggestion button is invalid.", show_alert=True)
        return

    suggestion = get_event_suggestion(suggestion_id, status="pending")
    if suggestion is None:
        await callback.answer("Pending suggestion not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_suggestion_detail(suggestion),
        reply_markup=suggestion_detail_keyboard(
            suggestion,
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(SUGGESTION_CARD_CALLBACK_PREFIX))
async def handle_suggestion_card_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    suggestion_id, list_limit = parse_suggestion_card_callback_payload(
        (callback.data or "").removeprefix(SUGGESTION_CARD_CALLBACK_PREFIX)
    )
    if suggestion_id is None:
        await callback.answer("Suggestion button is invalid.", show_alert=True)
        return

    suggestion = get_event_suggestion(suggestion_id, status="pending")
    if suggestion is None:
        await callback.answer("Pending suggestion not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_suggestion_card(suggestion),
        reply_markup=suggestion_show_keyboard(
            suggestion,
            limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(SUGGESTION_LIST_CALLBACK_PREFIX))
async def handle_suggestion_list_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    try:
        limit = parse_suggestion_list_callback_payload(
            (callback.data or "").removeprefix(SUGGESTION_LIST_CALLBACK_PREFIX)
        )
    except ValueError:
        limit = SUGGESTION_LIST_DEFAULT_LIMIT
    suggestions = list_event_suggestions(limit=limit)
    await callback.answer()
    await replace_with_suggestion_cards(callback.message, suggestions, limit=limit)


@router.callback_query(F.data.startswith(UPDATE_SHOW_CALLBACK_PREFIX))
async def handle_show_update_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    payload = (callback.data or "").removeprefix(UPDATE_SHOW_CALLBACK_PREFIX)
    update_id, list_limit = parse_update_show_callback_payload(payload)
    if update_id is None:
        await callback.answer("Update button is invalid.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_proposed_update_detail(update),
        reply_markup=proposed_update_detail_keyboard(
            update,
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_CARD_CALLBACK_PREFIX))
async def handle_update_card_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, list_limit = parse_update_action_callback_payload(
        (callback.data or "").removeprefix(UPDATE_CARD_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Update button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_proposed_update_card(update),
        reply_markup=proposed_update_show_keyboard(
            update,
            limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_LIST_CALLBACK_PREFIX))
async def handle_update_list_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    limit = parse_update_list_callback_payload(
        (callback.data or "").removeprefix(UPDATE_LIST_CALLBACK_PREFIX)
    )
    updates = list_proposed_event_updates(limit=limit)
    await callback.answer()
    await replace_with_proposed_update_cards(callback.message, updates, limit=limit)


@router.callback_query(F.data.startswith(UPDATE_APPLY_CALLBACK_PREFIX))
async def handle_apply_update_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, list_limit = parse_update_action_callback_payload(
        (callback.data or "").removeprefix(UPDATE_APPLY_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Update button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_update_review_confirmation(
            update,
            action="apply",
        ),
        reply_markup=proposed_update_confirmation_keyboard(
            update,
            action="apply",
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_REJECT_CALLBACK_PREFIX))
async def handle_reject_update_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, list_limit = parse_update_action_callback_payload(
        (callback.data or "").removeprefix(UPDATE_REJECT_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Update button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_update_review_confirmation(
            update,
            action="reject",
        ),
        reply_markup=proposed_update_confirmation_keyboard(
            update,
            action="reject",
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_PARTIAL_CALLBACK_PREFIX))
async def handle_partial_update_callback(callback: CallbackQuery) -> None:
    await show_partial_update_selection(
        callback,
        (callback.data or "").removeprefix(UPDATE_PARTIAL_CALLBACK_PREFIX),
    )


@router.callback_query(F.data.startswith(UPDATE_PARTIAL_TOGGLE_CALLBACK_PREFIX))
async def handle_partial_update_toggle_callback(callback: CallbackQuery) -> None:
    await show_partial_update_selection(
        callback,
        (callback.data or "").removeprefix(UPDATE_PARTIAL_TOGGLE_CALLBACK_PREFIX),
    )


async def show_partial_update_selection(callback: CallbackQuery, payload: str) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, list_limit, selected_mask = parse_update_partial_callback_payload(payload)
    if update_id is None:
        await callback.answer("Update button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    changed_fields = proposed_update_changed_field_names(update)
    if not changed_fields:
        await callback.answer("No changed fields to apply.", show_alert=True)
        return

    selected_fields = decode_update_field_mask(selected_mask, changed_fields)
    await callback.answer()
    await callback.message.edit_text(
        format_update_partial_selection(
            update,
            selected_fields=selected_fields,
        ),
        reply_markup=proposed_update_partial_keyboard(
            update,
            selected_fields=selected_fields,
            list_limit=list_limit,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_PARTIAL_CONFIRM_CALLBACK_PREFIX))
async def handle_partial_update_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, _list_limit, selected_mask = parse_update_partial_callback_payload(
        (callback.data or "").removeprefix(UPDATE_PARTIAL_CONFIRM_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    changed_fields = proposed_update_changed_field_names(update)
    selected_fields = decode_update_field_mask(selected_mask, changed_fields)
    if not selected_fields:
        await callback.answer("Select at least one field.", show_alert=True)
        return

    await callback.answer()
    await callback.message.edit_text(
        format_update_partial_confirmation(
            update,
            selected_fields=selected_fields,
        ),
        reply_markup=proposed_update_partial_confirmation_keyboard(
            update,
            selected_fields=selected_fields,
        ),
    )


@router.callback_query(F.data.startswith(UPDATE_PARTIAL_APPLY_CALLBACK_PREFIX))
async def handle_partial_update_apply_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, _list_limit, selected_mask = parse_update_partial_callback_payload(
        (callback.data or "").removeprefix(UPDATE_PARTIAL_APPLY_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    update = find_pending_update_by_record_id(update_id)
    if update is None:
        await callback.answer("Pending update not found.", show_alert=True)
        return

    changed_fields = proposed_update_changed_field_names(update)
    selected_fields = decode_update_field_mask(selected_mask, changed_fields)
    if not selected_fields:
        await callback.answer("Select at least one field.", show_alert=True)
        return

    await callback.answer()
    await partial_apply_update_by_record_id_in_place(
        callback.message,
        update_id,
        selected_fields=selected_fields,
        reviewer_user_id=str(callback.from_user.id),
    )


@router.callback_query(F.data.startswith(UPDATE_APPLY_CONFIRM_CALLBACK_PREFIX))
async def handle_apply_update_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, _list_limit = parse_update_action_callback_payload(
        (callback.data or "").removeprefix(UPDATE_APPLY_CONFIRM_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await apply_update_by_record_id_in_place(
        callback.message,
        update_id,
        reviewer_user_id=str(callback.from_user.id),
    )


@router.callback_query(F.data.startswith(UPDATE_REJECT_CONFIRM_CALLBACK_PREFIX))
async def handle_reject_update_confirm_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    update_id, _list_limit = parse_update_action_callback_payload(
        (callback.data or "").removeprefix(UPDATE_REJECT_CONFIRM_CALLBACK_PREFIX)
    )
    if update_id is None:
        await callback.answer("Confirm button is invalid.", show_alert=True)
        return
    if callback.message is None:
        await callback.answer("Message is not available.", show_alert=True)
        return

    await callback.answer()
    await reject_update_by_record_id_in_place(
        callback.message,
        update_id,
        reviewer_user_id=str(callback.from_user.id),
    )


@router.callback_query(F.data == UPDATE_REVIEW_CANCEL_CALLBACK)
async def handle_update_review_cancel_callback(callback: CallbackQuery) -> None:
    if not await require_moderator_callback(callback):
        return

    await callback.answer("Cancelled.")
    if callback.message is not None:
        await callback.message.answer("Cancelled.")


@router.callback_query(F.data == PANEL_CANCEL_CALLBACK)
async def handle_panel_cancel_callback(callback: CallbackQuery, state: FSMContext) -> None:
    if not await require_moderator_callback(callback):
        return

    await state.clear()
    await callback.answer("Cancelled.")
    if callback.message is not None:
        await callback.message.edit_text("Cancelled.", reply_markup=None)


@router.message(Command("reject_suggestion"))
async def handle_reject_suggestion(
    message: Message,
    command: CommandObject,
    state: FSMContext,
) -> None:
    if not await require_moderator(message):
        return

    suggestion_id = command.args or ""
    if not suggestion_id.strip():
        await state.set_state(RejectSuggestionStates.suggestion_id)
        await message.answer(
            waiting_prompt("Reject suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await reject_suggestion_by_record_id(message, suggestion_id)


@router.message(RejectSuggestionStates.suggestion_id)
async def handle_reject_suggestion_id(message: Message, state: FSMContext) -> None:
    if not await require_moderator(message):
        return
    if await reject_guided_flow_command(message):
        return

    suggestion_id = text_value(message)
    if not suggestion_id:
        await message.answer(
            waiting_prompt("Reject suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    await reject_suggestion_by_record_id(message, suggestion_id)
    await state.clear()


async def reject_suggestion_by_record_id(message: Message, value: str) -> None:
    suggestion_id = parse_suggestion_record_id(value)
    if suggestion_id is None:
        await message.answer(
            waiting_prompt("Reject suggestion", "suggestion ID", example="#1"),
            reply_markup=dialog_keyboard(),
        )
        return

    suggestion = find_pending_suggestion_by_record_id(suggestion_id)
    if suggestion is None:
        await message.answer(
            f"I could not find pending suggestion "
            f"<code>{format_suggestion_handle(suggestion_id)}</code>."
        )
        return

    await message.answer(
        format_suggestion_reject_confirmation(
            suggestion,
            label=format_suggestion_handle(suggestion.id),
        ),
        reply_markup=suggestion_reject_confirmation_keyboard(
            suggestion,
            list_limit=SUGGESTION_LIST_DEFAULT_LIMIT,
        ),
    )


def pending_suggestion_by_number(number: int):
    if number < 1:
        return None

    suggestions = list_event_suggestions(limit=EVENT_SUGGESTION_MAX_PENDING_TOTAL)
    if number > len(suggestions):
        return None

    return suggestions[number - 1]


def find_pending_suggestion_by_record_id(suggestion_id: int):
    return get_event_suggestion(suggestion_id, status="pending")


def find_pending_update_by_record_id(update_id: int) -> ProposedEventUpdateRecord | None:
    return get_proposed_event_update(update_id, status="pending")


def text_value(message: Message) -> str:
    return (message.text or "").strip()


def is_unexpected_guided_flow_command(value: str) -> bool:
    stripped = value.strip()
    if not stripped.startswith("/"):
        return False

    command = stripped.split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()
    return command != "/cancel"


async def reject_guided_flow_command(message: Message) -> bool:
    if not is_unexpected_guided_flow_command(text_value(message)):
        return False

    await message.answer(
        "Only /cancel is accepted while this dialog is waiting for input.",
        reply_markup=dialog_keyboard(),
    )
    return True


def message_user_id(message: Message) -> str | None:
    user = getattr(message, "from_user", None)
    user_id = getattr(user, "id", None)
    return str(user_id) if user_id is not None else None


async def require_moderator_callback(callback: CallbackQuery) -> bool:
    user = callback.from_user
    if is_moderator_account(user.id, user.username):
        return True

    await callback.answer("This action is available to moderators only.", show_alert=True)
    return False


def draft_to_state(draft: EventDraft) -> dict[str, object]:
    return {
        "source_url": draft.source_url,
        "name": draft.name,
        "public_id": draft.public_id,
        "city": draft.city,
        "country": draft.country,
        "timezone": draft.timezone,
        "event_date": draft.event_date,
        "distances": draft.distances,
        "regions": draft.regions,
        "official_url": draft.official_url,
        "registration_url": draft.registration_url,
        "registration_status": "unknown",
        "registration_open_at": None,
        "registration_open_precision": "unknown",
        "registration_close_at": None,
        "registration_url_candidates": draft.registration_url_candidates,
    }


def format_draft_summary(draft: EventDraft) -> str:
    return "\n".join(
        [
            format_major_title("Draft extracted from URL"),
            "",
            "Draft fields come from a structured extractor over fetched page evidence.",
            format_field_line("Confidence", f"{draft.confidence:.2f}"),
            format_evidence_for_display(draft.evidence),
            "",
            "I will ask you to confirm or correct each field.",
        ]
    )


def format_registration_update_result(update: RegistrationWindowUpdateResult) -> str:
    lines = [
        "<b>Registration scan</b>",
        "",
        format_field_line("Status", update.registration_status),
        format_field_line("Confidence", f"{update.confidence:.2f}"),
    ]
    if update.registration_open_at:
        lines.append(format_field_line("Opens", update.registration_open_at))
    if update.registration_url:
        lines.append(format_field_line("Registration URL", update.registration_url))
    if update.event_date:
        lines.append(format_field_line("Event date", update.event_date))
    if update.proposed_update_id is not None:
        lines.append("")
        lines.append(
            "Created proposed update "
            f"#{update.proposed_update_id} for moderator confirmation."
        )
    elif update.applied:
        lines.append("")
        lines.append("High-confidence update was applied automatically.")
    else:
        lines.append("")
        lines.append("No registration announcement detected yet.")

    lines.extend(["", format_evidence_for_display(update.evidence)])
    return "\n".join(lines)


def _append_researcher_source_check(
    lines: list[str],
    evidence: str | tuple[str, ...] | list[str] | None,
) -> None:
    block = format_researcher_source_check(parse_researcher_provenance(evidence))
    if block:
        lines.extend(["", block])


def _researcher_suggestion_evidence(suggestion) -> str | None:
    if any(
        (
            suggestion.submitter_user_id,
            suggestion.submitter_username,
            suggestion.submitter_display_name,
        )
    ):
        return None
    return suggestion.note if parse_researcher_provenance(suggestion.note) else None


def format_evidence_for_display(evidence: str) -> str:
    formatted = evidence.strip()
    if not formatted:
        return "unknown"

    formatted = re.sub(
        r"(Stored snapshot:\s+)(?:/[^\s]+/)?([^/\s]+\.json)(?=\.|\s|$)",
        r"\1\2",
        formatted,
    )
    formatted = re.sub(
        r"\s+(?=(?:Text hash|Stored snapshot|Title|Extractor provider|"
        r"Registration extractor provider|AI provider|Page blocked|Detected registration status|"
        r"Detected registration opening date|Detected registration URL|Detected event date):)",
        "\n",
        formatted,
    )
    formatted = re.sub(r"\s+(?=Extractor provider .+ failed:)", "\n", formatted)
    formatted = re.sub(r"\s+(?=Falling back to )", "\n", formatted)
    formatted = re.sub(r"\s+(?=AI provider is )", "\n", formatted)
    formatted = re.sub(r"\"\s+\"", "\"\n\"", formatted)
    formatted = re.sub(r"(?<=[.!?])\s+(?=\")", "\n", formatted)
    formatted = re.sub(r"(?<=[.!?]\")\s+(?=\")", "\n", formatted)
    formatted = re.sub(r"(?<!:)\s+(?=https?://)", "\n", formatted)
    formatted = re.sub(r"[ \t]+", " ", formatted)
    formatted = re.sub(r"\n+", "\n", formatted)
    source_check, key_info = split_evidence_lines(
        compact_evidence_lines(formatted.splitlines())
    )
    return format_evidence_sections(source_check, key_info)


def compact_evidence_lines(lines: list[str]) -> list[str]:
    compacted: list[str] = []
    quoted_snippets = 0
    max_quoted_snippets = 4

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("Text hash:"):
            continue
        if is_empty_detected_evidence_line(line):
            continue
        if line.startswith('"') and line.endswith('"'):
            if quoted_snippets >= max_quoted_snippets:
                continue
            quoted_snippets += 1
        compacted.append(line)

    return compacted


def split_evidence_lines(lines: list[str]) -> tuple[list[str], list[str]]:
    source_check: list[str] = []
    key_info: list[str] = []

    for line in lines:
        if is_source_check_line(line):
            source_check.append(format_source_check_line(line))
        else:
            key_info.append(line)

    return source_check, key_info


def is_source_check_line(line: str) -> bool:
    return (
        line.startswith("Fetched page snapshot with status ")
        or line.startswith("Stored snapshot:")
        or line.startswith("Title:")
        or line.startswith("Extractor provider")
        or line.startswith("Registration extractor provider")
        or line.startswith("AI provider")
        or line.startswith("Falling back to ")
        or line.startswith("Page fetch failed:")
        or line.startswith("Page blocked:")
        or line.startswith("Fallback extraction")
    )


def format_source_check_line(line: str) -> str:
    if match := re.fullmatch(r"Fetched page snapshot with status (\d+)\.", line):
        status = int(match.group(1))
        if 200 <= status <= 299:
            return f"Fetched page OK (status {status})."
        return f"Fetched page returned status {status}."

    if line.startswith("Stored snapshot:"):
        return format_source_check_field(line, "Stored snapshot:", "Snapshot")

    if line.startswith("Title:"):
        return format_source_check_field(line, "Title:", "Title")

    if line.startswith("Extractor provider:"):
        return format_source_check_field(line, "Extractor provider:", "Provider")

    if line.startswith("Registration extractor provider:"):
        return format_source_check_field(
            line,
            "Registration extractor provider:",
            "Provider",
        )

    if line.startswith("Page fetch failed:"):
        return format_source_check_field(line, "Page fetch failed:", "Page fetch failed")

    if line.startswith("Page blocked:"):
        value = line.removeprefix("Page blocked:").strip().removesuffix(".")
        return f"⚠️ <b>Warning</b>: {escape(value, quote=False)}"

    return line


def format_source_check_field(line: str, prefix: str, label: str) -> str:
    value = line.removeprefix(prefix).strip().removesuffix(".")
    return format_field_line(label, value)


def format_evidence_sections(source_check: list[str], key_info: list[str]) -> str:
    lines = ["<b>Source check</b>"]
    lines.extend(
        line if line.startswith(("<b>", "⚠️ <b>")) else escape(line, quote=False)
        for line in source_check or ["unknown"]
    )

    if key_info:
        lines.append("")
        lines.append("<b>Detected info</b>")
        lines.extend(format_key_info_line(line) for line in key_info)

    return "\n".join(lines)


def is_empty_detected_evidence_line(line: str) -> bool:
    return re.fullmatch(r"Detected [^:]+:\s*\.?", line) is not None


def format_key_info_line(line: str) -> str:
    if match := re.fullmatch(r"(Detected [^:]+):\s*(.+)", line):
        value = match.group(2).strip().removesuffix(".")
        return format_field_line(detected_info_label(match.group(1)), value)

    return escape(line, quote=False)


def detected_info_label(label: str) -> str:
    cleaned = label.removeprefix("Detected ").strip()
    return cleaned[:1].upper() + cleaned[1:]


async def warn_existing_url_events(message: Message, url: str) -> None:
    events = list_events_by_url(url)
    if not events:
        return

    await message.answer(format_existing_url_warning(events))


def format_existing_url_warning(events) -> str:
    event_lines = [
        f"- <b>{escape(event.name)}</b> | "
        f"{format_field_line('ID', event.public_id, kind='id')}"
        for event in events
    ]
    return (
        "⚠️ <b>Warning</b>: this URL is already used by tracked event(s):\n"
        + "\n".join(event_lines)
        + "\n\nYou can continue if this is another distance or related event."
    )


def format_existing_id_warning(event) -> str:
    return (
        "⚠️ <b>Warning</b>: this event ID is already tracked:\n"
        f"<b>{escape(event.name)}</b>\n"
        f"{format_field_line('ID', event.public_id, kind='id')}\n"
        "Send another public ID, or cancel this flow."
    )


def format_archive_event_confirmation(event) -> str:
    return (
        f"{format_event_detail(event)}\n\n"
        "<b>Confirm archive</b>\n"
        "This will archive exactly this event from active tracking."
    )


def is_archive_event_confirmation(value: str) -> bool:
    return value.strip().casefold() == "archive"


def archive_event_confirmation_keyboard(public_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Archive",
                    callback_data=archive_event_callback(public_id),
                ),
                InlineKeyboardButton(
                    text="Cancel",
                    callback_data=PANEL_CANCEL_CALLBACK,
                ),
            ]
        ]
    )


def archive_event_callback(public_id: str) -> str:
    return f"{ARCHIVE_EVENT_CALLBACK_PREFIX}{public_id}"


def format_restore_event_confirmation(event: TrackedEvent) -> str:
    return (
        f"{format_event_detail(event)}\n\n"
        "<b>Confirm restore</b>\n"
        "This will restore exactly this event to active tracking."
    )


def restore_event_confirmation_keyboard(
    public_id: str,
    *,
    list_limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Confirm",
                    callback_data=restore_event_confirm_callback(public_id),
                ),
                InlineKeyboardButton(
                    text="Back",
                    callback_data=archive_show_callback(
                        public_id,
                        limit=list_limit,
                    ),
                ),
            ]
        ]
    )


def restore_event_confirm_callback(public_id: str) -> str:
    return f"{RESTORE_CONFIRM_CALLBACK_PREFIX}{public_id}"


def format_delete_event_confirmation(event) -> str:
    return (
        f"{format_event_detail(event)}\n\n"
        "<b>Delete event</b>\n"
        "This will permanently delete exactly this event and its stored child rows. "
        "It will not appear in the archive and cannot be restored."
    )


def is_delete_event_confirmation(value: str, event_id: str) -> bool:
    return value.strip().casefold() == f"delete {event_id.casefold()}"


def format_delete_event_final_confirmation(event) -> str:
    return (
        f"{format_event_detail(event)}\n\n"
        "<b>Confirm permanent deletion</b>\n"
        "This is the final confirmation. The event cannot be restored after this action."
    )


def delete_event_preview_keyboard(public_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Delete",
                    callback_data=delete_event_callback(public_id),
                ),
                InlineKeyboardButton(
                    text="Cancel",
                    callback_data=PANEL_CANCEL_CALLBACK,
                ),
            ]
        ]
    )


def delete_event_confirmation_keyboard(public_id: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Confirm",
                    callback_data=delete_event_confirm_callback(public_id),
                ),
                InlineKeyboardButton(
                    text="Back",
                    callback_data=delete_event_preview_callback(public_id),
                ),
            ]
        ]
    )


def delete_event_preview_callback(public_id: str) -> str:
    return f"{DELETE_EVENT_PREVIEW_CALLBACK_PREFIX}{public_id}"


def delete_event_callback(public_id: str) -> str:
    return f"{DELETE_EVENT_CALLBACK_PREFIX}{public_id}"


def delete_event_confirm_callback(public_id: str) -> str:
    return f"{DELETE_EVENT_CONFIRM_CALLBACK_PREFIX}{public_id}"


def format_archived_events(archived_events) -> str:
    lines = [format_major_title("Archived events")]
    for archived in archived_events:
        lines.extend(["", format_archived_event_card(archived)])

    return "\n".join(lines)


def format_archived_event_card(archived) -> str:
    event = archived.event
    date = event.event_date or "date TBA"
    removed_at = archived.removed_at or "unknown"
    return "\n".join(
        [
            f"<b>{escape(event.name)}</b>",
            format_field_line("ID", event.public_id, kind="id"),
            f"{escape(event.location)} | {escape(event.distance_label)} | {escape(date)}",
            format_field_line("Removed", removed_at),
        ]
    )


def archived_events_keyboard(
    archived_events,
    *,
    limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=archive_show_callback(
                        archived.event.public_id,
                        limit=limit,
                    ),
                )
            ]
            for archived in archived_events
        ]
    )


def archived_event_card_keyboard(
    archived,
    *,
    limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=archive_show_callback(
                        archived.event.public_id,
                        limit=limit,
                    ),
                )
            ]
        ]
    )


def format_archived_event_detail(archived) -> str:
    event = archived.event
    removed_at = archived.removed_at or "unknown"
    return "\n".join(
        [
            format_event_detail(event),
            "",
            format_field_line("Removed", removed_at),
        ]
    )


def archived_event_detail_keyboard(
    archived,
    *,
    list_limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    public_id = archived.event.public_id
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Restore",
                    callback_data=restore_event_callback(
                        public_id,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Delete",
                    callback_data=delete_event_preview_callback(public_id),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Back",
                    callback_data=archive_list_callback(list_limit),
                ),
            ],
        ]
    )


def archive_list_callback(limit: int = ARCHIVE_LIST_DEFAULT_LIMIT) -> str:
    return f"{ARCHIVE_LIST_CALLBACK_PREFIX}{limit}"


def archive_show_callback(
    public_id: str,
    *,
    limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{ARCHIVE_SHOW_CALLBACK_PREFIX}{public_id}:{limit}"


def restore_event_callback(
    public_id: str,
    *,
    list_limit: int = ARCHIVE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{RESTORE_EVENT_CALLBACK_PREFIX}{public_id}:{list_limit}"


def format_suggestion_queue(
    suggestions,
    *,
    start: int = 1,
    title: str = "Pending suggestion",
) -> str:
    lines = [format_major_title(title)]
    for suggestion in suggestions:
        lines.extend(["", format_suggestion_card(suggestion)])

    return "\n".join(lines)


def format_suggestion_card(suggestion) -> str:
    lines = [
        f"<b>Suggestion {escape(format_suggestion_handle(suggestion.id))}</b>",
        format_field_line("Name", suggestion.event_name),
        format_field_line(
            "Distances",
            format_distance_input_value(suggestion.distances),
            kind="tag",
        ),
    ]

    return "\n".join(lines)


def suggestion_queue_keyboard(
    suggestions,
    *,
    start: int = 1,
    limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=suggestion_show_callback(
                        suggestion.id,
                        list_limit=limit,
                    ),
                ),
            ]
            for suggestion in suggestions
        ]
    )


def suggestion_show_keyboard(
    suggestion,
    *,
    limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=suggestion_show_callback(
                        suggestion.id,
                        list_limit=limit,
                    ),
                ),
            ]
        ]
    )


def format_suggestion_detail(suggestion, *, sequence: int | None = None) -> str:
    researcher_evidence = _researcher_suggestion_evidence(suggestion)
    provenance = parse_researcher_provenance(researcher_evidence)
    if provenance is not None:
        lines = [
            format_major_title(f"Suggestion {format_suggestion_handle(suggestion.id)}"),
            "",
            format_bounded_field_line("Name", suggestion.event_name, max_html_chars=300),
            format_bounded_field_line(
                "URL",
                suggestion.url or "unknown",
                max_html_chars=600,
            ),
            format_bounded_field_line(
                "Distances",
                format_distance_input_value(suggestion.distances),
                kind="tag",
                max_html_chars=200,
            ),
            format_field_line("From", format_submitter(suggestion)),
        ]
        _append_researcher_source_check(lines, researcher_evidence)
        return "\n".join(lines)

    lines = [
        format_major_title(f"Suggestion {format_suggestion_handle(suggestion.id)}"),
        "",
        format_field_line("Name", suggestion.event_name),
        format_field_line("URL", suggestion.url or "unknown"),
        format_field_line(
            "Distances",
            format_distance_input_value(suggestion.distances),
            kind="tag",
        ),
        format_field_line(
            "From",
            format_submitter(suggestion),
            kind=format_submitter_kind(suggestion),
        ),
    ]
    if suggestion.note:
        lines.append(format_field_line("Note", suggestion.note))

    return "\n".join(lines)


def suggestion_detail_keyboard(
    suggestion,
    *,
    sequence: int | None = None,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Apply",
                    callback_data=suggestion_add_callback(
                        suggestion.id,
                        sequence=sequence,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Reject",
                    callback_data=suggestion_remove_callback(
                        suggestion.id,
                        sequence=sequence,
                        list_limit=list_limit,
                    ),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Back",
                    callback_data=suggestion_card_callback(
                        suggestion.id,
                        list_limit=list_limit,
                    ),
                ),
            ],
        ]
    )


def format_suggestion_reject_confirmation(suggestion, *, label: str | None = None) -> str:
    lines = [
        "<b>Confirm suggestion rejection</b>",
        format_field_line("Suggestion", suggestion.event_name),
    ]
    if label:
        lines.append(format_field_line("Suggestion ID", label, kind="id"))
    lines.append("This will reject the suggestion and remove it from the pending queue.")
    _append_researcher_source_check(
        lines,
        _researcher_suggestion_evidence(suggestion),
    )
    return "\n".join(lines)


def format_event_added_confirmation(
    *,
    from_suggestion: bool,
    suggestion_note: str | None = None,
) -> str:
    lines = [
        (
            "Event added. The source suggestion was removed from the pending queue."
            if from_suggestion
            else "Event added."
        )
    ]
    _append_researcher_source_check(
        lines,
        suggestion_note,
    )
    return "\n".join(lines)


def suggestion_reject_confirmation_keyboard(
    suggestion,
    *,
    sequence: int | None = None,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Confirm",
                    callback_data=suggestion_remove_confirm_callback(
                        suggestion.id,
                        sequence=sequence,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Back",
                    callback_data=suggestion_show_callback(
                        suggestion.id,
                        list_limit=list_limit,
                    ),
                ),
            ]
        ]
    )


def suggestion_list_callback(limit: int = SUGGESTION_LIST_DEFAULT_LIMIT) -> str:
    return f"{SUGGESTION_LIST_CALLBACK_PREFIX}{limit}"


def suggestion_card_callback(
    suggestion_id: int,
    *,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{SUGGESTION_CARD_CALLBACK_PREFIX}{suggestion_id}:{list_limit}"


def suggestion_show_callback(
    suggestion_id: int,
    *,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{SUGGESTION_SHOW_CALLBACK_PREFIX}{suggestion_id}:{list_limit}"


def suggestion_add_callback(
    suggestion_id: int,
    *,
    sequence: int | None = None,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> str:
    sequence_value = sequence or 0
    return f"{SUGGESTION_ADD_CALLBACK_PREFIX}{suggestion_id}:{sequence_value}:{list_limit}"


def suggestion_remove_callback(
    suggestion_id: int,
    *,
    sequence: int | None = None,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> str:
    sequence_value = sequence or 0
    return f"{SUGGESTION_REMOVE_CALLBACK_PREFIX}{suggestion_id}:{sequence_value}:{list_limit}"


def suggestion_remove_confirm_callback(
    suggestion_id: int,
    *,
    sequence: int | None = None,
    list_limit: int = SUGGESTION_LIST_DEFAULT_LIMIT,
) -> str:
    sequence_value = sequence or 0
    return (
        f"{SUGGESTION_REMOVE_CONFIRM_CALLBACK_PREFIX}"
        f"{suggestion_id}:{sequence_value}:{list_limit}"
    )


def format_moderator_status(*, pending_updates: int, pending_suggestions: int) -> str:
    return "\n".join(
        [
            format_major_title("Todo"),
            format_field_line("Updates", pending_updates),
            format_field_line("Suggestion", pending_suggestions),
        ]
    )


def todo_keyboard(*, pending_updates: int, pending_suggestions: int) -> InlineKeyboardMarkup | None:
    rows = []
    if pending_updates:
        rows.append(
            [
                InlineKeyboardButton(
                    text="List update",
                    callback_data=proposed_update_list_callback(),
                )
            ]
        )
    if pending_suggestions:
        rows.append(
            [
                InlineKeyboardButton(
                    text="List suggestion",
                    callback_data=suggestion_list_callback(),
                )
            ]
        )

    return InlineKeyboardMarkup(inline_keyboard=rows) if rows else None


def format_proposed_update_list(updates: tuple[ProposedEventUpdateRecord, ...]) -> str:
    lines = [format_major_title("Pending updates")]
    for update in updates:
        lines.extend(["", format_proposed_update_card(update)])

    return "\n".join(lines)


def format_proposed_update_card(update: ProposedEventUpdateRecord) -> str:
    lines = [
        f"<b>Update {escape(format_update_handle(update.id))}</b>",
        format_field_line("Event ID", update.event_id, kind="id"),
        format_field_line("Type", update.update_type),
        format_field_line("Confidence", f"{update.confidence:.2f}"),
    ]
    changed_fields = proposed_update_changed_field_names(update)
    if changed_fields:
        lines.append(format_field_line("Fields", ", ".join(changed_fields)))
    elif update.change_summary and not is_generic_update_summary(update.change_summary):
        lines.append(format_field_line("Summary", update.change_summary))

    return "\n".join(lines)


def proposed_update_list_keyboard(
    updates: tuple[ProposedEventUpdateRecord, ...],
    *,
    limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=proposed_update_show_callback(
                        update.id,
                        list_limit=limit,
                    ),
                )
            ]
            for update in updates
        ]
    )


def proposed_update_show_keyboard(
    update: ProposedEventUpdateRecord,
    *,
    limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=proposed_update_show_callback(
                        update.id,
                        list_limit=limit,
                    ),
                )
            ]
        ]
    )


def proposed_update_detail_keyboard(
    update: ProposedEventUpdateRecord,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Apply",
                    callback_data=proposed_update_apply_callback(
                        update.id,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Partial",
                    callback_data=proposed_update_partial_callback(
                        update.id,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Reject",
                    callback_data=proposed_update_reject_callback(
                        update.id,
                        list_limit=list_limit,
                    ),
                ),
            ],
            [
                InlineKeyboardButton(
                    text="Back",
                    callback_data=proposed_update_card_callback(
                        update.id,
                        list_limit=list_limit,
                    ),
                ),
            ]
        ]
    )


def format_update_partial_selection(
    update: ProposedEventUpdateRecord,
    *,
    selected_fields: tuple[str, ...],
) -> str:
    lines = [
        format_major_title(f"Partial apply {format_update_handle(update.id)}"),
        "",
        format_field_line("Event ID", update.event_id, kind="id"),
        "Choose fields to apply.",
    ]
    if selected_fields:
        lines.append(format_field_line("Selected", ", ".join(selected_fields)))

    changes = proposed_update_changes(update)
    if changes:
        lines.extend(["", "<b>What's changed</b>"])
        lines.extend(changes)

    return "\n".join(lines)


def proposed_update_partial_keyboard(
    update: ProposedEventUpdateRecord,
    *,
    selected_fields: tuple[str, ...],
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    changed_fields = proposed_update_changed_field_names(update)
    selected = set(selected_fields)
    rows = [
        [
            InlineKeyboardButton(
                text=f"{'✅' if field in selected else '⬜'} {field}",
                callback_data=proposed_update_partial_toggle_callback(
                    update.id,
                    field,
                    selected_fields=selected_fields,
                    changed_fields=changed_fields,
                    list_limit=list_limit,
                ),
            )
        ]
        for field in changed_fields
    ]

    action_row = []
    if selected_fields:
        action_row.append(
            InlineKeyboardButton(
                text="Confirm",
                callback_data=proposed_update_partial_confirm_callback(
                    update.id,
                    selected_fields=selected_fields,
                    changed_fields=changed_fields,
                    list_limit=list_limit,
                ),
            )
        )
    action_row.append(
        InlineKeyboardButton(
            text="Back",
            callback_data=proposed_update_show_callback(
                update.id,
                list_limit=list_limit,
            ),
        )
    )
    rows.append(action_row)
    return InlineKeyboardMarkup(inline_keyboard=rows)


def format_update_partial_confirmation(
    update: ProposedEventUpdateRecord,
    *,
    selected_fields: tuple[str, ...],
) -> str:
    changed_fields = proposed_update_changed_field_names(update)
    remaining_fields = tuple(field for field in changed_fields if field not in selected_fields)
    lines = [
        format_major_title(f"Confirm partial apply {format_update_handle(update.id)}"),
        "",
        format_field_line("Event ID", update.event_id, kind="id"),
        format_field_line("Apply fields", ", ".join(selected_fields)),
    ]
    if remaining_fields:
        lines.append(format_field_line("Keep pending", ", ".join(remaining_fields)))
        lines.append(
            "This will apply selected fields and create a new pending update for the rest."
        )
    else:
        lines.append("This will apply selected fields and close the update.")

    _append_researcher_source_check(
        lines,
        update.evidence,
    )
    return "\n".join(lines)


def format_partial_apply_result(result) -> str:
    lines = [
        f"Partially applied update <code>{escape(format_update_handle(result.update.id))}</code> "
        f"to <b>{escape(result.event.name)}</b>.",
        format_field_line("Applied fields", ", ".join(result.applied_fields)),
    ]
    if result.follow_up_update is not None:
        lines.append(
            format_field_line(
                "New pending update",
                format_update_handle(result.follow_up_update.id),
                kind="id",
            )
        )
        lines.append(format_field_line("Remaining fields", ", ".join(result.remaining_fields)))

    lines.extend(["", format_event_detail(result.event)])
    return "\n".join(lines)


def proposed_update_partial_confirmation_keyboard(
    update: ProposedEventUpdateRecord,
    *,
    selected_fields: tuple[str, ...],
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    changed_fields = proposed_update_changed_field_names(update)
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Confirm",
                    callback_data=proposed_update_partial_apply_callback(
                        update.id,
                        selected_fields=selected_fields,
                        changed_fields=changed_fields,
                        list_limit=list_limit,
                    ),
                ),
                InlineKeyboardButton(
                    text="Cancel",
                    callback_data=proposed_update_partial_callback(
                        update.id,
                        selected_fields=selected_fields,
                        changed_fields=changed_fields,
                        list_limit=list_limit,
                    ),
                ),
            ]
        ]
    )


def format_update_review_confirmation(
    update: ProposedEventUpdateRecord,
    *,
    action: str,
) -> str:
    action_label = "Apply" if action == "apply" else "Reject"
    consequence = (
        "This will apply the proposed fields to the tracked event."
        if action == "apply"
        else "This will reject the proposal without changing event data."
    )
    lines = [
        format_major_title(
            f"Confirm {action_label.lower()} {format_update_handle(update.id)}"
        ),
        "",
        format_field_line("Event ID", update.event_id, kind="id"),
        format_field_line("Type", update.update_type),
    ]
    lines.append(consequence)
    _append_researcher_source_check(
        lines,
        update.evidence,
    )
    return "\n".join(lines)


def proposed_update_confirmation_keyboard(
    update: ProposedEventUpdateRecord,
    *,
    action: str,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> InlineKeyboardMarkup:
    confirm_callback = (
        proposed_update_apply_confirm_callback(
            update.id,
            list_limit=list_limit,
        )
        if action == "apply"
        else proposed_update_reject_confirm_callback(
            update.id,
            list_limit=list_limit,
        )
    )
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Confirm",
                    callback_data=confirm_callback,
                ),
                InlineKeyboardButton(
                    text="Cancel",
                    callback_data=proposed_update_show_callback(
                        update.id,
                        list_limit=list_limit,
                    ),
                ),
            ]
        ]
    )


def proposed_update_list_callback(limit: int = UPDATE_LIST_DEFAULT_LIMIT) -> str:
    return f"{UPDATE_LIST_CALLBACK_PREFIX}{limit}"


def proposed_update_card_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_CARD_CALLBACK_PREFIX}{update_id}:{list_limit}"


def proposed_update_show_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_SHOW_CALLBACK_PREFIX}{update_id}:{list_limit}"


def proposed_update_apply_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_APPLY_CALLBACK_PREFIX}{update_id}:{list_limit}"


def proposed_update_reject_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_REJECT_CALLBACK_PREFIX}{update_id}:{list_limit}"


def proposed_update_partial_callback(
    update_id: int,
    *,
    selected_fields: tuple[str, ...] = (),
    changed_fields: list[str] | tuple[str, ...] = (),
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    mask = encode_update_field_mask(selected_fields, changed_fields)
    return f"{UPDATE_PARTIAL_CALLBACK_PREFIX}{update_id}:{list_limit}:{mask}"


def proposed_update_partial_toggle_callback(
    update_id: int,
    field: str,
    *,
    selected_fields: tuple[str, ...],
    changed_fields: list[str] | tuple[str, ...],
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    selected = set(selected_fields)
    if field in selected:
        selected.remove(field)
    else:
        selected.add(field)
    mask = encode_update_field_mask(tuple(selected), changed_fields)
    return f"{UPDATE_PARTIAL_TOGGLE_CALLBACK_PREFIX}{update_id}:{list_limit}:{mask}"


def proposed_update_partial_confirm_callback(
    update_id: int,
    *,
    selected_fields: tuple[str, ...],
    changed_fields: list[str] | tuple[str, ...],
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    mask = encode_update_field_mask(selected_fields, changed_fields)
    return f"{UPDATE_PARTIAL_CONFIRM_CALLBACK_PREFIX}{update_id}:{list_limit}:{mask}"


def proposed_update_partial_apply_callback(
    update_id: int,
    *,
    selected_fields: tuple[str, ...],
    changed_fields: list[str] | tuple[str, ...],
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    mask = encode_update_field_mask(selected_fields, changed_fields)
    return f"{UPDATE_PARTIAL_APPLY_CALLBACK_PREFIX}{update_id}:{list_limit}:{mask}"


def proposed_update_apply_confirm_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_APPLY_CONFIRM_CALLBACK_PREFIX}{update_id}:{list_limit}"


def proposed_update_reject_confirm_callback(
    update_id: int,
    *,
    list_limit: int = UPDATE_LIST_DEFAULT_LIMIT,
) -> str:
    return f"{UPDATE_REJECT_CONFIRM_CALLBACK_PREFIX}{update_id}:{list_limit}"


def format_proposed_update_detail(
    update: ProposedEventUpdateRecord,
) -> str:
    provenance = parse_researcher_provenance(update.evidence)
    lines = [
        format_major_title(f"Update {format_update_handle(update.id)}"),
        "",
        format_field_line("Event ID", update.event_id, kind="id"),
        format_field_line("Type", update.update_type),
        format_field_line("Confidence", f"{update.confidence:.2f}"),
    ]
    if update.change_summary and not is_generic_update_summary(update.change_summary):
        if provenance is None:
            lines.append(format_field_line("Summary", update.change_summary))
        else:
            lines.append(
                format_bounded_field_line(
                    "Summary",
                    update.change_summary,
                    max_html_chars=400,
                )
            )

    changes = proposed_update_changes(
        update,
        max_value_html_chars=450 if provenance is not None else None,
    )
    if changes:
        lines.extend(["", "<b>What's changed</b>"])
        lines.extend(changes)

    if provenance is not None:
        _append_researcher_source_check(lines, update.evidence)
    elif update.evidence:
        lines.extend(["", format_evidence_for_display(" ".join(update.evidence))])

    return "\n".join(lines)


def proposed_update_changes(
    update: ProposedEventUpdateRecord,
    *,
    max_value_html_chars: int | None = None,
) -> list[str]:
    lines = []
    for field, proposed_value in update.proposed_fields.items():
        if is_empty_proposed_update_value(proposed_value):
            continue
        current_value = update.current_fields.get(field)
        if current_value == proposed_value:
            continue
        lines.append(
            "\n".join(
                [
                    f"- <b>{escape(str(field))}</b>",
                    "  "
                    + format_removed_json_field_value(
                        current_value,
                        field=field,
                        max_html_chars=max_value_html_chars,
                    ),
                    "  "
                    + format_json_field_value(
                        proposed_value,
                        field=field,
                        max_html_chars=max_value_html_chars,
                    ),
                ]
            )
        )

    return lines


def proposed_update_changed_field_names(update: ProposedEventUpdateRecord) -> list[str]:
    return [
        field
        for field, proposed_value in update.proposed_fields.items()
        if not is_empty_proposed_update_value(proposed_value)
        and update.current_fields.get(field) != proposed_value
    ]


def encode_update_field_mask(
    selected_fields: tuple[str, ...],
    changed_fields: list[str] | tuple[str, ...],
) -> int:
    selected = set(selected_fields)
    mask = 0
    for index, field in enumerate(changed_fields):
        if field in selected:
            mask |= 1 << index
    return mask


def decode_update_field_mask(
    mask: int,
    changed_fields: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        field for index, field in enumerate(changed_fields) if mask & (1 << index)
    )


def is_empty_proposed_update_value(value: object) -> bool:
    return value is None or value == "" or value == "unknown"


def is_generic_update_summary(summary: str) -> bool:
    return summary.startswith("Registration update proposed:")


def format_json_field_value(
    value: object,
    *,
    field: str,
    max_html_chars: int | None = None,
) -> str:
    if value in {None, ""}:
        raw_value = "unknown"
    elif isinstance(value, tuple | list):
        raw_value = ",".join(str(item) for item in value) or "unknown"
    else:
        raw_value = str(value)

    rendered_value = (
        escape(raw_value)
        if max_html_chars is None
        else bounded_html_escape(raw_value, max_html_chars=max_html_chars)
    )
    kind = field_value_kind(field)
    if kind == "id":
        return f"<code>{rendered_value}</code>"
    if kind == "tag":
        return f"<u>{rendered_value}</u>"
    return rendered_value


def format_removed_json_field_value(
    value: object,
    *,
    field: str,
    max_html_chars: int | None = None,
) -> str:
    return (
        "<s>"
        f"{format_json_field_value(value, field=field, max_html_chars=max_html_chars)}"
        "</s>"
    )


def format_optional_code(value: str | None) -> str:
    return f"<code>{escape(value)}</code>" if value else "<code>unknown</code>"


def field_value_kind(field: str) -> str:
    normalized = field.casefold()
    if normalized in {"id", "event_id", "public_id"} or normalized.endswith("_id"):
        return "id"
    if normalized in {"distance", "distances", "region_tags", "regions", "tags"}:
        return "tag"
    return "text"


def format_update_handle(update_id: int) -> str:
    return f"#{update_id}"


def format_suggestion_handle(suggestion_id: int) -> str:
    return f"#{suggestion_id}"


def format_region_tags(regions: tuple[str, ...]) -> str | None:
    return ",".join(regions) if regions else None


def format_submitter(suggestion) -> str:
    if suggestion.submitter_username:
        return f"@{suggestion.submitter_username}"
    if suggestion.submitter_display_name:
        return str(suggestion.submitter_display_name)
    if suggestion.submitter_user_id:
        return str(suggestion.submitter_user_id)

    if _researcher_suggestion_evidence(suggestion):
        return "Researcher worker"

    return "unknown"


def format_submitter_kind(suggestion) -> str:
    if suggestion.submitter_user_id and not (
        suggestion.submitter_username or suggestion.submitter_display_name
    ):
        return "id"
    return "text"


def event_to_edit_state(event) -> dict[str, object]:
    return {
        "name": event.name,
        "event_date": event.event_date,
        "city": event.city,
        "country": event.country,
        "timezone": event.timezone,
        "distances": event.distances,
        "regions": event.regions,
        "official_url": event.official_url,
        "registration_url": event.registration_url,
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at,
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at,
    }


def format_edit_event_prompt(event) -> str:
    lines = [
        "<b>💬 Edit event</b>",
        f"<b>{escape(event.name)}</b>",
        format_field_line("ID", event.public_id, kind="id"),
        "Public ID cannot be edited.",
        "",
        "<b>Editable fields</b>:",
    ]
    for field in EDITABLE_FIELDS:
        value = format_field_value(field, getattr(event, event_attr_name(field)))
        lines.append(f"- {format_edit_field_line(field, value)}")
    lines.extend(
        [
            "",
            "Send a field name.",
            "<b>Example</b>: <i>event_date</i>",
        ]
    )
    return "\n".join(lines)


def format_edit_field_error() -> str:
    lines = [
        "Send one editable field name.",
        "Public ID cannot be edited.",
        "",
        "<b>Editable fields</b>:",
    ]
    lines.extend(f"- <code>{field}</code>" for field in EDITABLE_FIELDS)
    return "\n".join(lines)


def event_attr_name(field: str) -> str:
    if field == "name":
        return "name"
    if field == "event_date":
        return "event_date"

    return field


def parse_edit_field(value: str) -> str | None:
    normalized = value.casefold().replace("-", " ").replace("_", " ").strip()
    if normalized in {"public id", "public_id", "id"}:
        return None

    return EDIT_FIELD_ALIASES.get(normalized)


async def ask_edit_value(message: Message, field: str, value: object) -> None:
    lines = [
        "<b>💬 Edit event</b>",
        format_field_line("Field", field, kind="id"),
        "Send a new value.",
        format_field_line(
            "Current",
            format_field_value(field, value),
            kind=field_value_kind(field),
        ),
    ]
    if field == "event_date":
        lines.append("Use <code>YYYY-MM-DD</code>, or <code>-</code> if unknown.")
    elif field == "distances":
        lines.append(
            "Public ID cannot be changed, so the current ID distance must stay included."
        )
    elif field == "registration_status":
        lines.append(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_STATUSES))}</i>."
        )
    elif field in {"registration_open_at", "registration_close_at"}:
        lines.append(
            "Use <code>YYYY-MM-DD</code>, <code>YYYY-MM-DD HH:MM</code>, "
            "or <code>-</code>."
        )
    elif field == "registration_open_precision":
        lines.append(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_OPEN_PRECISIONS))}</i>."
        )
    elif field == "regions":
        lines.append(
            "Use comma-separated tags, for example <i>global,eu,de</i>. "
            f"{format_field_line('Known', ','.join(REGION_LABELS), kind='tag')}"
        )
    elif field == "registration_url":
        lines.append("Use <code>-</code> if there is no separate registration page.")

    buttons = ("-",) if field in {
        "event_date",
        "registration_url",
        "registration_open_at",
        "registration_close_at",
    } else ()
    reply_markup = (
        distance_dialog_keyboard()
        if field == "distances"
        else dialog_keyboard(*buttons)
    )
    await message.answer("\n".join(lines), reply_markup=reply_markup)


def format_edit_field_line(field: str, value: object) -> str:
    return (
        f"<code>{escape(field)}</code>: "
        f"{format_json_field_value(value, field=field)}"
    )


def parse_edit_value(field: str, value: str) -> object:
    if field in {"name", "city", "country"}:
        if not value:
            raise ValueError(f"Send a value for <code>{FIELD_LABELS[field]}</code>.")
        return value
    if field == "timezone":
        if "/" not in value:
            raise ValueError(
                "Please send an IANA timezone, for example <i>Europe/Berlin</i>."
            )
        return value
    if field == "event_date":
        try:
            return parse_optional_date(value)
        except ValueError as error:
            raise ValueError(
                "Use <code>YYYY-MM-DD</code>, or <code>-</code> if unknown."
            ) from error
    if field == "registration_status":
        return parse_registration_status(value)
    if field in {"registration_open_at", "registration_close_at"}:
        return parse_optional_registration_time(value)
    if field == "registration_open_precision":
        return parse_registration_open_precision(value)
    if field == "distances":
        return parse_distances(value)
    if field == "regions":
        return parse_regions(value)
    if field == "official_url":
        if not valid_url(value):
            raise ValueError("Please send a full URL starting with http:// or https://.")
        return value
    if field == "registration_url":
        registration_url = parse_optional_url(value)
        if registration_url is not None and not valid_url(registration_url):
            raise ValueError(
                "Please send a full URL starting with http:// or https://, or <code>-</code>."
            )
        return registration_url

    raise ValueError("This field cannot be edited.")


async def ask_field_confirmation(message: Message, field: str, value: object) -> None:
    lines = [
        f"<b>💬 {FIELD_LABELS[field]}</b>",
        format_draft_field_line(field, value),
        "Reply ok to keep it, or send the corrected value.",
    ]
    if field == "event_date":
        lines.append("Use <code>YYYY-MM-DD</code>, or <code>-</code> if unknown.")
    elif field == "registration_status":
        lines.append(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_STATUSES))}</i>."
        )
    elif field in {"registration_open_at", "registration_close_at"}:
        lines.append(
            "Use <code>YYYY-MM-DD</code>, <code>YYYY-MM-DD HH:MM</code>, "
            "or <code>-</code>."
        )
    elif field == "registration_open_precision":
        lines.append(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_OPEN_PRECISIONS))}</i>."
        )
    elif field == "regions":
        lines.append(
            "Use comma-separated tags, for example <i>global,eu,de</i>. "
            f"{format_field_line('Known', ','.join(REGION_LABELS), kind='tag')}"
        )
    elif field == "registration_url":
        lines.append("Use <code>-</code> if there is no separate registration page.")

    buttons = ["ok"]
    if field in {
        "event_date",
        "registration_url",
        "registration_open_at",
        "registration_close_at",
    }:
        buttons.append("-")

    reply_markup = (
        distance_dialog_keyboard(*buttons)
        if field == "distances"
        else dialog_keyboard(*buttons)
    )
    await message.answer("\n".join(lines), reply_markup=reply_markup)


def format_draft_field_line(field: str, value: object) -> str:
    draft_value = format_field_value(field, value)
    if field_value_kind(field) == "id":
        return format_field_line("Draft", draft_value, kind="id")

    return f"<b>Draft</b>: <u>{escape(draft_value)}</u>"


def format_field_value(field: str, value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, tuple | list):
        if field == "distances":
            return format_distance_input_value(value)
        return ",".join(str(item) for item in value) or "unknown"

    return str(value) or "unknown"


def format_distance_input_value(value: tuple[object, ...] | list[object]) -> str:
    codes = [
        DISTANCE_KEY_TO_CODE.get(str(item), str(item))
        for item in value
    ]
    return ",".join(codes) or "unknown"


def confirmed_value(reply: str, current: object) -> object:
    if reply.casefold() in {"ok", "yes", "y", "keep", "confirm", "+"}:
        return current

    return reply


def parse_optional_date(value: str) -> str | None:
    if value in {"", "-", "unknown", "skip"}:
        return None

    date.fromisoformat(value)
    return value


def parse_optional_registration_time(value: str) -> str | None:
    value = value.strip()
    if value.casefold() in {"", "-", "unknown", "skip"}:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        date.fromisoformat(value)
        return value

    normalized = value.replace(" ", "T", 1)
    try:
        datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            "Use <code>YYYY-MM-DD</code>, <code>YYYY-MM-DD HH:MM</code>, "
            "or <code>-</code> if unknown."
        ) from error

    return value


def parse_registration_status(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in REGISTRATION_STATUSES:
        raise ValueError(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_STATUSES))}</i>."
        )
    return normalized


def parse_registration_open_precision(value: str) -> str:
    normalized = normalize_choice(value)
    if normalized not in REGISTRATION_OPEN_PRECISIONS:
        raise ValueError(
            "Use one of: "
            f"<i>{', '.join(sorted(REGISTRATION_OPEN_PRECISIONS))}</i>."
        )
    return normalized


def normalize_choice(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def parse_distances(value: str) -> tuple[str, ...]:
    normalized = value.casefold().replace("_", " ").replace("-", " ")
    terms = [term.strip() for term in normalized.replace("/", ",").split(",") if term.strip()]
    if not terms:
        raise ValueError(
            "Send one of these distance tags.\n"
            f"{format_field_line('Supported', supported_distance_help(), kind='tag')}"
        )

    distances: list[str] = []
    for term in terms:
        distance = parse_distance_term(term)
        if distance is None:
            raise ValueError(
                format_field_line(
                    "Supported distances",
                    supported_distance_help(),
                    kind="tag",
                )
            )
        distances.append(distance)

    return tuple(dict.fromkeys(distances))


def parse_distance_term(term: str) -> str | None:
    if term in DISTANCE_CODE_TO_KEY:
        return DISTANCE_CODE_TO_KEY[term]
    if term.endswith("k") and term[:-1] in DISTANCE_CODE_TO_KEY:
        return DISTANCE_CODE_TO_KEY[term[:-1]]
    if term in DISTANCE_KEY_TO_CODE:
        return term
    if term in {"half", "half marathon"}:
        return "half_marathon"

    return None


def public_id_distance_code(public_id: str) -> str:
    return public_id.rsplit(".", maxsplit=1)[-1]


def supported_distance_codes() -> str:
    return ",".join(f".{code}" for code in DISTANCE_CODE_TO_KEY)


def supported_distance_help() -> str:
    return ", ".join(
        f"{code}={DISTANCE_LABELS.get(distance_key, distance_key)}"
        for code, distance_key in DISTANCE_CODE_TO_KEY.items()
    )


def parse_regions(value: str) -> tuple[str, ...]:
    regions = tuple(
        dict.fromkeys(
            region.strip().casefold()
            for region in value.replace(";", ",").replace(" ", ",").split(",")
            if region.strip()
        )
    )
    if not regions:
        raise ValueError("Send at least one region tag, for example <i>global,eu</i>.")
    has_invalid_region = any(
        region not in {"global", "eu"} and (len(region) != 2 or not region.isalpha())
        for region in regions
    )
    if has_invalid_region:
        raise ValueError(
            "Use short region tags like <code>global</code>, <code>eu</code>, "
            "or ISO alpha-2 country tags like <code>de</code>."
        )

    return regions


def parse_optional_url(value: str) -> str | None:
    if value in {"", "-", "unknown", "same", "skip"}:
        return None

    return value


def parse_queue_number(value: str) -> int | None:
    normalized = value.strip()
    if normalized.startswith("#"):
        normalized = normalized[1:]
    if not normalized.isdigit():
        return None

    number = int(normalized)
    return number if number >= 1 else None


def parse_update_record_id(value: str) -> int | None:
    return parse_queue_number(value)


def parse_suggestion_record_id(value: str) -> int | None:
    return parse_queue_number(value)


def parse_record_id_callback(value: str) -> int | None:
    if not value.isdigit():
        return None

    record_id = int(value)
    return record_id if record_id >= 1 else None


def parse_update_list_callback_payload(value: str) -> int:
    try:
        return parse_update_limit(value)
    except ValueError:
        return UPDATE_LIST_DEFAULT_LIMIT


def parse_update_show_callback_payload(value: str) -> tuple[int | None, int]:
    parts = value.split(":")
    if len(parts) not in {1, 2}:
        return None, UPDATE_LIST_DEFAULT_LIMIT

    list_limit = (
        parse_update_list_callback_payload(parts[1])
        if len(parts) == 2
        else UPDATE_LIST_DEFAULT_LIMIT
    )
    return parse_record_id_callback(parts[0]), list_limit


def parse_update_action_callback_payload(value: str) -> tuple[int | None, int]:
    parts = value.split(":")
    if not parts:
        return None, UPDATE_LIST_DEFAULT_LIMIT

    update_id = parse_record_id_callback(parts[0])
    list_limit = (
        parse_update_list_callback_payload(parts[1])
        if len(parts) >= 2
        else UPDATE_LIST_DEFAULT_LIMIT
    )
    return update_id, list_limit


def parse_update_partial_callback_payload(value: str) -> tuple[int | None, int, int]:
    parts = value.split(":")
    if len(parts) not in {1, 2, 3}:
        return None, UPDATE_LIST_DEFAULT_LIMIT, 0

    update_id = parse_record_id_callback(parts[0])
    list_limit = (
        parse_update_list_callback_payload(parts[1])
        if len(parts) >= 2
        else UPDATE_LIST_DEFAULT_LIMIT
    )
    try:
        mask = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        mask = 0

    return update_id, list_limit, max(mask, 0)


def parse_suggestion_action_callback_payload(value: str) -> tuple[int | None, int | None, int]:
    parts = value.split(":")
    if not parts:
        return None, None, 1

    suggestion_id = parse_record_id_callback(parts[0])
    sequence = parse_queue_number(parts[1]) if len(parts) >= 2 else None
    try:
        list_limit = parse_suggestion_limit(parts[2]) if len(parts) >= 3 else 1
    except ValueError:
        list_limit = 1
    return suggestion_id, sequence, list_limit


def parse_suggestion_show_callback_payload(value: str) -> tuple[int | None, int | None, int]:
    parts = value.split(":")
    if len(parts) == 2:
        return None, parse_record_id_callback(parts[0]), parse_suggestion_list_callback_payload(
            parts[1]
        )
    if len(parts) != 3:
        return None, None, SUGGESTION_LIST_DEFAULT_LIMIT

    try:
        list_limit = parse_suggestion_list_callback_payload(parts[2])
    except ValueError:
        list_limit = SUGGESTION_LIST_DEFAULT_LIMIT
    return parse_queue_number(parts[0]), parse_record_id_callback(parts[1]), list_limit


def parse_suggestion_card_callback_payload(value: str) -> tuple[int | None, int]:
    parts = value.split(":")
    if len(parts) not in {1, 2}:
        return None, SUGGESTION_LIST_DEFAULT_LIMIT

    list_limit = (
        parse_suggestion_list_callback_payload(parts[1])
        if len(parts) == 2
        else SUGGESTION_LIST_DEFAULT_LIMIT
    )
    return parse_record_id_callback(parts[0]), list_limit


def parse_suggestion_list_callback_payload(value: str) -> int:
    try:
        return parse_suggestion_limit(value)
    except ValueError:
        return SUGGESTION_LIST_DEFAULT_LIMIT


def parse_archive_show_callback_payload(value: str) -> tuple[str, int]:
    parts = value.split(":")
    if len(parts) not in {1, 2}:
        return "", ARCHIVE_LIST_DEFAULT_LIMIT

    try:
        list_limit = (
            parse_archive_limit(parts[1])
            if len(parts) == 2
            else ARCHIVE_LIST_DEFAULT_LIMIT
        )
    except ValueError:
        list_limit = ARCHIVE_LIST_DEFAULT_LIMIT
    return parts[0], list_limit


def parse_restore_callback_payload(value: str) -> tuple[str, int]:
    parts = value.split(":")
    if len(parts) not in {1, 2}:
        return "", ARCHIVE_LIST_DEFAULT_LIMIT
    if len(parts) == 1:
        return parts[0], ARCHIVE_LIST_DEFAULT_LIMIT

    try:
        list_limit = parse_archive_limit(parts[1])
    except ValueError:
        list_limit = ARCHIVE_LIST_DEFAULT_LIMIT
    return parts[0], list_limit


def parse_show_suggestion_number(value: str) -> int:
    normalized = value.strip()
    if not normalized:
        return 1

    number = parse_queue_number(normalized)
    if number is None:
        raise ValueError("Use a suggestion number, for example <i>3</i>.")

    return number


def parse_suggestion_limit(value: str) -> int:
    normalized = value.strip()
    if not normalized:
        return SUGGESTION_LIST_DEFAULT_LIMIT
    if not normalized.isdigit() or int(normalized) < 1:
        raise ValueError("Use a list size, for example <i>5</i>.")

    return min(int(normalized), EVENT_SUGGESTION_MAX_PENDING_TOTAL)


def parse_archive_limit(value: str) -> int:
    normalized = value.strip()
    if not normalized:
        return ARCHIVE_LIST_DEFAULT_LIMIT
    if not normalized.isdigit() or int(normalized) < 1:
        raise ValueError("Use a list size, for example <i>20</i>.")

    return min(int(normalized), ARCHIVE_LIST_MAX_LIMIT)


def parse_update_limit(value: str) -> int:
    normalized = value.strip()
    if not normalized:
        return UPDATE_LIST_DEFAULT_LIMIT
    if not normalized.isdigit() or int(normalized) < 1:
        raise ValueError("Use a list size, for example <i>20</i>.")

    return min(int(normalized), UPDATE_LIST_MAX_LIMIT)


def optional_string(value: object) -> str | None:
    if value is None:
        return None

    return str(value)


def valid_url(value: str) -> bool:
    return value.startswith(("https://", "http://"))
