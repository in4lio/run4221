from __future__ import annotations

from html import escape

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message

from run4221.bot.auth import is_moderator_account
from run4221.bot.formatting import format_field_line
from run4221.bot.keyboards import (
    dialog_keyboard,
    distance_dialog_keyboard,
    is_cancel_text,
    remove_dialog_keyboard,
)
from run4221.bot.prompts import waiting_prompt
from run4221.db.repository import (
    EventSuggestionCreate,
    EventWriteError,
    add_event_suggestion,
)
from run4221.events import DISTANCE_CODE_TO_KEY, DISTANCE_KEY_TO_CODE, DISTANCE_LABELS

router = Router(name="suggestions")


class SuggestEventStates(StatesGroup):
    name = State()
    url = State()
    distances = State()
    note = State()


@router.message(Command("suggest"))
async def handle_suggest(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(SuggestEventStates.name)
    await message.answer(
        waiting_prompt(
            "Suggest event",
            "event name",
            example="Berlin Marathon",
            extra=("Publication is not guaranteed.",),
        ),
        reply_markup=dialog_keyboard(),
    )


@router.message(SuggestEventStates.name, lambda message: is_cancel_text(message.text))
@router.message(SuggestEventStates.url, lambda message: is_cancel_text(message.text))
@router.message(SuggestEventStates.distances, lambda message: is_cancel_text(message.text))
@router.message(SuggestEventStates.note, lambda message: is_cancel_text(message.text))
async def handle_suggest_cancel_button(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Cancelled.", reply_markup=remove_dialog_keyboard())


@router.message(SuggestEventStates.name, ~F.text.startswith("/"))
async def handle_suggest_name(message: Message, state: FSMContext) -> None:
    name = text_value(message)
    if not name:
        await message.answer("Send the event name, for example <i>Berlin Marathon</i>.")
        return

    await state.update_data(event_name=name)
    await state.set_state(SuggestEventStates.url)
    await message.answer(
        waiting_prompt("Suggest event", "official event or registration URL"),
        reply_markup=dialog_keyboard(),
    )


@router.message(SuggestEventStates.url, ~F.text.startswith("/"))
async def handle_suggest_url(message: Message, state: FSMContext) -> None:
    value = text_value(message)
    if not valid_url(value):
        await message.answer("Send a full URL starting with http:// or https://.")
        return

    await state.update_data(url=value)
    await state.set_state(SuggestEventStates.distances)
    await message.answer(
        waiting_prompt(
            "Suggest event",
            "distance codes",
            example="42",
        ),
        reply_markup=distance_dialog_keyboard(),
    )


@router.message(SuggestEventStates.distances, ~F.text.startswith("/"))
async def handle_suggest_distances(message: Message, state: FSMContext) -> None:
    try:
        distances = parse_distances(text_value(message))
    except ValueError as error:
        await message.answer(str(error))
        return

    await state.update_data(distances=distances)
    await state.set_state(SuggestEventStates.note)
    await message.answer(
        waiting_prompt("Suggest event", "optional note", example="-"),
        reply_markup=dialog_keyboard("-"),
    )


@router.message(SuggestEventStates.note, ~F.text.startswith("/"))
async def handle_suggest_note(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    submitter = submitter_from_message(message)
    try:
        suggestion = add_event_suggestion(
            EventSuggestionCreate(
                event_name=str(data["event_name"]),
                url=optional_string(data.get("url")),
                event_date=None,
                location=None,
                region_tags=(),
                distances=tuple(data.get("distances") or ()),
                note=parse_optional_text(text_value(message)),
                submitter_user_id=submitter["user_id"],
                submitter_username=submitter["username"],
                submitter_display_name=submitter["display_name"],
                submitter_is_moderator=bool(submitter["is_moderator"]),
            )
        )
    except EventWriteError as error:
        await message.answer(f"Could not save suggestion: {escape(str(error))}")
        return

    await state.clear()
    await message.answer(
        "Suggestion submitted.\n"
        f"{format_field_line('Request ID', f'#{suggestion.id}', kind='id')}\n"
        "A moderator will review it. Publication is not guaranteed.",
        reply_markup=remove_dialog_keyboard(),
    )


def submitter_from_message(message: Message) -> dict[str, str | bool | None]:
    user = getattr(message, "from_user", None)
    if user is None:
        return {
            "user_id": None,
            "username": None,
            "display_name": None,
            "is_moderator": False,
        }

    display_name = getattr(user, "full_name", None) or " ".join(
        part
        for part in (
            getattr(user, "first_name", None),
            getattr(user, "last_name", None),
        )
        if part
    )
    user_id = getattr(user, "id", None)
    username = getattr(user, "username", None)
    return {
        "user_id": str(user_id or "") or None,
        "username": username,
        "display_name": display_name or None,
        "is_moderator": is_moderator_account(user_id, username),
    }


def parse_optional_text(value: str) -> str | None:
    value = value.strip()
    if value in {"", "-", "unknown", "skip"}:
        return None

    return value


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


def supported_distance_help() -> str:
    return ", ".join(
        f"{code}={DISTANCE_LABELS.get(distance_key, distance_key)}"
        for code, distance_key in DISTANCE_CODE_TO_KEY.items()
    )


def optional_string(value: object) -> str | None:
    if value is None:
        return None

    return str(value)


def valid_url(value: str) -> bool:
    return value.startswith(("https://", "http://"))


def text_value(message: Message) -> str:
    return (message.text or "").strip()
