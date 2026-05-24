import logging
from html import escape

from aiogram import F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from run4221.bot.auth import is_moderator_account
from run4221.bot.commands import set_moderator_commands_for_chat
from run4221.bot.formatting import (
    format_event_card,
    format_event_detail,
    format_major_title,
)
from run4221.bot.keyboards import (
    CANCEL_CALLBACK,
    EVENT_DETAIL_PREFIX,
    HELP_CALLBACK,
    dialog_keyboard,
    event_detail_keyboard,
    is_cancel_text,
    remove_dialog_keyboard,
)
from run4221.bot.prompts import waiting_prompt
from run4221.events import (
    TrackedEvent,
    find_event,
    format_tag_display,
    list_events,
    list_events_by_tag,
    list_open_events,
    normalize_event_id,
    resolve_event_lookup,
    search_events,
)

router = Router(name="public")
logger = logging.getLogger(__name__)

IMPLEMENTATION_NOTICE = (
    "<b>🚧 Service is currently in implementation stage.</b>"
)
START_WELCOME = "Welcome to Full’n’Half running events tracker!"
PUBLIC_LIST_EVENTS_CALLBACK = "public:list_events"
PUBLIC_SEARCH_BERLIN_CALLBACK = "public:search:berlin"


class SearchEventsStates(StatesGroup):
    query = State()


class ShowEventStates(StatesGroup):
    event_id = State()


async def send_event_cards(
    message: Message,
    events: tuple[TrackedEvent, ...],
    *,
    title: str,
    title_reply_markup=None,
) -> None:
    if not events:
        await message.answer("No tracked events found.", reply_markup=title_reply_markup)
        return

    await message.answer(format_major_title(title), reply_markup=title_reply_markup)
    for event in events:
        await message.answer(format_event_card(event), reply_markup=event_detail_keyboard(event))


def public_discovery_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="List events",
                    callback_data=PUBLIC_LIST_EVENTS_CALLBACK,
                ),
                InlineKeyboardButton(
                    text="Search Berlin",
                    callback_data=PUBLIC_SEARCH_BERLIN_CALLBACK,
                ),
            ]
        ]
    )


@router.message(CommandStart())
async def handle_start(message: Message) -> None:
    include_moderator = message_is_moderator(message)
    await ensure_moderator_command_menu(message, include_moderator=include_moderator)
    await message.answer(format_start_text(include_moderator=include_moderator))


@router.message(Command("help"))
async def handle_help(message: Message) -> None:
    include_moderator = message_is_moderator(message)
    await ensure_moderator_command_menu(message, include_moderator=include_moderator)
    await message.answer(format_help_text(include_moderator=include_moderator))


def message_is_moderator(message: Message) -> bool:
    user = getattr(message, "from_user", None)
    if user is None:
        return False

    return is_moderator_account(user.id, user.username)


async def ensure_moderator_command_menu(
    message: Message,
    *,
    include_moderator: bool,
) -> None:
    if not include_moderator:
        return

    try:
        await set_moderator_commands_for_chat(message.bot, message.chat.id)
    except TelegramAPIError as error:
        logger.warning("Could not set moderator command menu: %s", error)


def format_start_text(*, include_moderator: bool) -> str:
    return "\n\n".join(
        [
            START_WELCOME,
            format_help_text(include_moderator=include_moderator),
        ]
    )


def format_help_text(*, include_moderator: bool) -> str:
    lines = [
        "I can help you track marathon and half marathon registration openings. "
        "Content is managed by AI agents, with public updates posted in "
        "<a href=\"https://t.me/run4221\">@run4221</a> and "
        "<a href=\"https://run4221.com\">run4221.com</a>.",
        "",
        IMPLEMENTATION_NOTICE,
        "",
        "You can control me by sending these commands:",
        "",
        "/help - show this help",
        "",
        "/list_events - list tracked events",
        "/list_events &lt;tag&gt; - list events by tag, e.g. <i>major</i>",
        "/list_open - list open registrations",
        "/list_open &lt;tag&gt; - list open registrations by tag, e.g. <i>us</i>",
        "/search_events &lt;query&gt; - search events, e.g. <i>berlin</i>",
        "/show_event &lt;event_id&gt; - show event details, e.g. <i>berlin.42</i>",
        "",
        "/suggest - suggest a new event for tracking",
    ]

    if include_moderator:
        lines.extend(
            [
                "",
                "<b>Management</b>",
                "",
                "/todo - show pending work",
                "",
                "<b>Events</b>",
                "/add_event &lt;url&gt; - add event from URL, "
                "e.g. <i>https://example[.]com/race</i>",
                "/archive_event &lt;event_id&gt; - preview and confirm archive, "
                "e.g. <i>berlin.42</i>",
                "/delete_event &lt;event_id&gt; - permanently delete event, e.g. <i>berlin.42</i>",
                "/edit_event &lt;event_id&gt; - edit event fields, e.g. <i>berlin.42</i>",
                "/list_archive - list archived events",
                "/restore_event &lt;event_id&gt; - restore archived event, e.g. <i>berlin.42</i>",
                "/update_event &lt;event_id&gt; - run registration scan, e.g. <i>berlin.42</i>",
                "",
                "<b>Updates</b>",
                "/apply_update &lt;update_id&gt; - apply a pending update, e.g. <i>#1</i>",
                "/list_updates - list pending updates",
                "/next_update - show oldest pending update",
                "/reject_update &lt;update_id&gt; - reject a pending update, e.g. <i>#1</i>",
                "/show_update &lt;update_id&gt; - show pending update details, e.g. <i>#1</i>",
                "",
                "<b>Suggestions</b>",
                "/apply_suggestion &lt;suggestion_id&gt; - apply pending suggestion, "
                "e.g. <i>#1</i>",
                "/list_suggestions - list pending suggestions",
                "/next_suggestion - show oldest pending suggestion",
                "/reject_suggestion &lt;suggestion_id&gt; - reject pending suggestion, "
                "e.g. <i>#1</i>",
                "/show_suggestion &lt;suggestion_id&gt; - show pending suggestion, e.g. <i>#1</i>",
            ]
        )
    return "\n".join(lines)


def requested_tag_display(tag: str) -> str:
    return format_tag_display(tag)


@router.message(Command("list_events"))
async def handle_events(message: Message, command: CommandObject) -> None:
    tag = command.args.strip() if command.args else ""
    if tag:
        events = list_events_by_tag(tag, limit=10)
        tag_display = requested_tag_display(tag)
        if not events:
            await message.answer(
                f"No tracked events found for tag <code>{escape(tag_display)}</code>. "
                "Try listing events or searching.",
                reply_markup=public_discovery_keyboard(),
            )
            return

        await send_event_cards(message, events, title=f"Events tagged: {tag_display}")
        return

    events = list_events(limit=10)
    await send_event_cards(message, events, title="Tracked events")


@router.message(Command("list_open"))
async def handle_open(message: Message, command: CommandObject) -> None:
    tag = command.args.strip() if command.args else ""
    events = list_open_events(limit=10, tag=tag or None)
    if not events:
        if tag:
            tag_display = requested_tag_display(tag)
            await message.answer(
                f"No tracked registrations are currently open for tag "
                f"<code>{escape(tag_display)}</code>. Try listing tracked events.",
                reply_markup=public_discovery_keyboard(),
            )
            return

        await message.answer(
            "No tracked registrations are currently open.",
        )
        return

    title = (
        f"Open registrations tagged: {requested_tag_display(tag)}"
        if tag
        else "Open registrations"
    )
    await send_event_cards(message, events, title=title)


@router.message(Command("search_events"))
async def handle_search(message: Message, command: CommandObject, state: FSMContext) -> None:
    query = command.args or ""
    if not query.strip():
        await state.set_state(SearchEventsStates.query)
        await message.answer(
            waiting_prompt("Search events", "search term", example="berlin"),
            reply_markup=dialog_keyboard(),
        )
        return

    await send_search_results(message, query)


@router.message(SearchEventsStates.query, lambda message: is_cancel_text(message.text))
@router.message(ShowEventStates.event_id, lambda message: is_cancel_text(message.text))
async def handle_dialog_cancel_button(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Cancelled.", reply_markup=remove_dialog_keyboard())


@router.message(SearchEventsStates.query, ~F.text.startswith("/"))
async def handle_search_query(message: Message, state: FSMContext) -> None:
    query = text_value(message)
    if not query:
        await message.answer("Send a search term, for example <i>berlin</i>.")
        return

    await state.clear()
    await send_search_results(message, query, cleanup_keyboard=True)


async def send_search_results(
    message: Message,
    query: str,
    *,
    cleanup_keyboard: bool = False,
) -> None:
    query = query.strip()
    matches = search_events(query)[:10]
    await send_event_cards(
        message,
        matches,
        title=f"Search results for: {query}",
        title_reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
    )


@router.message(Command("show_event"))
async def handle_event(message: Message, command: CommandObject, state: FSMContext) -> None:
    event_id = command.args or ""
    if not event_id.strip():
        await state.set_state(ShowEventStates.event_id)
        await message.answer(
            waiting_prompt("Show event", "event ID", example="berlin.42"),
            reply_markup=dialog_keyboard(),
        )
        return

    await send_event_details(message, event_id)


@router.message(ShowEventStates.event_id, ~F.text.startswith("/"))
async def handle_event_id(message: Message, state: FSMContext) -> None:
    event_id = text_value(message)
    if not event_id:
        await message.answer("Send an event ID, for example <i>berlin.42</i>.")
        return

    await state.clear()
    await send_event_details(message, event_id, cleanup_keyboard=True)


async def send_event_details(
    message: Message,
    event_id: str,
    *,
    cleanup_keyboard: bool = False,
) -> None:
    normalized_id = normalize_event_id(event_id)
    lookup = resolve_event_lookup(event_id, limit=5)
    if lookup.exact is None:
        matches = lookup.suggestions
        if len(matches) == 1:
            await message.answer(
                format_event_detail(matches[0]),
                reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
            )
            return
        if matches:
            await send_event_cards(
                message,
                matches,
                title=f"Possible matches for: {event_id}",
                title_reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
            )
            return

        await message.answer(
            f"I could not find event ID <code>{escape(normalized_id)}</code>. "
            "Try listing events or searching.",
            reply_markup=remove_dialog_keyboard()
            if cleanup_keyboard
            else public_discovery_keyboard(),
        )
        return

    await message.answer(
        format_event_detail(lookup.exact),
        reply_markup=remove_dialog_keyboard() if cleanup_keyboard else None,
    )


def text_value(message: Message) -> str:
    return (message.text or "").strip()


@router.callback_query(F.data == CANCEL_CALLBACK)
async def handle_cancel_callback(callback: CallbackQuery, state: FSMContext) -> None:
    current_state = await state.get_state()
    await state.clear()
    await callback.answer("Cancelled.")
    if callback.message is not None:
        text = "Cancelled." if current_state is not None else "Nothing to cancel."
        await callback.message.answer(text, reply_markup=remove_dialog_keyboard())


@router.callback_query(F.data == HELP_CALLBACK)
async def handle_help_callback(callback: CallbackQuery) -> None:
    include_moderator = is_moderator_account(
        callback.from_user.id,
        callback.from_user.username,
    )
    if callback.message is not None:
        await ensure_moderator_command_menu(
            callback.message,
            include_moderator=include_moderator,
        )

    await callback.answer()
    if callback.message is not None:
        await callback.message.answer(format_help_text(include_moderator=include_moderator))


@router.callback_query(F.data == PUBLIC_LIST_EVENTS_CALLBACK)
async def handle_public_list_events_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message is not None:
        await send_event_cards(callback.message, list_events(limit=10), title="Tracked events")


@router.callback_query(F.data == PUBLIC_SEARCH_BERLIN_CALLBACK)
async def handle_public_search_berlin_callback(callback: CallbackQuery) -> None:
    await callback.answer()
    if callback.message is not None:
        await send_search_results(callback.message, "berlin")


@router.callback_query(F.data.startswith(EVENT_DETAIL_PREFIX))
async def handle_event_detail_callback(callback: CallbackQuery) -> None:
    public_id = (callback.data or "").removeprefix(EVENT_DETAIL_PREFIX)
    event = find_event(public_id)
    if event is None:
        await callback.answer("Event not found.", show_alert=True)
        return

    await callback.answer()
    if callback.message is not None:
        await callback.message.answer(format_event_detail(event))
