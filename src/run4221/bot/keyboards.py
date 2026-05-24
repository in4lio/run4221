from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from run4221.events import TrackedEvent

CANCEL_CALLBACK = "flow:cancel"
CANCEL_BUTTON = "Cancel"
CANCEL_COMMAND = "/cancel"
DISTANCE_BUTTONS = ("42", "21", "42,21")
EVENT_DETAIL_PREFIX = "event:detail:"
HELP_CALLBACK = "public:help"


def cancel_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Cancel",
                    callback_data=CANCEL_CALLBACK,
                )
            ]
        ]
    )


def dialog_keyboard(*buttons: str) -> ReplyKeyboardMarkup:
    command_buttons = tuple(dict.fromkeys((*buttons, CANCEL_BUTTON)))
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=button) for button in command_buttons]],
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Type a value or use a button",
    )


def distance_dialog_keyboard(*leading_buttons: str) -> ReplyKeyboardMarkup:
    command_buttons = tuple(dict.fromkeys(leading_buttons))
    keyboard = []
    if command_buttons:
        keyboard.append([KeyboardButton(text=button) for button in command_buttons])
    keyboard.extend(
        [
            [KeyboardButton(text=button) for button in DISTANCE_BUTTONS],
            [KeyboardButton(text=CANCEL_BUTTON)],
        ]
    )
    return ReplyKeyboardMarkup(
        keyboard=keyboard,
        resize_keyboard=True,
        one_time_keyboard=True,
        input_field_placeholder="Choose distance codes",
    )


def is_cancel_text(value: str | None) -> bool:
    return (value or "").strip().casefold() in {
        CANCEL_BUTTON.casefold(),
        CANCEL_COMMAND,
    }


def remove_dialog_keyboard() -> ReplyKeyboardRemove:
    return ReplyKeyboardRemove(remove_keyboard=True)


def help_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Help",
                    callback_data=HELP_CALLBACK,
                )
            ]
        ]
    )


def event_detail_callback(public_id: str) -> str:
    return f"{EVENT_DETAIL_PREFIX}{public_id}"


def event_detail_keyboard(event: TrackedEvent) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Show",
                    callback_data=event_detail_callback(event.public_id),
                )
            ]
        ]
    )
