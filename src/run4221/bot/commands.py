from __future__ import annotations

from aiogram import Bot
from aiogram.types import BotCommand, BotCommandScopeChat

PUBLIC_COMMANDS = (
    BotCommand(command="start", description="Start the bot"),
    BotCommand(command="help", description="Show available commands"),
    BotCommand(command="list_events", description="List tracked events"),
    BotCommand(command="list_open", description="List open registrations"),
    BotCommand(command="search_events", description="Search tracked events"),
    BotCommand(command="show_event", description="Show event details"),
    BotCommand(command="suggest", description="Suggest an event for tracking"),
)

MODERATOR_COMMANDS = (
    BotCommand(command="todo", description="Show moderator todo"),
    BotCommand(command="channel_drafts", description="Review channel news drafts"),
    BotCommand(command="channel_correction", description="Prepare public correction"),
    BotCommand(command="add_event", description="Add event from URL"),
    BotCommand(command="archive_event", description="Preview and archive event"),
    BotCommand(command="delete_event", description="Permanently delete event"),
    BotCommand(command="edit_event", description="Edit tracked event"),
    BotCommand(command="list_archive", description="List archived events"),
    BotCommand(command="restore_event", description="Restore archived event"),
    BotCommand(command="update_event", description="Run registration scan"),
    BotCommand(command="apply_update", description="Apply pending update"),
    BotCommand(command="list_updates", description="List pending updates"),
    BotCommand(command="next_update", description="Show oldest pending update"),
    BotCommand(command="reject_update", description="Reject pending update"),
    BotCommand(command="show_update", description="Show pending update"),
    BotCommand(command="apply_suggestion", description="Apply pending suggestion"),
    BotCommand(command="list_suggestions", description="List pending suggestions"),
    BotCommand(command="next_suggestion", description="Show oldest pending suggestion"),
    BotCommand(command="reject_suggestion", description="Reject pending suggestion"),
    BotCommand(command="show_suggestion", description="Show pending suggestion"),
    BotCommand(command="cancel", description="Cancel guided flow"),
)


def public_bot_commands() -> list[BotCommand]:
    return list(PUBLIC_COMMANDS)


def moderator_bot_commands() -> list[BotCommand]:
    return [*PUBLIC_COMMANDS, *MODERATOR_COMMANDS]


async def configure_bot_commands(bot: Bot, moderator_ids: tuple[int, ...] = ()) -> None:
    await bot.set_my_commands(public_bot_commands())
    for moderator_id in moderator_ids:
        await set_moderator_commands_for_chat(bot, moderator_id)


async def set_moderator_commands_for_chat(bot: Bot, chat_id: int | str) -> None:
    await bot.set_my_commands(
        moderator_bot_commands(),
        scope=BotCommandScopeChat(chat_id=chat_id),
    )
