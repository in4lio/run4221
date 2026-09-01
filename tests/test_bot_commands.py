import asyncio

from aiogram.types import BotCommandScopeChat

from run4221.bot.commands import configure_bot_commands


class FakeBot:
    def __init__(self) -> None:
        self.calls = []

    async def set_my_commands(self, commands, **kwargs) -> None:
        self.calls.append((commands, kwargs))


def run(coro) -> None:
    asyncio.run(coro)


def test_configure_bot_commands_sets_default_and_moderator_scopes() -> None:
    bot = FakeBot()

    run(configure_bot_commands(bot, moderator_ids=(42,)))

    assert len(bot.calls) == 2
    public_commands, public_kwargs = bot.calls[0]
    assert [command.command for command in public_commands] == [
        "start",
        "help",
        "list_events",
        "list_open",
        "search_events",
        "show_event",
        "suggest",
    ]
    assert public_kwargs == {}

    moderator_commands, moderator_kwargs = bot.calls[1]
    moderator_command_names = [command.command for command in moderator_commands]
    assert "show_event" in moderator_command_names
    assert moderator_command_names[-21:] == [
        "todo",
        "channel_drafts",
        "channel_correction",
        "add_event",
        "archive_event",
        "delete_event",
        "edit_event",
        "list_archive",
        "restore_event",
        "update_event",
        "apply_update",
        "list_updates",
        "next_update",
        "reject_update",
        "show_update",
        "apply_suggestion",
        "list_suggestions",
        "next_suggestion",
        "reject_suggestion",
        "show_suggestion",
        "cancel",
    ]
    assert isinstance(moderator_kwargs["scope"], BotCommandScopeChat)
    assert moderator_kwargs["scope"].chat_id == 42
