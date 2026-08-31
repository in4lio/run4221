from __future__ import annotations

from aiogram.types import Message

from run4221.config import ModeratorAccounts, get_settings


def is_moderator_account(
    user_id: int | None,
    username: str | None,
    moderator_accounts: ModeratorAccounts | None = None,
) -> bool:
    del username
    allowed_ids, _ = moderator_accounts or get_settings().moderator_accounts
    return user_id is not None and user_id in allowed_ids


def is_moderator_id(
    user_id: int | None,
    moderator_ids: tuple[int, ...] | None = None,
) -> bool:
    return is_moderator_account(user_id, None, (moderator_ids or (), ()))


def is_moderator_username(
    username: str | None,
    moderator_usernames: tuple[str, ...] | None = None,
) -> bool:
    del username, moderator_usernames
    return False


async def require_moderator(message: Message) -> bool:
    user = message.from_user
    user_id = user.id if user is not None else None
    username = user.username if user is not None else None
    if is_moderator_account(user_id, username):
        return True

    await message.answer("This command is available to moderators only.")
    return False
