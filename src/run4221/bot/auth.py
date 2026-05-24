from __future__ import annotations

from aiogram.types import Message

from run4221.config import ModeratorAccounts, get_settings, normalize_username


def is_moderator_account(
    user_id: int | None,
    username: str | None,
    moderator_accounts: ModeratorAccounts | None = None,
) -> bool:
    allowed_ids, allowed_usernames = moderator_accounts or get_settings().moderator_accounts
    if user_id is not None and user_id in allowed_ids:
        return True

    normalized_username = normalize_username(username)
    return bool(normalized_username and normalized_username in allowed_usernames)


def is_moderator_id(
    user_id: int | None,
    moderator_ids: tuple[int, ...] | None = None,
) -> bool:
    return is_moderator_account(user_id, None, (moderator_ids or (), ()))


def is_moderator_username(
    username: str | None,
    moderator_usernames: tuple[str, ...] | None = None,
) -> bool:
    return is_moderator_account(None, username, ((), moderator_usernames or ()))


async def require_moderator(message: Message) -> bool:
    user = message.from_user
    user_id = user.id if user is not None else None
    username = user.username if user is not None else None
    if is_moderator_account(user_id, username):
        return True

    await message.answer("This command is available to moderators only.")
    return False
