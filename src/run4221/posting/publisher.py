from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from aiogram.exceptions import TelegramNetworkError, TelegramServerError

from run4221.posting.ledger import (
    ChannelMessageRecord,
    claim_channel_message,
    list_ready_channel_message_ids,
    mark_channel_message_failed,
    mark_channel_message_published,
)


async def publish_channel_message(
    bot: Any,
    message_id: int,
    *,
    database_url: str | None = None,
    now: datetime | None = None,
) -> ChannelMessageRecord | None:
    claimed = await asyncio.to_thread(
        claim_channel_message,
        message_id,
        database_url=database_url,
        now=now,
    )
    if claimed is None or claimed.status != "publishing":
        return claimed
    try:
        sent = await bot.send_message(
            chat_id=claimed.target_chat_id,
            text=claimed.text,
            link_preview_options={"is_disabled": True},
        )
    except (TimeoutError, TelegramNetworkError, TelegramServerError) as exc:
        # Telegram may have accepted the send even though the response never
        # confirmed it, so these outcomes require a human reconciliation.
        return await asyncio.to_thread(
            mark_channel_message_failed,
            message_id,
            reason=str(exc),
            ambiguous=True,
            database_url=database_url,
        )
    except Exception as exc:
        return await asyncio.to_thread(
            mark_channel_message_failed,
            message_id,
            reason=str(exc),
            ambiguous=False,
            database_url=database_url,
        )
    return await asyncio.to_thread(
        mark_channel_message_published,
        message_id,
        telegram_message_id=sent.message_id,
        database_url=database_url,
    )


async def publish_ready_channel_messages(
    bot: Any,
    *,
    database_url: str | None = None,
    limit: int = 10,
    now: datetime | None = None,
) -> tuple[ChannelMessageRecord, ...]:
    results: list[ChannelMessageRecord] = []
    message_ids = await asyncio.to_thread(
        list_ready_channel_message_ids,
        database_url=database_url,
        limit=limit,
        now=now,
    )
    for message_id in message_ids:
        result = await publish_channel_message(
            bot,
            message_id,
            database_url=database_url,
            now=now,
        )
        if result is not None:
            results.append(result)
    return tuple(results)
