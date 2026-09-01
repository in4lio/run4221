from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from run4221.posting.ledger import recover_stale_publishing_messages, sync_channel_schedules
from run4221.posting.publisher import publish_ready_channel_messages

logger = logging.getLogger(__name__)


async def run_channel_publisher_cycle(
    bot: Any,
    *,
    database_url: str | None = None,
    now: datetime | None = None,
) -> None:
    # Bookkeeping failures must never block delivery of already-approved messages.
    try:
        recovered = await asyncio.to_thread(
            recover_stale_publishing_messages,
            database_url=database_url,
            now=now,
        )
        if recovered:
            logger.warning("Marked %d stale channel deliveries as ambiguous", recovered)
    except Exception:
        logger.exception("Stale channel delivery recovery failed")
    try:
        await asyncio.to_thread(sync_channel_schedules, database_url=database_url, now=now)
    except Exception:
        logger.exception("Channel schedule reconciliation failed")
    results = await publish_ready_channel_messages(
        bot,
        database_url=database_url,
        now=now,
    )
    if results:
        logger.info("Processed %d channel messages", len(results))


async def run_channel_publisher_loop(
    bot: Any,
    *,
    database_url: str | None = None,
    interval_seconds: int = 60,
) -> None:
    while True:
        try:
            await run_channel_publisher_cycle(bot, database_url=database_url)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Channel publisher cycle failed")
        await asyncio.sleep(interval_seconds)
