import asyncio
import logging
from contextlib import suppress

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from run4221.bot.commands import configure_bot_commands
from run4221.bot.fallback import router as fallback_router
from run4221.bot.moderator import router as moderator_router
from run4221.bot.public import router as public_router
from run4221.bot.suggestions import router as suggestions_router
from run4221.config import get_settings
from run4221.db.bootstrap import initialize_database
from run4221.db.session import require_initialized_database
from run4221.posting.scheduler import run_channel_publisher_loop


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if settings.app_env == "production":
        require_initialized_database(settings.database_url)
    initialize_database(
        settings.database_url,
        seed_initial_events=settings.seed_initial_events,
    )

    bot = Bot(
        token=settings.telegram_bot_token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = Dispatcher()
    dispatcher.include_router(public_router)
    dispatcher.include_router(suggestions_router)
    dispatcher.include_router(moderator_router)
    dispatcher.include_router(fallback_router)

    await configure_bot_commands(bot, settings.moderator_ids)
    logging.getLogger(__name__).info("Starting %s in polling mode", settings.telegram_bot_username)
    publisher_task = None
    if settings.telegram_channel_posting_enabled:
        publisher_task = asyncio.create_task(
            run_channel_publisher_loop(
                bot,
                database_url=settings.database_url,
                interval_seconds=settings.telegram_channel_poll_seconds,
            ),
            name="telegram-channel-publisher",
        )
    else:
        logging.getLogger(__name__).info("Telegram channel publishing is disabled")
    try:
        await dispatcher.start_polling(
            bot,
            allowed_updates=dispatcher.resolve_used_update_types(),
        )
    finally:
        if publisher_task is not None:
            publisher_task.cancel()
            with suppress(asyncio.CancelledError):
                await publisher_task


def run() -> None:
    asyncio.run(main())
