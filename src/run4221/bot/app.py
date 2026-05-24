import asyncio
import logging

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


async def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    await dispatcher.start_polling(bot, allowed_updates=dispatcher.resolve_used_update_types())


def run() -> None:
    asyncio.run(main())
