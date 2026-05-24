from aiogram import F, Router
from aiogram.types import Message

from run4221.bot.keyboards import help_keyboard

router = Router(name="fallback")


@router.message(F.text.startswith("/"))
async def handle_unknown_command(message: Message) -> None:
    await message.answer("Unknown command.", reply_markup=help_keyboard())
