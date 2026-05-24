import asyncio

from run4221.bot import fallback


class FakeMessage:
    def __init__(self) -> None:
        self.answers: list[str] = []

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append(text)


def test_unknown_command_gets_helpful_response() -> None:
    message = FakeMessage()

    asyncio.run(fallback.handle_unknown_command(message))

    assert message.answers == ["Unknown command."]
