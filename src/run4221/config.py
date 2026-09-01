from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

type ModeratorAccounts = tuple[tuple[int, ...], tuple[str, ...]]


class TelegramChannelSettings(BaseSettings):
    """Channel settings that can be loaded without requiring the bot token."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    telegram_channel_id: str = "@run4221"


class Settings(TelegramChannelSettings):
    """Application settings loaded from environment variables or local .env."""

    app_env: Literal["local", "production", "test"] = "local"
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/run4221.sqlite3"
    seed_initial_events: bool = True

    telegram_bot_token: SecretStr
    telegram_bot_username: str = "@run4221bot"
    telegram_channel_posting_enabled: bool = False
    telegram_channel_poll_seconds: int = Field(default=60, ge=5, le=3600)
    telegram_moderator_accounts: str = ""
    telegram_moderator_ids: str = ""
    telegram_moderator_usernames: str = ""
    ai_extractor_provider: Literal["heuristic", "openai"] = "heuristic"
    run4221_prompt_source: Literal["file", "db"] = "file"
    run4221_prompt_dir: str = "private/prompts"
    openai_api_key: SecretStr | None = None
    openai_extract_model: str = "gpt-5.4-mini"

    def configured_moderator_accounts(self) -> ModeratorAccounts:
        combined = ",".join(
            value
            for value in (
                self.telegram_moderator_accounts,
                self.telegram_moderator_ids,
                self.telegram_moderator_usernames,
            )
            if value.strip()
        )
        ids, usernames = parse_moderator_accounts(combined)
        if usernames:
            raise ValueError(
                "Telegram moderator authorization requires immutable numeric user IDs; "
                "usernames are not accepted. Configure TELEGRAM_MODERATOR_IDS."
            )
        return ids, ()

    @model_validator(mode="after")
    def validate_moderator_accounts(self) -> Self:
        self.configured_moderator_accounts()
        return self

    @property
    def moderator_accounts(self) -> ModeratorAccounts:
        return self.configured_moderator_accounts()

    @property
    def moderator_ids(self) -> tuple[int, ...]:
        ids, _ = self.moderator_accounts
        return ids

    @property
    def moderator_usernames(self) -> tuple[str, ...]:
        _, usernames = self.moderator_accounts
        return usernames


def parse_moderator_accounts(value: str) -> ModeratorAccounts:
    ids: list[int] = []
    usernames: list[str] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part.isdigit():
            ids.append(int(part))
            continue

        username = normalize_username(part)
        if username:
            usernames.append(username)

    return tuple(dict.fromkeys(ids)), tuple(dict.fromkeys(usernames))


def normalize_username(value: str | None) -> str:
    return (value or "").strip().removeprefix("@").casefold()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def get_telegram_channel_settings() -> TelegramChannelSettings:
    return TelegramChannelSettings()
