from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from run4221.db.prompts import (
    RESEARCH_AGENT_PROMPT,
    PromptVersionRecord,
    get_file_prompt,
    get_runtime_prompt,
)
from run4221.researcher.schemas import (
    RESEARCHER_MAX_PENDING_SUGGESTIONS,
    ResearchBudget,
)


class ResearcherSettings(BaseSettings):
    """Settings for the standalone researcher process.

    The worker deliberately has its own settings model so starting it never requires
    Telegram credentials. Shared resources keep their established environment names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RESEARCHER_",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: Literal["local", "production", "test"] = Field(
        default="local",
        validation_alias=AliasChoices("RESEARCHER_APP_ENV", "APP_ENV"),
    )
    database_url: str = Field(
        default="sqlite:///data/run4221.sqlite3",
        min_length=1,
        validation_alias=AliasChoices("RESEARCHER_DATABASE_URL", "DATABASE_URL"),
    )
    openai_api_key: SecretStr = Field(
        min_length=1,
        validation_alias=AliasChoices("RESEARCHER_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    model: str = Field(default="gpt-5.6-luna", min_length=1)
    enabled: bool = False
    discovery_enabled: bool = False
    rendering_enabled: bool = False
    interval_seconds: int = Field(default=3_600, ge=60, le=604_800)
    artifact_dir: str = Field(default="data/research_runs", min_length=1)
    schedule_path: str = Field(default="data/researcher_schedule.json", min_length=1)
    lock_path: str = Field(default="data/researcher.lock", min_length=1)
    health_path: str = Field(default="data/researcher_health.json", min_length=1)
    health_stale_after_seconds: int = Field(default=180, ge=30, le=3_600)
    trusted_domains: str = ""
    trusted_registry_urls: str = ""
    discovery_queries: str = ""
    prompt_source: Literal["file", "db"] = Field(
        default="file",
        validation_alias=AliasChoices("RESEARCHER_PROMPT_SOURCE", "RUN4221_PROMPT_SOURCE"),
    )
    prompt_dir: str = Field(
        default="private/prompts",
        min_length=1,
        validation_alias=AliasChoices("RESEARCHER_PROMPT_DIR", "RUN4221_PROMPT_DIR"),
    )

    max_events_per_cycle: int = Field(default=5, ge=1, le=100)
    max_candidates_per_cycle: int = Field(default=3, ge=1, le=100)
    max_agent_turns_per_job: int = Field(default=6, ge=1, le=20)
    max_web_searches_per_job: int = Field(default=2, ge=0, le=20)
    max_static_pages_per_job: int = Field(default=4, ge=1, le=50)
    max_rendered_pages_per_job: int = Field(default=0, ge=0, le=10)
    max_retries_per_job: int = Field(default=2, ge=0, le=10)
    max_output_tokens_per_job: int = Field(default=4_000, ge=128, le=16_000)
    max_wall_time_seconds_per_job: int = Field(default=90, ge=10, le=900)
    max_pending_suggestions: int = Field(
        default=RESEARCHER_MAX_PENDING_SUGGESTIONS,
        ge=0,
        le=RESEARCHER_MAX_PENDING_SUGGESTIONS,
    )
    max_pending_updates: int = Field(default=50, ge=0, le=500)

    @property
    def budget(self) -> ResearchBudget:
        return ResearchBudget(
            max_events_per_cycle=self.max_events_per_cycle,
            max_candidates_per_cycle=self.max_candidates_per_cycle,
            max_agent_turns_per_job=self.max_agent_turns_per_job,
            max_web_searches_per_job=self.max_web_searches_per_job,
            max_static_pages_per_job=self.max_static_pages_per_job,
            max_rendered_pages_per_job=self.max_rendered_pages_per_job,
            max_retries_per_job=self.max_retries_per_job,
            max_output_tokens_per_job=self.max_output_tokens_per_job,
            max_wall_time_seconds_per_job=self.max_wall_time_seconds_per_job,
            max_pending_suggestions=self.max_pending_suggestions,
            max_pending_updates=self.max_pending_updates,
        )

    @property
    def trusted_domain_values(self) -> frozenset[str]:
        return frozenset(_comma_separated(self.trusted_domains))

    @property
    def trusted_registry_url_values(self) -> tuple[str, ...]:
        return _comma_separated(self.trusted_registry_urls)

    @property
    def discovery_query_values(self) -> tuple[str, ...]:
        return _comma_separated(self.discovery_queries)


def load_researcher_prompt(settings: ResearcherSettings) -> PromptVersionRecord:
    """Load the researcher prompt through the existing file/DB versioning boundary."""

    if settings.prompt_source == "db":
        return get_runtime_prompt(
            RESEARCH_AGENT_PROMPT,
            database_url=settings.database_url,
        )

    return get_file_prompt(RESEARCH_AGENT_PROMPT, prompts_dir=settings.prompt_dir)


@lru_cache(maxsize=1)
def get_researcher_settings() -> ResearcherSettings:
    return ResearcherSettings()


def _comma_separated(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
