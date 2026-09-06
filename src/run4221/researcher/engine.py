from __future__ import annotations

import asyncio
from collections.abc import Callable

from agents import set_default_openai_key

from run4221.db.bootstrap import ensure_database_schema
from run4221.db.prompts import PromptVersionRecord
from run4221.db.research import get_refresh_source
from run4221.researcher.agent import ResearchAgentJob
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.config import ResearcherSettings, load_researcher_prompt
from run4221.researcher.service import (
    ProfileJobResult,
    ResearcherService,
    ResearchJobResult,
)


class EngineConfigError(Exception):
    """The researcher engine cannot be configured from settings or its prompt."""


class SourceNotFoundError(LookupError):
    """No active research source exists for the requested event.

    Deliberately not a ValueError: callers map this exact condition to a
    "no active source" message, and a pydantic ValidationError (a ValueError
    subclass) must keep surfacing through the generic failure path instead.
    """


class ResearchEngine:
    """One AI lever, two entry points.

    Cached state is only the validated settings, the loaded prompt record,
    and a schema-initialization flag; every call constructs a fresh
    single-use job and service so each entry point starts with a full budget.
    """

    def __init__(
        self,
        *,
        settings: ResearcherSettings,
        prompt: PromptVersionRecord,
    ) -> None:
        self._settings = settings
        self._prompt = prompt
        self._schema_ready = False

    @property
    def settings(self) -> ResearcherSettings:
        return self._settings

    @property
    def prompt(self) -> PromptVersionRecord:
        return self._prompt

    def build_service(
        self,
        *,
        persist_queue: bool = True,
        on_run_started: Callable[[str], None] | None = None,
    ) -> ResearcherService:
        """Construct one fresh single-job service; the engine's only factory."""

        if not self._schema_ready:
            # Schema creation runs once per engine, not once per service call:
            # the per-job accessor functions must never pay create_all again.
            ensure_database_schema(self._settings.database_url)
            self._schema_ready = True
        prompt_reference = (
            f"{self._prompt.prompt_key}:{self._prompt.source}:v{self._prompt.version}"
        )
        budget = self._settings.budget
        return ResearcherService(
            database_url=self._settings.database_url,
            artifacts=ResearchArtifactStore(self._settings.artifact_dir),
            agent=ResearchAgentJob(
                instructions=self._prompt.content,
                prompt_reference=prompt_reference,
                budget=budget,
                model=self._settings.model,
            ),
            budget=budget,
            persist_queue=persist_queue,
            on_run_started=on_run_started,
        )

    async def profile(self, url: str) -> ProfileJobResult:
        """Draft one cited event profile; profile never persists by construction."""

        service = await asyncio.to_thread(self.build_service, persist_queue=True)
        return await service.profile(url)

    async def refresh_source(self, event_id: str) -> ResearchJobResult:
        """Refresh the event's registration page source, else its top source."""

        source = await asyncio.to_thread(
            get_refresh_source,
            event_id,
            database_url=self._settings.database_url,
        )
        if source is None:
            raise SourceNotFoundError(f"No active research source for event: {event_id}")
        service = await asyncio.to_thread(self.build_service, persist_queue=True)
        return await service.refresh(source)


def build_engine(settings: ResearcherSettings | None = None) -> ResearchEngine:
    """Load settings and prompt once, inject the provider key once, fail closed."""

    if settings is None:
        try:
            settings = ResearcherSettings()
        except Exception as error:
            raise EngineConfigError(
                "Researcher settings are invalid or incomplete."
            ) from error
    try:
        prompt = load_researcher_prompt(settings)
    except Exception as error:
        raise EngineConfigError("Researcher prompt could not be loaded.") from error
    set_default_openai_key(
        settings.openai_api_key.get_secret_value(),
        use_for_tracing=False,
    )
    return ResearchEngine(settings=settings, prompt=prompt)
