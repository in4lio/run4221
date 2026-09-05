from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.db.repository import EventCreate, add_event
from run4221.researcher import engine as engine_module
from run4221.researcher.agent import ResearchAgentJob
from run4221.researcher.config import ResearcherSettings
from run4221.researcher.engine import (
    EngineConfigError,
    ResearchEngine,
    SourceNotFoundError,
    build_engine,
)
from run4221.researcher.service import ResearcherService


def database_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'researcher-engine.sqlite3'}"


def settings(tmp_path: Path, **overrides: object) -> ResearcherSettings:
    values: dict[str, object] = {
        "_env_file": None,
        "openai_api_key": "test-key",
        "database_url": database_url(tmp_path),
        "prompt_source": "file",
        "prompt_dir": str(tmp_path / "prompts"),
        "artifact_dir": str(tmp_path / "runs"),
        "lock_path": str(tmp_path / "researcher.lock"),
        "health_path": str(tmp_path / "researcher-health.json"),
    }
    values.update(overrides)
    return ResearcherSettings(**values)


def initialize(tmp_path: Path) -> str:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    prompts = tmp_path / "prompts"
    prompts.mkdir(exist_ok=True)
    (prompts / "research_agent.instructions.txt").write_text(
        "Use captured evidence only.", encoding="utf-8"
    )
    return url


def add_tracked_event(
    tmp_path: Path,
    *,
    public_id: str = "test.42",
    name: str = "Test Marathon",
    registration_url: str | None = None,
) -> str:
    event = add_event(
        EventCreate(
            public_id=public_id,
            name=name,
            city="Test City",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2027-05-01",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url=f"https://official.example/{public_id}",
            registration_url=registration_url,
        ),
        database_url=database_url(tmp_path),
    )
    return event.id


def quiet_key_injection(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    injected: list[str] = []
    monkeypatch.setattr(
        engine_module,
        "set_default_openai_key",
        lambda key, use_for_tracing: injected.append(key),
    )
    return injected


def install_fake_service(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    instances: list[object] = []

    class FakeService:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            instances.append(self)

        async def profile(self, url: str) -> tuple[str, str, object]:
            return ("profile", url, self)

        async def refresh(self, source: object) -> tuple[str, object, object]:
            return ("refresh", source, self)

    monkeypatch.setattr(engine_module, "ResearcherService", FakeService)
    return instances


def test_ae3_build_engine_without_key_raises_engine_config_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RESEARCHER_OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(EngineConfigError, match="settings"):
        build_engine()


def test_build_engine_with_missing_prompt_raises_engine_config_error(
    tmp_path: Path,
) -> None:
    (tmp_path / "prompts").mkdir()

    with pytest.raises(EngineConfigError, match="prompt"):
        build_engine(settings(tmp_path))


def test_ae3_key_is_injected_once_across_repeated_engine_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize(tmp_path)
    injected = quiet_key_injection(monkeypatch)
    instances = install_fake_service(monkeypatch)

    engine = build_engine(settings(tmp_path))
    asyncio.run(engine.profile("https://example.com/a"))
    asyncio.run(engine.profile("https://example.com/b"))

    assert injected == ["test-key"]
    assert len(instances) == 2


def test_each_engine_call_constructs_a_fresh_full_budget_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize(tmp_path)
    quiet_key_injection(monkeypatch)
    instances = install_fake_service(monkeypatch)
    config = settings(tmp_path)
    engine = build_engine(config)

    asyncio.run(engine.profile("https://example.com/a"))
    first_job = instances[0].kwargs["agent"]
    assert isinstance(first_job, ResearchAgentJob)
    # Exhaust the first single-use job before the second call starts.
    first_job.budget.remaining_turns = 0

    asyncio.run(engine.profile("https://example.com/b"))
    second_job = instances[1].kwargs["agent"]

    assert second_job is not first_job
    assert second_job.budget.remaining_turns == config.max_agent_turns_per_job
    assert second_job.budget.remaining_web_searches == config.max_web_searches_per_job
    assert second_job.budget.remaining_output_tokens == config.max_output_tokens_per_job


def test_refresh_source_prefers_active_registration_page_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize(tmp_path)
    quiet_key_injection(monkeypatch)
    instances = install_fake_service(monkeypatch)
    with_registration = add_tracked_event(
        tmp_path,
        registration_url="https://register.example/test.42",
    )
    official_only = add_tracked_event(
        tmp_path,
        public_id="plain.42",
        name="Plain Marathon",
    )
    engine = build_engine(settings(tmp_path))

    _, registration_source, _ = asyncio.run(engine.refresh_source(with_registration))
    _, official_source, _ = asyncio.run(engine.refresh_source(official_only))

    assert registration_source.url == "https://register.example/test.42"
    assert official_source.url == "https://official.example/plain.42"
    assert all(service.kwargs["persist_queue"] is True for service in instances)


def test_refresh_source_without_active_source_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialize(tmp_path)
    quiet_key_injection(monkeypatch)
    install_fake_service(monkeypatch)
    engine = build_engine(settings(tmp_path))

    with pytest.raises(SourceNotFoundError, match="No active research source"):
        asyncio.run(engine.refresh_source("ghost.42"))
    # The typed error must never be a ValueError: callers map the type to a
    # "no active source" message and a pydantic ValidationError (a ValueError
    # subclass) has to keep surfacing through the generic failure path.
    assert not issubclass(SourceNotFoundError, ValueError)


def test_engine_caches_only_settings_and_prompt(tmp_path: Path) -> None:
    initialize(tmp_path)
    config = settings(tmp_path)
    engine = ResearchEngine(
        settings=config,
        prompt=engine_module.load_researcher_prompt(config),
    )

    service = engine.build_service(persist_queue=False)

    assert isinstance(service, ResearcherService)
    assert service.persist_queue is False
    assert service.database_url == config.database_url
    assert engine.settings is config
    assert engine.prompt.prompt_key == "research_agent"
