from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from run4221.db.bootstrap import initialize_database
from run4221.db.repository import EventCreate, EventSuggestionCreate, add_event
from run4221.db.research import admit_suggestion, get_research_source, list_due_sources
from run4221.researcher.app import (
    DiscoverySchedule,
    ResearcherWorker,
    async_main,
    check_config,
    run_with_lock,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.config import ResearcherSettings
from run4221.researcher.health import HealthStore
from run4221.researcher.schemas import (
    ArtifactReference,
    ResearchCandidate,
    ResearchDecision,
    ResearchRunStatus,
)
from run4221.researcher.service import ResearchJobResult, decision_queue_marker

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def database_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'researcher.sqlite3'}"


def settings(tmp_path: Path, **overrides: object) -> ResearcherSettings:
    values: dict[str, object] = {
        "_env_file": None,
        "openai_api_key": "test-key",
        "database_url": database_url(tmp_path),
        "prompt_source": "file",
        "prompt_dir": str(tmp_path / "prompts"),
        "artifact_dir": str(tmp_path / "runs"),
        "schedule_path": str(tmp_path / "researcher-schedule.json"),
        "lock_path": str(tmp_path / "researcher.lock"),
        "health_path": str(tmp_path / "researcher-health.json"),
        "trusted_domains": "official.example, registry.example",
        "trusted_registry_urls": "https://registry.example/marathons",
        "discovery_queries": "germany marathon 2027, france marathon 2027",
        "enabled": True,
        "discovery_enabled": True,
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


def add_source(tmp_path: Path, *, public_id: str = "test.42") -> str:
    event = add_event(
        EventCreate(
            public_id=public_id,
            name="Test Marathon",
            city="Test City",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2027-05-01",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url=f"https://official.example/{public_id}",
        ),
        database_url=database_url(tmp_path),
    )
    return event.id


def result(*, failed: bool = False) -> ResearchJobResult:
    run_id = str(uuid4())
    status = ResearchRunStatus(
        status="failed" if failed else "succeeded",
        outcome="inconclusive" if failed else "no_change",
    )
    reference = ArtifactReference(
        run_id=run_id,
        artifact_name="terminal.json",
        source_url="https://official.example/event",
        content_hash="a" * 64,
    )
    return ResearchJobResult(run_id, status, reference)


class FakeService:
    def __init__(self, root: Path, on_run_started, calls: list[tuple[str, object]], fail=False):
        self.artifacts = ResearchArtifactStore(root)
        self.on_run_started = on_run_started
        self.calls = calls
        self.fail = fail

    async def refresh(self, source):
        self.calls.append(("refresh", source.event.id))
        self.on_run_started(str(uuid4()))
        if self.fail:
            raise RuntimeError("provider down")
        return result()

    async def discover(self, query: str):
        self.calls.append(("discover", query))
        self.on_run_started(str(uuid4()))
        if self.fail:
            raise RuntimeError("provider down")
        return result()


class Factory:
    def __init__(self, tmp_path: Path, failures: list[bool] | None = None) -> None:
        self.tmp_path = tmp_path
        self.failures = list(failures or [])
        self.calls: list[tuple[str, object]] = []
        self.shadow_modes: list[bool] = []

    def __call__(self, shadow: bool, on_run_started):
        self.shadow_modes.append(shadow)
        fail = self.failures.pop(0) if self.failures else False
        return FakeService(self.tmp_path / "failure-runs", on_run_started, self.calls, fail)


def worker(tmp_path: Path, factory: Factory, **overrides: object) -> ResearcherWorker:
    config = settings(tmp_path, **overrides)
    return ResearcherWorker(
        config,
        service_factory=factory,
        health=HealthStore(config.health_path, now=lambda: NOW),
        schedule=DiscoverySchedule(config.schedule_path),
        now=lambda: NOW,
    )


def test_check_config_validates_without_constructing_agent(tmp_path: Path) -> None:
    initialize(tmp_path)
    config = settings(tmp_path)

    checked = check_config(config)

    assert checked.prompt.prompt_key == "research_agent"
    assert checked.model == "gpt-5.6-luna"
    assert checked.budget.max_agent_turns_per_job == 6


def test_check_config_cli_makes_no_agent_or_api_call(tmp_path: Path, monkeypatch) -> None:
    initialize(tmp_path)
    config = settings(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", config.database_url)
    monkeypatch.setenv("RESEARCHER_PROMPT_DIR", config.prompt_dir)
    monkeypatch.setenv("RESEARCHER_TRUSTED_DOMAINS", config.trusted_domains)
    monkeypatch.setenv("RESEARCHER_TRUSTED_REGISTRY_URLS", config.trusted_registry_urls)
    monkeypatch.setenv("RESEARCHER_DISCOVERY_QUERIES", config.discovery_queries)
    monkeypatch.setenv("RESEARCHER_DISCOVERY_ENABLED", "true")
    monkeypatch.setattr(
        "run4221.researcher.app.set_default_openai_key",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("agent configured")),
    )

    assert asyncio.run(async_main(["--check-config"])) == 0


def test_operator_one_shot_processes_exact_event_and_shadow_is_read_only(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    factory = Factory(tmp_path)
    app = worker(tmp_path, factory)
    app.initialize_health()

    execution = asyncio.run(app.run_event(event_id, shadow=True))

    assert execution.failed is False
    assert factory.calls == [("refresh", event_id)]
    assert factory.shadow_modes == [True]
    assert get_research_source(event_id, database_url=database_url(tmp_path)) is not None


def test_lock_contention_skips_without_constructing_service(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    factory = Factory(tmp_path)
    app = worker(tmp_path, factory)

    from run4221.researcher.lock import ProcessLock

    first = ProcessLock(app.settings.lock_path)
    assert first.acquire()
    try:
        executed = asyncio.run(run_with_lock(app, lambda: app.run_event(event_id)))
    finally:
        first.release()

    assert executed is None
    assert factory.calls == []


def test_cycle_claims_before_failure_continues_and_restart_does_not_readmit(tmp_path: Path) -> None:
    initialize(tmp_path)
    first_event = add_source(tmp_path, public_id="first.42")
    second_event = add_source(tmp_path, public_id="second.42")
    factory = Factory(tmp_path, failures=[True, False, False])
    app = worker(
        tmp_path,
        factory,
        max_events_per_cycle=2,
        discovery_queries="germany marathon 2027",
    )
    app.initialize_health()

    cycle = asyncio.run(app.run_cycle())

    assert cycle.attempted == 3  # two sources, then one bounded discovery job
    assert cycle.failed == 1
    assert factory.calls[:2] == [("refresh", first_event), ("refresh", second_event)]
    assert app.health.read().consecutive_failures == 0  # later successes recover health
    assert list_due_sources(
        due_before=NOW - timedelta(days=1),
        limit=10,
        database_url=database_url(tmp_path),
    ) == ()

    restarted_factory = Factory(tmp_path)
    restarted = worker(
        tmp_path,
        restarted_factory,
        max_events_per_cycle=2,
        discovery_queries="germany marathon 2027",
    )
    restarted.initialize_health()
    same_day = asyncio.run(restarted.run_cycle())

    assert same_day.attempted == 0
    assert restarted_factory.calls == []

    one_shot = asyncio.run(restarted.run_event(first_event))
    assert one_shot.failed is False
    assert restarted_factory.calls == [("refresh", first_event)]


def test_loop_survives_job_failure_and_runs_next_cycle(tmp_path: Path) -> None:
    initialize(tmp_path)
    add_source(tmp_path)
    factory = Factory(tmp_path, failures=[True])
    sleeps: list[float] = []

    async def no_wait(seconds: float) -> None:
        sleeps.append(seconds)

    app = ResearcherWorker(
        settings(tmp_path, discovery_enabled=False),
        service_factory=factory,
        health=HealthStore(tmp_path / "researcher-health.json", now=lambda: NOW),
        schedule=DiscoverySchedule(tmp_path / "researcher-schedule.json"),
        now=lambda: NOW,
        sleep=no_wait,
    )
    app.initialize_health()

    asyncio.run(app.run_loop(max_cycles=2))

    assert len(factory.calls) == 1
    assert sleeps == [app.settings.interval_seconds]


def test_discovery_schedule_claim_is_atomic_and_keyed_by_query_revision(tmp_path: Path) -> None:
    schedule = DiscoverySchedule(tmp_path / "schedule.json")

    assert schedule.claim("Germany  marathon 2027", NOW)
    assert not schedule.claim(" germany marathon 2027 ", NOW + timedelta(hours=2))
    assert schedule.claim("Germany marathon 2028", NOW + timedelta(hours=2))
    assert schedule.claim("Germany marathon 2027", NOW + timedelta(days=1))
    assert list(tmp_path.glob(".*.tmp")) == []


def test_startup_reconciles_a_committed_prepared_decision(tmp_path: Path) -> None:
    initialize(tmp_path)
    config = settings(tmp_path, discovery_enabled=False)
    store = ResearchArtifactStore(config.artifact_dir)
    run_id = store.create_run(job_type="discovery")
    evidence = store.write_artifact(
        run_id,
        artifact_type="page_snapshot",
        source_url="https://official.example/reconcile-marathon",
        content={"captured": True},
    )
    decision = ResearchDecision(
        action="suggest_event",
        summary="Captured official event page.",
        candidate=ResearchCandidate(
            source_url="https://official.example/reconcile-marathon",
            title="Reconcile Marathon",
            snippet="Marathon details.",
            distances=("marathon",),
        ),
        evidence=[evidence],
    )
    prepared = store.prepare_decision(
        run_id,
        source_url=decision.candidate.source_url,
        decision=decision,
        committed_status=ResearchRunStatus(
            status="succeeded",
            outcome="suggestion_created",
        ),
    )
    admission = admit_suggestion(
        EventSuggestionCreate(
            event_name="Reconcile Marathon",
            url=decision.candidate.source_url,
            event_date=None,
            location=None,
            region_tags=(),
            distances=("marathon",),
            note=decision_queue_marker(prepared),
            submitter_user_id=None,
            submitter_username=None,
            submitter_display_name=None,
        ),
        database_url=config.database_url,
    )
    assert admission.outcome == "admitted"

    app = ResearcherWorker(
        config,
        service_factory=Factory(tmp_path),
        health=HealthStore(config.health_path, now=lambda: NOW),
    )
    reconciled = app.reconcile_prepared()

    assert len(reconciled) == 1
    assert reconciled[0].state == "committed"
    assert reconciled[0].terminal_reference is not None
    assert store.read_artifact(reconciled[0].terminal_reference)["content"][
        "queue_reference"
    ] == "event_suggestion:1"
