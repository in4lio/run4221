from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.db.repository import EventCreate, ProposedEventUpdateCreate, add_event
from run4221.db.research import (
    admit_proposed_update,
    get_research_source,
    list_due_sources,
)
from run4221.researcher.app import (
    ResearcherWorker,
    async_main,
    check_config,
    run_with_lock,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.config import ResearcherSettings
from run4221.researcher.engine import SourceNotFoundError
from run4221.researcher.health import HealthStore
from run4221.researcher.schemas import (
    ArtifactReference,
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
        "lock_path": str(tmp_path / "researcher.lock"),
        "health_path": str(tmp_path / "researcher-health.json"),
        "schedule_enabled": True,
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


def add_source(
    tmp_path: Path,
    *,
    public_id: str = "test.42",
    registration_url: str | None = None,
) -> str:
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
            registration_url=registration_url,
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
    monkeypatch.setattr(
        "run4221.researcher.engine.set_default_openai_key",
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


def test_run_event_without_active_source_raises_typed_error(tmp_path: Path) -> None:
    initialize(tmp_path)
    factory = Factory(tmp_path)
    app = worker(tmp_path, factory)
    app.initialize_health()

    with pytest.raises(SourceNotFoundError, match="No active research source"):
        asyncio.run(app.run_event("ghost.42"))

    assert factory.calls == []


def test_run_event_prefers_the_registration_page_source(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(
        tmp_path,
        registration_url="https://register.example/test.42",
    )
    refreshed_urls: list[str] = []

    class RecordingService:
        def __init__(self) -> None:
            self.artifacts = ResearchArtifactStore(tmp_path / "runs")

        async def refresh(self, source):
            refreshed_urls.append(source.url)
            return result()

    app = ResearcherWorker(
        settings(tmp_path),
        service_factory=lambda shadow, on_run_started: RecordingService(),
        health=HealthStore(tmp_path / "researcher-health.json", now=lambda: NOW),
        now=lambda: NOW,
    )
    app.initialize_health()

    execution = asyncio.run(app.run_event(event_id))

    assert execution.failed is False
    # One-shot runs resolve with the engine's preference: registration first.
    assert refreshed_urls == ["https://register.example/test.42"]


def test_once_cli_reports_missing_source_with_typed_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = initialize(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("RESEARCHER_PROMPT_DIR", str(tmp_path / "prompts"))

    exit_code = asyncio.run(async_main(["--once", "--event-id", "ghost.42"]))

    assert exit_code == 2
    assert "No active research source for event: ghost.42" in capsys.readouterr().err


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
    factory = Factory(tmp_path, failures=[True, False])
    app = worker(tmp_path, factory, max_events_per_cycle=2)
    app.initialize_health()

    cycle = asyncio.run(app.run_cycle())

    assert cycle.attempted == 2  # exactly the due sources, nothing else
    assert cycle.failed == 1
    assert factory.calls == [("refresh", first_event), ("refresh", second_event)]
    assert app.health.read().consecutive_failures == 0  # later successes recover health
    assert list_due_sources(
        due_before=NOW - timedelta(days=1),
        limit=10,
        database_url=database_url(tmp_path),
    ) == ()

    restarted_factory = Factory(tmp_path)
    restarted = worker(tmp_path, restarted_factory, max_events_per_cycle=2)
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
        settings(tmp_path),
        service_factory=factory,
        health=HealthStore(tmp_path / "researcher-health.json", now=lambda: NOW),
        now=lambda: NOW,
        sleep=no_wait,
    )
    app.initialize_health()

    asyncio.run(app.run_loop(max_cycles=2))

    assert len(factory.calls) == 1
    assert sleeps == [app.settings.interval_seconds]


def prepare_absent_decision(
    config: ResearcherSettings,
    event_id: str,
    *,
    admit: bool = False,
) -> tuple[ResearchArtifactStore, str]:
    """Create one prepared-but-unfinalized run old enough to be reconciled."""

    # Producing the run in the past keeps it outside the 2x producer-cap
    # freshness window when the worker reconciles with the real clock.
    store = ResearchArtifactStore(config.artifact_dir, now=lambda: NOW)
    run_id = store.create_run(job_type="refresh")
    source_url = f"https://official.example/{event_id}"
    evidence = store.write_artifact(
        run_id,
        artifact_type="page_snapshot",
        source_url=source_url,
        content={"captured": True},
    )
    decision = ResearchDecision.model_validate(
        {
            "action": "propose_update",
            "summary": "The captured approved source says registration is open.",
            "confidence": 0.95,
            "proposed_fields": {"registration_status": "open"},
            "evidence": [evidence],
            "applicability": [
                {
                    "evidence": evidence,
                    "event_identity": "confirmed",
                    "event_edition": "confirmed",
                    "distance_category": "confirmed",
                    "applicable_fields": ["registration_status"],
                }
            ],
            "field_support": [
                {"field": "registration_status", "evidence": [evidence]}
            ],
        }
    )
    prepared = store.prepare_decision(
        run_id,
        source_url=source_url,
        decision=decision,
        committed_status=ResearchRunStatus(
            status="succeeded",
            outcome="proposal_created",
        ),
        producer_deadline_seconds=90,
    )
    if admit:
        admission = admit_proposed_update(
            ProposedEventUpdateCreate(
                event_id=event_id,
                update_type="registration_window",
                current_fields={"registration_status": "unknown"},
                proposed_fields={"registration_status": "open"},
                evidence=(decision_queue_marker(prepared),),
                confidence=0.95,
                change_summary=decision.summary,
            ),
            database_url=config.database_url,
        )
        assert admission.outcome == "admitted"
    return store, run_id


def test_startup_reconciles_a_committed_prepared_decision(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    config = settings(tmp_path)
    store, _run_id = prepare_absent_decision(config, event_id, admit=True)

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
    ] == "proposed_event_update:1"


def test_worker_skips_reconciling_a_fresh_prepared_run(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    config = settings(tmp_path)
    # Prepared with the real clock: still inside its 2x producer-cap window.
    store = ResearchArtifactStore(config.artifact_dir)
    run_id = store.create_run(job_type="refresh")
    source_url = f"https://official.example/{event_id}"
    evidence = store.write_artifact(
        run_id,
        artifact_type="page_snapshot",
        source_url=source_url,
        content={"captured": True},
    )
    decision = ResearchDecision.model_validate(
        {
            "action": "propose_update",
            "summary": "The captured approved source says registration is open.",
            "confidence": 0.95,
            "proposed_fields": {"registration_status": "open"},
            "evidence": [evidence],
            "applicability": [
                {
                    "evidence": evidence,
                    "event_identity": "confirmed",
                    "event_edition": "confirmed",
                    "distance_category": "confirmed",
                    "applicable_fields": ["registration_status"],
                }
            ],
            "field_support": [
                {"field": "registration_status", "evidence": [evidence]}
            ],
        }
    )
    store.prepare_decision(
        run_id,
        source_url=source_url,
        decision=decision,
        committed_status=ResearchRunStatus(
            status="succeeded",
            outcome="proposal_created",
        ),
        producer_deadline_seconds=90,
    )

    app = ResearcherWorker(
        config,
        service_factory=Factory(tmp_path),
        health=HealthStore(config.health_path, now=lambda: NOW),
    )
    reconciled = app.reconcile_prepared()

    assert reconciled == ()
    assert not (Path(config.artifact_dir) / run_id / "terminal.json").exists()


def terminal_payload(config: ResearcherSettings, run_id: str) -> dict[str, object]:
    path = Path(config.artifact_dir) / run_id / "terminal.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_worker_reconciles_prepared_runs_at_every_cycle_start(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    config = settings(tmp_path)
    _store, first_run = prepare_absent_decision(config, event_id)
    app = worker(tmp_path, Factory(tmp_path))
    app.initialize_health()

    asyncio.run(app.run_cycle())

    assert terminal_payload(config, first_run)["content"]["queue_state"] == "absent"

    # A second cycle reconciles again: a run prepared between cycles is
    # finished at the next cycle start, not only at process startup.
    _store, second_run = prepare_absent_decision(config, event_id)
    asyncio.run(app.run_cycle())

    assert terminal_payload(config, second_run)["content"]["queue_state"] == "absent"


def test_worker_second_cycle_skips_previously_terminal_run(tmp_path: Path) -> None:
    initialize(tmp_path)
    event_id = add_source(tmp_path)
    config = settings(tmp_path)
    _store, run_id = prepare_absent_decision(config, event_id)
    app = worker(tmp_path, Factory(tmp_path))
    app.initialize_health()

    asyncio.run(app.run_cycle())
    terminal_path = Path(config.artifact_dir) / run_id / "terminal.json"
    assert terminal_path.exists()

    # A run cached as terminal is never re-read: even with the terminal file
    # gone, the next cycle must not reconcile that directory again.
    terminal_path.unlink()
    asyncio.run(app.run_cycle())

    assert not terminal_path.exists()


def old_terminal_run(
    config: ResearcherSettings,
    *,
    age_days: int,
    finalize: bool = True,
) -> str:
    store = ResearchArtifactStore(
        config.artifact_dir,
        now=lambda: NOW - timedelta(days=age_days),
    )
    run_id = store.create_run(job_type="refresh")
    if finalize:
        store.finalize_without_queue(
            run_id,
            source_url="https://official.example/expired",
            status=ResearchRunStatus(status="skipped", outcome="inconclusive"),
        )
    return run_id


def test_cycle_prunes_only_expired_terminal_runs(tmp_path: Path) -> None:
    initialize(tmp_path)
    config = settings(tmp_path, schedule_enabled=False)
    expired_terminal = old_terminal_run(config, age_days=120)
    expired_open = old_terminal_run(config, age_days=120, finalize=False)
    young_terminal = old_terminal_run(config, age_days=1)
    app = worker(tmp_path, Factory(tmp_path), schedule_enabled=False)
    app.initialize_health()

    asyncio.run(app.run_cycle())

    runs_root = Path(config.artifact_dir)
    assert not (runs_root / expired_terminal).exists()
    # Runs without terminal.json are never touched, however old they are.
    assert (runs_root / expired_open).exists()
    assert (runs_root / young_terminal).exists()


def test_zero_run_retention_days_disables_pruning(tmp_path: Path) -> None:
    initialize(tmp_path)
    config = settings(tmp_path, run_retention_days=0)
    expired_terminal = old_terminal_run(config, age_days=365)
    app = worker(tmp_path, Factory(tmp_path), run_retention_days=0)
    app.initialize_health()

    asyncio.run(app.run_cycle())

    assert (Path(config.artifact_dir) / expired_terminal).exists()
