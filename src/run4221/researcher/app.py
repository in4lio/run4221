from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agents import set_default_openai_key

from run4221.db.prompts import PromptVersionRecord
from run4221.db.research import (
    ResearchSourceRecord,
    find_research_queue_reference,
    get_research_source,
    list_due_sources,
    mark_source_checked,
)
from run4221.db.session import require_initialized_database
from run4221.researcher.agent import ResearchAgentJob
from run4221.researcher.artifacts import (
    QueueResolution,
    ReconciliationResult,
    ResearchArtifactStore,
    fsync_directory,
)
from run4221.researcher.config import ResearcherSettings, load_researcher_prompt
from run4221.researcher.health import HealthStore, check_researcher_health
from run4221.researcher.lock import ProcessLock
from run4221.researcher.policy import SourceTrustPolicy
from run4221.researcher.schemas import ResearchBudget, ResearchRunStatus, RunOutcome, RunState
from run4221.researcher.service import (
    AUDIT_SOURCE_URL,
    ResearcherService,
    ResearchJobResult,
    decision_queue_marker,
)

LOGGER = logging.getLogger(__name__)
DAILY_WINDOW = timedelta(days=1)


@dataclass(frozen=True)
class CheckedConfig:
    prompt: PromptVersionRecord
    model: str
    budget: ResearchBudget
    trust_policy: SourceTrustPolicy

@dataclass(frozen=True)
class JobExecution:
    result: ResearchJobResult
    failed: bool


@dataclass(frozen=True)
class CycleResult:
    attempted: int = 0
    failed: int = 0


class ServiceFactory(Protocol):
    def __call__(
        self,
        shadow: bool,
        on_run_started: Callable[[str], None],
    ) -> ResearcherService: ...


class DiscoverySchedule:
    """Crash-safe once-per-UTC-day admission for discovery query revisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def validate(self) -> None:
        self._read()

    def claim(self, query: str, at: datetime) -> bool:
        if at.tzinfo is None:
            raise ValueError("Schedule timestamps must be timezone-aware.")
        normalized = _normalize_query(query)
        if not normalized:
            raise ValueError("Discovery query cannot be empty.")
        query_hash = _query_hash(normalized)
        window = at.astimezone(UTC).date().isoformat()
        payload = self._read()
        claims = payload["claims"]
        assert isinstance(claims, dict)
        current = claims.get(query_hash)
        if isinstance(current, dict) and current.get("window") == window:
            return False
        claims[query_hash] = {
            "window": window,
            "claimed_at": at.astimezone(UTC).isoformat(),
        }
        self._write(payload)
        return True

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return {"schema_version": 1, "claims": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Researcher discovery schedule is invalid.") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("claims"), dict)
        ):
            raise RuntimeError("Researcher discovery schedule is invalid.")
        return payload

    def _write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid4()}.tmp"
        raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            fsync_directory(self.path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)


def check_config(settings: ResearcherSettings) -> CheckedConfig:
    """Validate startup dependencies without constructing an agent or calling OpenAI."""

    require_initialized_database(settings.database_url)
    prompt = load_researcher_prompt(settings)
    trust_policy = SourceTrustPolicy(
        trusted_domains=settings.trusted_domain_values,
        trusted_registry_urls=settings.trusted_registry_url_values,
    )
    if settings.discovery_enabled and not settings.discovery_query_values:
        raise RuntimeError("Discovery is enabled but no discovery queries are configured.")
    if settings.discovery_enabled and not trust_policy.trusted_domains:
        raise RuntimeError("Discovery is enabled but no trusted domains are configured.")
    DiscoverySchedule(settings.schedule_path).validate()
    return CheckedConfig(
        prompt=prompt,
        model=settings.model,
        budget=settings.budget,
        trust_policy=trust_policy,
    )


class ResearcherWorker:
    def __init__(
        self,
        settings: ResearcherSettings,
        *,
        checked: CheckedConfig | None = None,
        service_factory: ServiceFactory | None = None,
        health: HealthStore | None = None,
        schedule: DiscoverySchedule | None = None,
        now: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.settings = settings
        self.checked = checked
        self.health = health or HealthStore(settings.health_path)
        self.schedule = schedule or DiscoverySchedule(settings.schedule_path)
        self.now = now or (lambda: datetime.now(UTC))
        self.sleep = sleep
        self.service_factory = service_factory or self._default_service_factory

    def initialize_health(self) -> None:
        self.health.initialize(enabled=self.settings.enabled)

    def reconcile_prepared(self) -> tuple[ReconciliationResult, ...]:
        """Finish interrupted queue lifecycles before accepting new work."""

        store = ResearchArtifactStore(self.settings.artifact_dir)

        def resolve(prepared):
            try:
                queue_reference = find_research_queue_reference(
                    prepared.decision.action.value,
                    decision_marker=decision_queue_marker(prepared.reference),
                    database_url=self.settings.database_url,
                )
            except Exception as error:
                return QueueResolution.inconclusive(
                    f"Queue lookup failed with {type(error).__name__}."
                )
            if queue_reference is None:
                return QueueResolution.absent()
            return QueueResolution.committed(queue_reference)

        return store.reconcile_prepared(resolve)

    async def run_event(self, event_id: str, *, shadow: bool = False) -> JobExecution:
        source = get_research_source(event_id, database_url=self.settings.database_url)
        if source is None:
            raise ValueError(f"No active research source for event: {event_id}")
        return await self._execute_refresh(source, shadow=shadow)

    async def run_discovery(self, query: str, *, shadow: bool = False) -> JobExecution:
        return await self._execute_job(
            label=f"discovery:{_query_hash(query)[:12]}",
            source_url=AUDIT_SOURCE_URL,
            shadow=shadow,
            invoke=lambda service: service.discover(query),
        )

    async def run_cycle(self, *, shadow: bool = False) -> CycleResult:
        self.health.set_idle(enabled=self.settings.enabled)
        if not self.settings.enabled:
            return CycleResult()

        now = self.now()
        sources = list_due_sources(
            due_before=now - DAILY_WINDOW,
            limit=self.settings.max_events_per_cycle,
            database_url=self.settings.database_url,
        )
        attempted = failed = 0
        for source in sources:
            if not mark_source_checked(
                source.source_id,
                checked_at=now,
                database_url=self.settings.database_url,
            ):
                continue
            execution = await self._execute_refresh(source, shadow=shadow)
            attempted += 1
            failed += int(execution.failed)

        if self.settings.discovery_enabled:
            for query in self.settings.discovery_query_values:
                if not self.schedule.claim(query, now):
                    continue
                execution = await self.run_discovery(query, shadow=shadow)
                attempted += 1
                failed += int(execution.failed)
                break

        self.health.set_idle(enabled=True)
        return CycleResult(attempted=attempted, failed=failed)

    async def run_loop(self, *, shadow: bool = False, max_cycles: int | None = None) -> None:
        cycles = 0
        while max_cycles is None or cycles < max_cycles:
            try:
                await self.run_cycle(shadow=shadow)
            except Exception:
                LOGGER.exception("Researcher cycle failed; continuing after the interval.")
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return
            await self.sleep(self.settings.interval_seconds)

    async def _execute_refresh(
        self,
        source: ResearchSourceRecord,
        *,
        shadow: bool,
    ) -> JobExecution:
        return await self._execute_job(
            label=f"refresh:{source.event.id}",
            source_url=source.url,
            shadow=shadow,
            invoke=lambda service: service.refresh(source),
        )

    async def _execute_job(
        self,
        *,
        label: str,
        source_url: str,
        shadow: bool,
        invoke: Callable[[ResearcherService], Awaitable[ResearchJobResult]],
    ) -> JobExecution:
        self.health.start_job()
        announced_run_id: str | None = None

        def on_run_started(run_id: str) -> None:
            nonlocal announced_run_id
            announced_run_id = run_id
            self.health.progress(run_id)

        service = self.service_factory(shadow, on_run_started)
        pulse = asyncio.create_task(self._pulse_health())
        try:
            result = await invoke(service)
        except Exception as error:
            LOGGER.exception("Researcher job failed: %s", label)
            result = self._audit_failure(
                service,
                announced_run_id=announced_run_id,
                source_url=source_url,
                error=error,
            )
        finally:
            pulse.cancel()
            with suppress(asyncio.CancelledError):
                await pulse

        failed = result.status.status is RunState.FAILED
        self.health.finish_job(
            f"{result.status.status.value}:{result.status.outcome.value}",
            failed=failed,
        )
        return JobExecution(result=result, failed=failed)

    async def _pulse_health(self) -> None:
        interval = max(10, self.settings.health_stale_after_seconds // 3)
        while True:
            await asyncio.sleep(interval)
            self.health.progress()

    def _audit_failure(
        self,
        service: ResearcherService,
        *,
        announced_run_id: str | None,
        source_url: str,
        error: Exception,
    ) -> ResearchJobResult:
        status = ResearchRunStatus(
            status=RunState.FAILED,
            outcome=RunOutcome.INCONCLUSIVE,
            detail=f"Worker job failed ({type(error).__name__}).",
        )
        if announced_run_id is not None:
            try:
                terminal = service.artifacts.finalize_without_queue(
                    announced_run_id,
                    source_url=source_url,
                    status=status,
                )
                return ResearchJobResult(announced_run_id, status, terminal)
            except Exception:
                pass
        run_id = service.artifacts.create_run(
            job_type="worker_failure",
            metadata={"error_type": type(error).__name__},
        )
        terminal = service.artifacts.finalize_without_queue(
            run_id,
            source_url=source_url,
            status=status,
        )
        return ResearchJobResult(run_id, status, terminal)

    def _default_service_factory(
        self,
        shadow: bool,
        on_run_started: Callable[[str], None],
    ) -> ResearcherService:
        checked = self.checked or check_config(self.settings)
        set_default_openai_key(
            self.settings.openai_api_key.get_secret_value(),
            use_for_tracing=False,
        )
        prompt_reference = (
            f"{checked.prompt.prompt_key}:{checked.prompt.source}:v{checked.prompt.version}"
        )
        return ResearcherService(
            database_url=self.settings.database_url,
            artifacts=ResearchArtifactStore(self.settings.artifact_dir),
            agent=ResearchAgentJob(
                instructions=checked.prompt.content,
                prompt_reference=prompt_reference,
                budget=checked.budget,
                model=checked.model,
            ),
            trust_policy=checked.trust_policy,
            budget=checked.budget,
            persist_queue=not shadow,
            on_run_started=on_run_started,
        )


async def run_with_lock(
    worker: ResearcherWorker,
    operation: Callable[[], Awaitable[object]],
) -> object | None:
    lock = ProcessLock(worker.settings.lock_path)
    if not lock.acquire():
        LOGGER.info("Researcher lock is already owned; skipping this invocation.")
        return None
    try:
        worker.initialize_health()
        worker.reconcile_prepared()
        return await operation()
    finally:
        lock.release()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run4221-researcher")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--discover", metavar="QUERY")
    mode.add_argument("--cycle", action="store_true")
    mode.add_argument("--loop", action="store_true")
    mode.add_argument("--health", action="store_true")
    parser.add_argument("--event-id")
    parser.add_argument("--shadow", action="store_true")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.health:
        path = os.getenv("RESEARCHER_HEALTH_PATH", "data/researcher_health.json")
        stale = int(os.getenv("RESEARCHER_HEALTH_STALE_AFTER_SECONDS", "180"))
        check_researcher_health(path, stale_after_seconds=stale)
        return 0
    if args.once != bool(args.event_id):
        parser.error("--once and --event-id must be used together")

    settings = ResearcherSettings()
    checked = check_config(settings)
    if args.check_config:
        print("run4221 researcher configuration passed")
        return 0

    worker = ResearcherWorker(settings, checked=checked)
    if args.once:
        async def operation() -> object:
            return await worker.run_event(args.event_id, shadow=args.shadow)
    elif args.discover is not None:
        async def operation() -> object:
            return await worker.run_discovery(args.discover, shadow=args.shadow)
    elif args.cycle:
        async def operation() -> object:
            return await worker.run_cycle(shadow=args.shadow)
    else:
        async def operation() -> object:
            return await worker.run_loop(shadow=args.shadow)
    await run_with_lock(worker, operation)
    return 0


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(asyncio.run(async_main()))


def _normalize_query(query: str) -> str:
    return " ".join(query.split())


def _query_hash(query: str) -> str:
    normalized = _normalize_query(query)
    if not normalized:
        raise ValueError("Discovery query cannot be empty.")
    return hashlib.sha256(normalized.casefold().encode()).hexdigest()

