from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

WorkerMode = Literal["paused", "enabled"]
WorkerActivity = Literal["idle", "active"]


@dataclass(frozen=True)
class WorkerHealth:
    schema_version: int
    mode: WorkerMode
    activity: WorkerActivity
    current_run_id: str | None
    last_progress_at: str
    last_terminal_outcome: str | None
    consecutive_failures: int

    @classmethod
    def from_dict(cls, payload: object) -> WorkerHealth:
        if not isinstance(payload, dict):
            raise RuntimeError("Researcher health root must be an object.")
        try:
            state = cls(**payload)
            _parse_timestamp(state.last_progress_at)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Researcher health state is invalid.") from error
        if state.schema_version != 1:
            raise RuntimeError("Unsupported researcher health schema.")
        if state.mode not in {"paused", "enabled"}:
            raise RuntimeError("Researcher health mode is invalid.")
        if state.activity not in {"idle", "active"}:
            raise RuntimeError("Researcher health activity is invalid.")
        if state.consecutive_failures < 0:
            raise RuntimeError("Researcher failure count is invalid.")
        return state


class HealthStore:
    def __init__(self, path: str | Path, *, now=None) -> None:
        self.path = Path(path)
        self._now = now or (lambda: datetime.now(UTC))

    def initialize(self, *, enabled: bool) -> WorkerHealth:
        previous = self.read() if self.path.exists() else None
        return self._write(
            WorkerHealth(
                schema_version=1,
                mode="enabled" if enabled else "paused",
                activity="idle",
                current_run_id=None,
                last_progress_at=self._timestamp(),
                last_terminal_outcome=(previous.last_terminal_outcome if previous else None),
                consecutive_failures=(previous.consecutive_failures if previous else 0),
            )
        )

    def set_idle(self, *, enabled: bool) -> WorkerHealth:
        state = self.read()
        return self._write(
            replace(
                state,
                mode="enabled" if enabled else "paused",
                activity="idle",
                current_run_id=None,
                last_progress_at=self._timestamp(),
            )
        )

    def start_job(self) -> WorkerHealth:
        state = self.read()
        return self._write(
            replace(
                state,
                activity="active",
                current_run_id=None,
                last_progress_at=self._timestamp(),
            )
        )

    def progress(self, run_id: str | None = None) -> WorkerHealth:
        state = self.read()
        return self._write(
            replace(
                state,
                current_run_id=(run_id[:160] if run_id else state.current_run_id),
                last_progress_at=self._timestamp(),
            )
        )

    def finish_job(self, outcome: str, *, failed: bool) -> WorkerHealth:
        state = self.read()
        return self._write(
            replace(
                state,
                activity="idle",
                current_run_id=None,
                last_progress_at=self._timestamp(),
                last_terminal_outcome=outcome[:200],
                consecutive_failures=(state.consecutive_failures + 1 if failed else 0),
            )
        )

    def read(self) -> WorkerHealth:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Researcher health state is unavailable.") from error
        return WorkerHealth.from_dict(payload)

    def _write(self, state: WorkerHealth) -> WorkerHealth:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.parent / f".{self.path.name}.{uuid4()}.tmp"
        raw = (json.dumps(asdict(state), sort_keys=True, separators=(",", ":")) + "\n").encode()
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return state

    def _timestamp(self) -> str:
        now = self._now()
        if now.tzinfo is None:
            raise ValueError("Health timestamps must be timezone-aware.")
        return now.astimezone(UTC).isoformat()


def check_researcher_health(
    path: str | Path,
    *,
    stale_after_seconds: int,
    now: datetime | None = None,
) -> WorkerHealth:
    if stale_after_seconds < 1:
        raise ValueError("stale_after_seconds must be positive.")
    state = HealthStore(path).read()
    checked_at = now or datetime.now(UTC)
    if state.activity == "active":
        age = (
            checked_at.astimezone(UTC) - _parse_timestamp(state.last_progress_at)
        ).total_seconds()
        if age > stale_after_seconds:
            raise RuntimeError("Researcher active work is stale.")
    return state


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must be timezone-aware.")
    return parsed.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    path = os.getenv("RESEARCHER_HEALTH_PATH", "data/researcher_health.json")
    stale = int(os.getenv("RESEARCHER_HEALTH_STALE_AFTER_SECONDS", "180"))
    check_researcher_health(path, stale_after_seconds=stale)
    print("run4221 researcher health check passed")


if __name__ == "__main__":
    main()
