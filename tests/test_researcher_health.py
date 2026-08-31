from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from run4221.researcher.health import HealthStore, check_researcher_health

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def test_idle_states_are_healthy_and_failed_jobs_remain_visible(tmp_path: Path) -> None:
    path = tmp_path / "researcher-health.json"
    store = HealthStore(path, now=lambda: NOW)

    store.initialize(enabled=False)
    assert store.read().mode == "paused"
    check_researcher_health(path, now=NOW, stale_after_seconds=60)

    store.set_idle(enabled=True)
    active = store.start_job()
    assert active.current_run_id is None
    store.finish_job("failed:inconclusive", failed=True)

    state = store.read()
    assert state.mode == "enabled"
    assert state.activity == "idle"
    assert state.last_terminal_outcome == "failed:inconclusive"
    assert state.consecutive_failures == 1
    check_researcher_health(path, now=NOW + timedelta(hours=1), stale_after_seconds=60)


def test_stale_active_work_is_unhealthy_but_progress_recovers(tmp_path: Path) -> None:
    path = tmp_path / "researcher-health.json"
    store = HealthStore(path, now=lambda: NOW)
    store.initialize(enabled=True)
    store.start_job()

    with pytest.raises(RuntimeError, match="stale"):
        check_researcher_health(
            path,
            now=NOW + timedelta(seconds=61),
            stale_after_seconds=60,
        )

    progressed = HealthStore(path, now=lambda: NOW + timedelta(seconds=50))
    progressed.progress("run-123")
    check_researcher_health(
        path,
        now=NOW + timedelta(seconds=100),
        stale_after_seconds=60,
    )
    assert progressed.read().current_run_id == "run-123"


def test_health_replacement_is_atomic_and_success_resets_failures(tmp_path: Path) -> None:
    path = tmp_path / "researcher-health.json"
    store = HealthStore(path, now=lambda: NOW)
    store.initialize(enabled=True)
    store.start_job()
    store.finish_job("failed:inconclusive", failed=True)
    store.start_job()
    store.finish_job("succeeded:no_change", failed=False)

    assert store.read().consecutive_failures == 0
    assert list(tmp_path.glob(".*.tmp")) == []
