from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.health import check_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_check_validates_initialized_database(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'health.sqlite3'}"
    initialize_database(database_url, seed_initial_events=False)

    check_database(database_url)


def test_health_check_rejects_missing_database_without_creating_it(tmp_path) -> None:
    database_path = tmp_path / "missing.sqlite3"

    with pytest.raises(RuntimeError, match="has not been initialized"):
        check_database(f"sqlite:///{database_path}")

    assert not database_path.exists()


def test_health_check_rejects_database_without_application_schema(tmp_path) -> None:
    database_path = tmp_path / "empty.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")

    with pytest.raises(RuntimeError, match="missing the events table"):
        check_database(f"sqlite:///{database_path}")


def test_deploy_runs_quality_gate_health_check_and_rollback() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "group: production-mutation" in workflow
    assert "uv run ruff check" in workflow
    assert "uv run pytest" in workflow
    assert "--wait --wait-timeout 90" in workflow
    assert "docker compose run --rm --build bot uv run python -m run4221.health" in workflow
    assert 'if [[ "$service_replaced" == "true" ]]' in workflow
    assert "docker compose exec -T bot uv run python -m run4221.health" in workflow
    assert "trap rollback ERR" in workflow


def test_bootstrap_shares_mutation_lock_and_treats_confirmation_as_data() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/bootstrap-production.yml").read_text(
        encoding="utf-8"
    )

    assert "group: production-mutation" in workflow
    assert "--wait --wait-timeout 90" in workflow
    assert "CONFIRM_RESET: ${{ inputs.confirm_reset }}" in workflow
    assert 'if [[ "$CONFIRM_RESET" != "RESET_SUGGESTION_BOOTSTRAP" ]]' in workflow
    assert 'if [[ "${{ inputs.confirm_reset }}"' not in workflow
