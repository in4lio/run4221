from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.health import check_database, check_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_project_file(path: str) -> str:
    return (PROJECT_ROOT / path).read_text(encoding="utf-8")


def compose_service(compose: str, service: str, next_service: str | None = None) -> str:
    start = compose.index(f"  {service}:\n")
    if next_service is None:
        return compose[start:]
    end = compose.index(f"  {next_service}:\n", start)
    return compose[start:end]


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


def test_runtime_health_rejects_unsafe_bot_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test:token")
    monkeypatch.setenv("TELEGRAM_MODERATOR_ACCOUNTS", "@mutable-username")

    with pytest.raises(RuntimeError, match="numeric user IDs"):
        check_runtime()


def test_deploy_runs_quality_gate_health_check_and_rollback() -> None:
    workflow = read_project_file(".github/workflows/deploy.yml")

    assert "group: production-mutation" in workflow
    assert "uv run ruff check" in workflow
    assert "uv run pytest" in workflow
    assert "--wait --wait-timeout 90" in workflow
    assert (
        "docker compose run --rm -T --interactive=false --build bot "
        "uv run python -m run4221.health"
    ) in workflow
    assert workflow.count("docker compose run") == workflow.count(
        "docker compose run --rm -T --interactive=false"
    )
    assert 'if [[ "$service_replaced" == "true" ]]' in workflow
    assert workflow.count("docker compose exec") == workflow.count(
        "docker compose exec -T --interactive=false"
    )
    assert (
        "docker compose exec -T --interactive=false bot "
        "uv run python -m run4221.health"
    ) in workflow
    assert "trap rollback ERR" in workflow


def test_compose_keeps_bot_and_researcher_static_and_least_privileged() -> None:
    compose = read_project_file("compose.yaml")
    dockerfile = read_project_file("Dockerfile")
    bot = compose_service(compose, "bot", "researcher")
    researcher = compose_service(compose, "researcher")

    assert "target: bot" in bot
    assert "target: researcher" in researcher
    assert "AS bot" in dockerfile
    assert "AS researcher" in dockerfile
    assert "--extra researcher" in dockerfile
    assert "playwright" not in dockerfile.casefold()
    assert "chromium" not in dockerfile.casefold()

    assert "./data:/app/data" in bot
    assert "./data:/app/data" in researcher
    assert "private/runtime/bot.env" in bot
    assert "private/runtime/researcher.env" in researcher
    assert "target: /app/.env" in bot
    assert "target: /app/.env" in researcher
    assert "TELEGRAM" not in researcher
    assert "replicas: 1" in researcher
    assert 'CMD ["uv", "run", "run4221-researcher", "--loop"]' in dockerfile
    assert "command:" not in researcher
    assert "run4221.researcher.health" in researcher
    assert "condition: service_healthy" in researcher


def test_deploy_preflights_researcher_before_replacement_and_restores_topology() -> None:
    workflow = read_project_file(".github/workflows/deploy.yml")

    assert "uv sync --frozen --extra dev --extra researcher" in workflow
    assert 'previous_sha="$(git rev-parse HEAD)"' in workflow
    assert 'previous_services="$(docker compose config --services' in workflow
    assert 'previous_bot_image="$(docker inspect' in workflow
    assert 'previous_researcher_image="$(docker inspect' in workflow
    assert 'previous_bot_rollback_tag="run4221-bot:rollback-$previous_sha"' in workflow
    assert (
        'previous_researcher_rollback_tag="run4221-researcher:rollback-$previous_sha"'
        in workflow
    )
    assert workflow.index('docker image tag "$previous_bot_image"') < workflow.index(
        "docker compose run"
    )
    assert 'docker image tag "$previous_bot_rollback_tag" run4221-bot:latest' in workflow
    assert "run4221-researcher --check-config" in workflow
    assert workflow.index("run4221-researcher --check-config") < workflow.index(
        "service_replaced=true"
    )
    assert "docker image tag" in workflow
    assert "--remove-orphans" in workflow
    assert (
        "docker compose exec -T --interactive=false researcher "
        "uv run python -m run4221.researcher.health"
        in workflow
    )
    assert "down -v" not in workflow


def test_compose_validation_does_not_expand_runtime_secrets() -> None:
    compose = read_project_file("compose.yaml")

    assert "env_file:" not in compose
    assert "TELEGRAM_BOT_TOKEN:" not in compose
    assert "OPENAI_API_KEY:" not in compose
    assert "BOT_ENV_FILE" in compose
    assert "RESEARCHER_ENV_FILE" in compose


def test_bootstrap_shares_mutation_lock_and_treats_confirmation_as_data() -> None:
    workflow = read_project_file(".github/workflows/bootstrap-production.yml")

    assert "group: production-mutation" in workflow
    assert workflow.count("docker compose run") == workflow.count(
        "docker compose run --rm -T --interactive=false"
    )
    assert "--wait --wait-timeout 90" in workflow
    assert "CONFIRM_RESET: ${{ inputs.confirm_reset }}" in workflow
    assert 'if [[ "$CONFIRM_RESET" != "RESET_SUGGESTION_BOOTSTRAP" ]]' in workflow
    assert 'if [[ "${{ inputs.confirm_reset }}"' not in workflow


def test_bootstrap_restores_owner_only_private_file_permissions_after_sync() -> None:
    workflow = read_project_file(".github/workflows/bootstrap-production.yml")
    sync_step = workflow[
        workflow.index("      - name: Sync private data to VPS") :
        workflow.index("      - name: Seed prompts")
    ]

    assert "chmod 700 '$remote_private_dir'" in sync_step
    assert "'$remote_private_dir/data' '$remote_private_dir/prompts'" in sync_step
    assert "-type f -exec chmod 600 {} +" in sync_step


def test_bootstrap_stops_both_writers_and_checkpoints_wal_before_reset() -> None:
    workflow = read_project_file(".github/workflows/bootstrap-production.yml")
    reset_step = workflow[workflow.index("      - name: Reset database and seed suggestions") :]

    assert "docker compose stop bot researcher" in workflow
    assert "PRAGMA wal_checkpoint(TRUNCATE)" in reset_step
    assert "PRAGMA integrity_check" in reset_step
    assert ".pre-reset.sqlite3" in reset_step
    assert "run4221.sqlite3-wal" in reset_step
    assert "run4221.sqlite3-shm" in reset_step
    assert reset_step.index("docker compose stop bot researcher") < reset_step.index(
        "--reset-sqlite"
    )
    assert "docker compose up -d --build --remove-orphans --wait --wait-timeout 90" in reset_step


def test_example_keeps_researcher_paused_with_a_dedicated_credential() -> None:
    example = read_project_file(".env.example")

    assert "BOT_ENV_FILE=./private/runtime/bot.env" in example
    assert "RESEARCHER_ENV_FILE=./private/runtime/researcher.env" in example
    assert "RESEARCHER_OPENAI_API_KEY=" in example
    assert "RESEARCHER_ENABLED=false" in example
    assert "RESEARCHER_DISCOVERY_ENABLED=false" in example
    assert "RESEARCHER_RENDERING_ENABLED=false" in example
