from __future__ import annotations

from pathlib import Path

from run4221.health import check_database

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_health_check_opens_configured_database(tmp_path) -> None:
    check_database(f"sqlite:///{tmp_path / 'health.sqlite3'}")


def test_deploy_runs_quality_gate_health_check_and_rollback() -> None:
    workflow = (PROJECT_ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "group: production-mutation" in workflow
    assert "uv run ruff check" in workflow
    assert "uv run pytest" in workflow
    assert "--wait --wait-timeout 90" in workflow
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
