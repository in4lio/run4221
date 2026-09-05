from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

import run4221.researcher
from run4221.db.prompts import RESEARCH_AGENT_PROMPT, upsert_active_prompt_version
from run4221.researcher.config import (
    ResearcherSettings,
    load_researcher_prompt,
)
from run4221.researcher.schemas import (
    ArtifactReference,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
    ResearchRunStatus,
)


def database_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-test.sqlite3'}"


def test_researcher_settings_have_safe_independent_defaults() -> None:
    settings = ResearcherSettings(_env_file=None, openai_api_key="test-api-key")

    assert settings.model == "gpt-5.6-luna"
    assert settings.schedule_enabled is False
    assert settings.rendering_enabled is False
    assert settings.max_output_tokens_per_job == 2_000
    assert settings.budget == ResearchBudget()
    assert "telegram_bot_token" not in type(settings).model_fields


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"openai_api_key": None}, "openai_api_key"),
        ({"interval_seconds": 0}, "interval_seconds"),
        ({"max_agent_turns_per_job": 0}, "max_agent_turns_per_job"),
        ({"max_pending_suggestions": 21}, "max_pending_suggestions"),
    ],
)
def test_researcher_settings_reject_invalid_configuration_without_revealing_key(
    overrides: dict[str, object],
    match: str,
) -> None:
    secret = "sk-never-echo-this-research-key"
    values: dict[str, object] = {"openai_api_key": secret}
    values.update(overrides)

    with pytest.raises(ValidationError, match=match) as error:
        ResearcherSettings(
            _env_file=None,
            **values,
        )

    assert secret not in str(error.value)


def test_renamed_researcher_enabled_env_flag_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The scheduling flag was renamed; a stale RESEARCHER_ENABLED in the
    # environment must fail startup instead of being silently ignored.
    monkeypatch.setenv("RESEARCHER_ENABLED", "true")

    with pytest.raises(ValidationError, match="RESEARCHER_SCHEDULE_ENABLED"):
        ResearcherSettings(_env_file=None, openai_api_key="test-api-key")

    monkeypatch.delenv("RESEARCHER_ENABLED")
    settings = ResearcherSettings(_env_file=None, openai_api_key="test-api-key")
    assert settings.schedule_enabled is False


def test_profile_wall_cap_is_shorter_than_refresh_and_env_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ResearcherSettings(_env_file=None, openai_api_key="test-api-key")

    assert settings.max_wall_time_seconds_per_profile_job == 60
    assert (
        settings.max_wall_time_seconds_per_profile_job
        < settings.max_wall_time_seconds_per_job
    )
    assert settings.budget.max_wall_time_seconds_per_profile_job == 60

    monkeypatch.setenv("RESEARCHER_MAX_WALL_TIME_SECONDS_PER_PROFILE_JOB", "45")
    configured = ResearcherSettings(_env_file=None, openai_api_key="test-api-key")

    assert configured.budget.max_wall_time_seconds_per_profile_job == 45


def test_researcher_prompt_loads_from_file(tmp_path: Path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_path = prompts_dir / "research_agent.instructions.txt"
    prompt_path.write_text("Use captured evidence only.", encoding="utf-8")
    settings = ResearcherSettings(
        _env_file=None,
        openai_api_key="test-api-key",
        prompt_source="file",
        prompt_dir=str(prompts_dir),
    )

    prompt = load_researcher_prompt(settings)

    assert prompt.prompt_key == RESEARCH_AGENT_PROMPT
    assert prompt.source == "file"
    assert prompt.content == "Use captured evidence only."


def test_researcher_prompt_loads_from_database(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    upsert_active_prompt_version(
        RESEARCH_AGENT_PROMPT,
        "Assess captured evidence only.",
        database_url=url,
    )
    settings = ResearcherSettings(
        _env_file=None,
        openai_api_key="test-api-key",
        database_url=url,
        prompt_source="db",
    )

    prompt = load_researcher_prompt(settings)

    assert prompt.prompt_key == RESEARCH_AGENT_PROMPT
    assert prompt.source == "db"
    assert prompt.content == "Assess captured evidence only."


def test_researcher_prompt_requires_active_database_version(tmp_path: Path) -> None:
    settings = ResearcherSettings(
        _env_file=None,
        openai_api_key="test-api-key",
        database_url=database_url(tmp_path),
        prompt_source="db",
    )

    with pytest.raises(RuntimeError, match="No usable DB prompt"):
        load_researcher_prompt(settings)


def test_researcher_contract_rejects_unknown_actions_and_fields() -> None:
    with pytest.raises(ValidationError):
        ResearchDecision.model_validate(
            {
                "action": "apply_update",
                "summary": "Apply directly.",
                "evidence": [],
            }
        )

    with pytest.raises(ValidationError):
        ResearchDecision.model_validate(
            {
                "action": "no_change",
                "summary": "No stable source.",
                "confidence": 0.2,
                "evidence": [],
                "approve": True,
            }
        )


def test_researcher_decision_rejects_mixed_queue_payloads() -> None:
    candidate = {
        "source_url": "https://example.com/marathon",
        "title": "Example Marathon",
        "snippet": "Official event page.",
    }
    proposed_fields = {"registration_status": "open"}

    with pytest.raises(ValidationError, match="cannot include a queue payload"):
        ResearchDecision(
            action="no_change",
            summary="A non-persisting action cannot carry a candidate.",
            candidate=candidate,
        )

    with pytest.raises(ValidationError, match="cannot include a candidate"):
        ResearchDecision(
            action="propose_update",
            summary="Propose a captured registration change.",
            candidate=candidate,
            proposed_fields=proposed_fields,
        )


@pytest.mark.parametrize(
    "schema_payload",
    [
        {
            "schema": ResearchCandidate,
            "payload": {
                "source_url": "javascript:alert(1)",
                "title": "Example Marathon",
                "snippet": "Candidate.",
            },
        },
        {
            "schema": ArtifactReference,
            "payload": {
                "run_id": "019c6e27-e55b-73d1-87d8-4e01f1f75043",
                "artifact_name": "evidence.json",
                "source_url": "https://user:password@example.com/race",
                "content_hash": "a" * 64,
            },
        },
    ],
)
def test_researcher_contract_rejects_unsafe_urls(schema_payload: dict[str, object]) -> None:
    schema = schema_payload["schema"]

    with pytest.raises(ValidationError):
        schema.model_validate(schema_payload["payload"])


def test_researcher_contract_rejects_oversized_evidence() -> None:
    with pytest.raises(ValidationError):
        ResearchCandidate(
            source_url="https://example.com/marathon",
            title="Example Marathon",
            snippet="x" * 1_001,
        )


def test_researcher_run_status_uses_closed_status_and_outcome_enums() -> None:
    status = ResearchRunStatus(status="succeeded", outcome="no_change")

    assert status.status.value == "succeeded"
    assert status.outcome.value == "no_change"
    with pytest.raises(ValidationError):
        ResearchRunStatus(status="approved", outcome="published")


def test_researcher_package_has_no_privileged_imports() -> None:
    package_dir = Path(run4221.researcher.__file__).parent
    forbidden_modules = {
        "run4221.agent",
        "run4221.bot",
        "run4221.db.repository",
    }
    violations: list[str] = []

    for path in package_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            for module in modules:
                if any(
                    module == forbidden or module.startswith(f"{forbidden}.")
                    for forbidden in forbidden_modules
                ):
                    violations.append(f"{path.name}: {module}")

    assert violations == []
