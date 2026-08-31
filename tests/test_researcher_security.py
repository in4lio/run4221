from __future__ import annotations

import ast
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.exceptions import MaxTurnsExceeded

from run4221.researcher import agent as researcher_agent
from run4221.researcher.agent import (
    AgentRunState,
    AssessmentRequest,
    CapturedSnapshotEvidence,
    ResearchAgentJob,
    ScoutOutput,
    ScoutRequest,
)
from run4221.researcher.schemas import ArtifactReference, ResearchBudget


class RaisingRunner:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    async def run(self, starting_agent, input, **kwargs):
        self.calls += 1
        raise self.error


class MalformedRunner:
    def __init__(self, output: object) -> None:
        self.output = output
        self.calls = 0

    async def run(self, starting_agent, input, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            final_output=self.output,
            raw_responses=[
                SimpleNamespace(
                    response_id="resp_malformed",
                    usage=SimpleNamespace(
                        input_tokens=1,
                        output_tokens=1,
                        total_tokens=2,
                    ),
                    output=[],
                )
            ],
        )


class FakeProviderError(Exception):
    def __init__(
        self,
        *,
        status_code: int | None = None,
        code: str | None = None,
        secret: str,
    ) -> None:
        super().__init__(f"provider failure contained {secret}")
        self.status_code = status_code
        self.code = code


def reference() -> ArtifactReference:
    return ArtifactReference(
        run_id="019c6e27-e55b-73d1-87d8-4e01f1f75043",
        artifact_name="page_snapshot-test.json",
        source_url="https://example.com/marathon",
        content_hash="a" * 64,
    )


def assessment_request() -> AssessmentRequest:
    return AssessmentRequest(
        mode="refresh",
        evidence=(
            CapturedSnapshotEvidence(
                reference=reference(),
                final_url="https://example.com/marathon",
                fetched_at=datetime(2026, 8, 31, tzinfo=UTC),
                normalized_text="Registration is open.",
                text_hash="b" * 64,
            ),
        ),
    )


@pytest.mark.parametrize(
    ("error", "expected_state", "expected_code"),
    [
        (
            FakeProviderError(status_code=401, secret="sk-auth-secret"),
            AgentRunState.FAILED,
            "authentication",
        ),
        (
            FakeProviderError(
                status_code=429,
                code="insufficient_quota",
                secret="sk-quota-secret",
            ),
            AgentRunState.FAILED,
            "quota",
        ),
        (
            FakeProviderError(status_code=429, secret="sk-rate-secret"),
            AgentRunState.FAILED,
            "rate_limit",
        ),
        (
            TimeoutError("timeout contained sk-timeout-secret"),
            AgentRunState.CAPPED,
            "timeout",
        ),
        (
            MaxTurnsExceeded("max turns contained sk-turn-secret"),
            AgentRunState.CAPPED,
            "max_turns",
        ),
    ],
)
def test_provider_errors_are_classified_without_exception_text(
    error: BaseException,
    expected_state: AgentRunState,
    expected_code: str,
) -> None:
    runner = RaisingRunner(error)
    job = ResearchAgentJob(
        instructions="Assess captured evidence.",
        prompt_reference="research_agent:v1",
        budget=ResearchBudget(max_wall_time_seconds_per_job=10),
        runner=runner,
    )

    result = asyncio.run(job.assess(assessment_request()))

    assert result.state is expected_state
    assert result.error_code == expected_code
    assert "sk-" not in (result.detail or "")
    assert "contained" not in (result.detail or "")


def test_malformed_structured_output_is_inconclusive_and_secret_safe() -> None:
    runner = MalformedRunner("malformed output with sk-malformed-secret")
    job = ResearchAgentJob(
        instructions="Assess captured evidence.",
        prompt_reference="research_agent:v1",
        budget=ResearchBudget(max_wall_time_seconds_per_job=10),
        runner=runner,
    )

    result = asyncio.run(job.assess(assessment_request()))

    assert result.state is AgentRunState.INCONCLUSIVE
    assert result.error_code == "malformed_output"
    assert "sk-malformed-secret" not in (result.detail or "")


def test_assessor_rejects_non_snapshot_input_before_runner_call() -> None:
    runner = MalformedRunner("unused")
    job = ResearchAgentJob(
        instructions="Assess captured evidence.",
        prompt_reference="research_agent:v1",
        budget=ResearchBudget(max_wall_time_seconds_per_job=10),
        runner=runner,
    )

    with pytest.raises(TypeError, match="AssessmentRequest"):
        asyncio.run(job.assess({"raw_page": "untrusted"}))

    assert runner.calls == 0


def test_unmetered_response_exhausts_output_budget_conservatively() -> None:
    runner = MalformedRunner(
        ScoutOutput(candidates=())
    )
    runner.output = ScoutOutput(candidates=())

    async def unmetered_run(starting_agent, input, **kwargs):
        runner.calls += 1
        return SimpleNamespace(
            final_output=runner.output,
            raw_responses=[
                SimpleNamespace(response_id="resp_unmetered", usage=None, output=[])
            ],
        )

    runner.run = unmetered_run
    job = ResearchAgentJob(
        instructions="Research safely.",
        prompt_reference="research_agent:v1",
        budget=ResearchBudget(
            max_web_searches_per_job=1,
            max_wall_time_seconds_per_job=10,
        ),
        runner=runner,
    )

    first = asyncio.run(job.scout(ScoutRequest(mode="discovery", query="marathon")))
    second = asyncio.run(job.assess(assessment_request()))

    assert first.state is AgentRunState.SUCCEEDED
    assert second.state is AgentRunState.CAPPED
    assert second.error_code == "output_token_budget"
    assert runner.calls == 1


def test_agent_adapter_has_no_privileged_imports() -> None:
    path = Path(researcher_agent.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_modules = {
        "run4221.agent",
        "run4221.bot",
        "run4221.db",
        "run4221.ai.registration_window",
        "run4221.events",
    }
    violations: list[str] = []

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
                violations.append(module)

    assert violations == []
