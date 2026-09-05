import asyncio
import inspect

from pydantic import ValidationError

from run4221.agent.moderator_tools import (
    ModeratorAgentTools,
    moderator_agent_tool_specs,
)
from run4221.db.bootstrap import initialize_database
from run4221.db.repository import (
    EventSuggestionCreate,
    ProposedEventUpdateCreate,
    add_event_suggestion,
    create_proposed_event_update,
    find_event,
    get_event_suggestion,
)
from run4221.db.seed import seed_initial_data
from run4221.db.session import session_scope
from run4221.researcher.engine import EngineConfigError, SourceNotFoundError
from run4221.researcher.schemas import (
    ArtifactReference,
    EventProfileDraft,
    ResearchRunStatus,
)
from run4221.researcher.service import ProfileJobResult, ResearchJobResult
from tests.seed_fixtures import sample_seed_events

RUN_ID = "2d1aa0bb-13c1-4f1b-b81f-a7f6b83b62dc"


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-agent-tools.sqlite3'}"


def artifact_reference() -> ArtifactReference:
    return ArtifactReference(
        run_id=RUN_ID,
        artifact_name="terminal.json",
        source_url="https://example.com/agent-marathon",
        content_hash="a" * 64,
    )


def profile_job_result(
    draft: EventProfileDraft | None,
    *,
    status: str = "succeeded",
    outcome: str = "profile_completed",
    detail: str | None = None,
) -> ProfileJobResult:
    return ProfileJobResult(
        run_id=RUN_ID,
        status=ResearchRunStatus(status=status, outcome=outcome, detail=detail),
        terminal_reference=artifact_reference(),
        draft=draft,
    )


def refresh_job_result(
    *,
    status: str = "succeeded",
    outcome: str = "no_change",
    detail: str | None = None,
    queue_reference: str | None = None,
    conflicting_update_id: int | None = None,
) -> ResearchJobResult:
    return ResearchJobResult(
        run_id=RUN_ID,
        status=ResearchRunStatus(status=status, outcome=outcome, detail=detail),
        terminal_reference=artifact_reference(),
        queue_reference=queue_reference,
        conflicting_update_id=conflicting_update_id,
    )


def sample_draft(source_url: str = "https://example.com/agent-marathon") -> EventProfileDraft:
    return EventProfileDraft(
        source_url=source_url,
        name="Agent Test Marathon",
        public_id="agent-test.42",
        city="Agent City",
        country="Germany",
        timezone="Europe/Berlin",
        event_date="2027-05-01",
        distances=("marathon",),
        regions=("de", "eu"),
        official_url=source_url,
        registration_url=None,
        summary="Fixture evidence.",
        confidence=0.93,
    )


class FakeEngine:
    def __init__(
        self,
        *,
        profile_result: ProfileJobResult | None = None,
        refresh_result: ResearchJobResult | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.profile_result = profile_result
        self.refresh_result = refresh_result
        self.refresh_error = refresh_error
        self.profile_calls: list[str] = []
        self.refresh_calls: list[str] = []

    async def profile(self, url: str) -> ProfileJobResult:
        self.profile_calls.append(url)
        assert self.profile_result is not None
        return self.profile_result

    async def refresh_source(self, event_id: str) -> ResearchJobResult:
        self.refresh_calls.append(event_id)
        if self.refresh_error is not None:
            raise self.refresh_error
        assert self.refresh_result is not None
        return self.refresh_result


def initialize_sample_database(url: str) -> None:
    initialize_database(url, seed_initial_events=False)
    with session_scope(url) as session:
        seed_initial_data(session, sample_seed_events())


def suggestion_payload() -> EventSuggestionCreate:
    return EventSuggestionCreate(
        event_name="Agent Test Marathon",
        url="https://example.com/agent-marathon",
        event_date="2027-05-01",
        location="Agent City, Germany",
        region_tags=("de", "eu"),
        distances=("marathon",),
        note="Please track it.",
        submitter_user_id="42",
        submitter_username="runner",
        submitter_display_name="Runner",
    )


def event_fields(public_id: str = "agent-test.42") -> dict:
    return {
        "public_id": public_id,
        "name": "Agent Test Marathon",
        "city": "Agent City",
        "country": "Germany",
        "timezone": "Europe/Berlin",
        "distances": ["marathon"],
        "regions": ["de", "eu"],
        "official_url": "https://example.com/agent-marathon",
        "event_date": "2027-05-01",
    }


def test_moderator_agent_tool_specs_include_restricted_delete() -> None:
    specs = {spec["name"]: spec for spec in moderator_agent_tool_specs()}

    assert "apply_update" in specs
    assert "apply_suggestion" in specs
    assert specs["delete_event"]["destructive"] is True
    assert specs["delete_event"]["requires_confirmation_code"] is True


def test_moderator_agent_tools_create_edit_archive_restore_and_delete(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url, delete_confirmation_code="secret")

    created = tools.create_event(event_fields())
    assert created.ok is True
    assert created.data["public_id"] == "agent-test.42"

    edited = tools.edit_event("agent-test.42", {"city": "New Agent City"})
    assert edited.ok is True
    assert edited.data["city"] == "New Agent City"

    archived = tools.archive_event("agent-test.42")
    assert archived.ok is True
    assert tools.list_archive().data[0]["event"]["public_id"] == "agent-test.42"

    restored = tools.restore_event("agent-test.42")
    assert restored.ok is True

    denied = tools.delete_event(
        "agent-test.42",
        confirmation_code="wrong",
        confirm_event_id="agent-test.42",
    )
    assert denied.ok is False
    assert denied.restricted is True
    assert find_event("agent-test.42", url) is not None

    mismatch = tools.delete_event(
        "agent-test.42",
        confirmation_code="secret",
        confirm_event_id="other.42",
    )
    assert mismatch.ok is False
    assert mismatch.restricted is True

    deleted = tools.delete_event(
        "agent-test.42",
        confirmation_code="secret",
        confirm_event_id="agent-test.42",
    )
    assert deleted.ok is True
    assert find_event("agent-test.42", url) is None


def test_moderator_agent_tools_delete_is_disabled_without_guard_code(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)

    result = tools.delete_event(
        "berlin.42",
        confirmation_code="secret",
        confirm_event_id="berlin.42",
    )

    assert result.ok is False
    assert result.restricted is True
    assert "disabled" in (result.error or "")


def test_moderator_agent_tools_update_and_suggestion_queues(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_sample_database(url)
    tools = ModeratorAgentTools(database_url=url)
    event_id = find_event("berlin.42", url).id

    update = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event_id,
            update_type="registration_window",
            current_fields={"registration_status": "unknown"},
            proposed_fields={"registration_status": "open"},
            evidence=("Agent test evidence.",),
            confidence=0.9,
            change_summary="Registration status changed.",
        ),
        database_url=url,
    )
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)

    todo = tools.todo()
    assert todo.data == {"pending_updates": 1, "pending_suggestions": 1}
    assert tools.next_update().data["id"] == update.id
    assert tools.show_update(update.id).data["handle"] == f"#{update.id}"
    assert tools.next_suggestion().data["id"] == suggestion.id
    assert tools.show_suggestion(suggestion.id).data["handle"] == f"#{suggestion.id}"

    applied = tools.apply_update(update.id)
    assert applied.ok is True
    assert applied.data["event"]["registration_status"] == "open"

    rejected = tools.reject_suggestion(suggestion.id)
    assert rejected.ok is True
    assert rejected.data["status"] == "removed"


def test_moderator_agent_search_uses_repository_contract(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_sample_database(url)
    tools = ModeratorAgentTools(database_url=url)

    result = tools.search_events("berlin", limit=1)

    assert result.ok is True
    assert len(result.data) == 1
    assert result.data[0]["city"] == "Berlin"


def test_moderator_agent_apply_suggestion_returns_draft_without_converting(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    engine = FakeEngine(
        profile_result=profile_job_result(
            sample_draft(),
            detail="Fixture evidence.",
        )
    )
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)

    result = asyncio.run(tools.apply_suggestion(suggestion.id))

    assert result.ok is True
    assert engine.profile_calls == [suggestion.url]
    assert result.data["suggestion"]["id"] == suggestion.id
    assert result.data["draft"]["public_id"] == "agent-test.42"
    assert result.data["run_id"] == RUN_ID
    assert result.data["next_step"] == (
        "create_event with source_suggestion_id after moderator review"
    )
    assert get_event_suggestion(suggestion.id, database_url=url).status == "pending"


def test_moderator_agent_discover_event_profile_returns_typed_run(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    engine = FakeEngine(profile_result=profile_job_result(sample_draft()))
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.discover_event_profile("https://example.com/agent-marathon"))

    assert result.ok is True
    assert result.data["run_id"] == RUN_ID
    assert result.data["status"] == "succeeded"
    assert result.data["outcome"] == "profile_completed"
    assert result.data["draft"]["name"] == "Agent Test Marathon"
    assert result.data["draft"]["registration_url_candidates"] == []


def test_moderator_agent_discover_event_profile_fails_closed_without_draft(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    engine = FakeEngine(
        profile_result=profile_job_result(
            None,
            status="failed",
            outcome="inconclusive",
            detail="OpenAI authentication failed.",
        )
    )
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.discover_event_profile("https://example.com/agent-marathon"))

    assert result.ok is False
    assert "configuration or usage limits" in (result.error or "")
    assert "OpenAI authentication failed." in (result.error or "")
    assert RUN_ID in (result.error or "")


def test_moderator_agent_update_event_uses_engine_refresh(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools_for_create = ModeratorAgentTools(database_url=url)
    created = tools_for_create.create_event(event_fields())
    assert created.ok is True

    engine = FakeEngine(
        refresh_result=refresh_job_result(
            status="succeeded",
            outcome="proposal_created",
            detail="Registration opens on 2027-01-01.",
            queue_reference="proposed_event_update:4",
        )
    )
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.update_event("agent-test.42"))

    assert result.ok is True
    assert engine.refresh_calls == ["agent-test.42"]
    assert result.data["queue_reference"] == "proposed_event_update:4"
    assert result.data["conflicting_update_id"] is None
    assert result.data["message"] == "Created pending update #4 for moderator review."
    assert "auto_confirm" not in inspect.signature(ModeratorAgentTools.update_event).parameters


def test_moderator_agent_update_event_reports_conflicting_pending_update(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools_for_create = ModeratorAgentTools(database_url=url)
    assert tools_for_create.create_event(event_fields()).ok is True

    engine = FakeEngine(
        refresh_result=refresh_job_result(
            status="skipped",
            outcome="inconclusive",
            detail="A conflicting pending proposal already exists.",
            conflicting_update_id=11,
        )
    )
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.update_event("agent-test.42"))

    assert result.ok is True
    assert result.data["conflicting_update_id"] == 11
    assert result.data["message"] == "Update #11 is already pending for this event."


def test_moderator_agent_update_event_without_source_fails_with_named_error(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools_for_create = ModeratorAgentTools(database_url=url)
    assert tools_for_create.create_event(event_fields()).ok is True

    engine = FakeEngine(
        refresh_error=SourceNotFoundError(
            "No active research source for event: agent-test.42"
        )
    )
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.update_event("agent-test.42"))

    assert result.ok is False
    assert result.error == "No active research source for event: agent-test.42"


def test_moderator_agent_update_event_validation_error_is_a_generic_failure(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools_for_create = ModeratorAgentTools(database_url=url)
    assert tools_for_create.create_event(event_fields()).ok is True

    try:
        EventProfileDraft.model_validate({})
        raise AssertionError("EventProfileDraft.model_validate({}) must fail")
    except ValidationError as error:
        validation_error = error
    engine = FakeEngine(refresh_error=validation_error)
    tools = ModeratorAgentTools(database_url=url, engine=engine)  # type: ignore[arg-type]

    result = asyncio.run(tools.update_event("agent-test.42"))

    assert result.ok is False
    assert (result.error or "").startswith("Could not update event:")
    assert "No active research source" not in (result.error or "")


def test_moderator_agent_tools_fail_closed_when_engine_is_not_configured(
    tmp_path,
    monkeypatch,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools_for_create = ModeratorAgentTools(database_url=url)
    assert tools_for_create.create_event(event_fields()).ok is True

    def broken_build_engine():
        raise EngineConfigError("Researcher settings are invalid or incomplete.")

    monkeypatch.setattr("run4221.researcher.engine.build_engine", broken_build_engine)
    tools = ModeratorAgentTools(database_url=url)

    discover = asyncio.run(tools.discover_event_profile("https://example.com/x"))
    update = asyncio.run(tools.update_event("agent-test.42"))

    for result in (discover, update):
        assert result.ok is False
        assert (result.error or "").startswith("Researcher engine is not configured:")


def test_moderator_agent_create_event_can_convert_suggestion(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)

    result = tools.create_event(event_fields(), source_suggestion_id=suggestion.id)

    assert result.ok is True
    assert get_event_suggestion(suggestion.id, database_url=url).status == "converted"


def test_moderator_agent_create_event_requires_pending_suggestion(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)

    result = tools.create_event(event_fields(), source_suggestion_id=999)

    assert result.ok is False
    assert find_event("agent-test.42", url) is None


def test_moderator_agent_create_event_rolls_back_suggestion_on_event_failure(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)
    first = tools.create_event(event_fields())

    result = tools.create_event(event_fields(), source_suggestion_id=suggestion.id)

    assert first.ok is True
    assert result.ok is False
    assert get_event_suggestion(suggestion.id, database_url=url).status == "pending"
