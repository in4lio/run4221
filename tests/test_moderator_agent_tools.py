import asyncio

from run4221.agent.moderator_tools import (
    ModeratorAgentTools,
    moderator_agent_tool_specs,
)
from run4221.ai.event_extractor import EventDraft
from run4221.db.bootstrap import initialize_database
from run4221.db.repository import (
    EventSuggestionCreate,
    ProposedEventUpdateCreate,
    add_event_suggestion,
    create_proposed_event_update,
    find_event,
    get_event_suggestion,
)


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-agent-tools.sqlite3'}"


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
    initialize_database(url)
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


def test_moderator_agent_apply_suggestion_returns_draft_without_converting(
    tmp_path,
    monkeypatch,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)

    async def fake_extract_event_draft_from_url(source_url: str) -> EventDraft:
        return EventDraft(
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
            confidence=0.93,
            evidence="Fixture evidence.",
        )

    monkeypatch.setattr(
        "run4221.agent.moderator_tools.extract_event_draft_from_url",
        fake_extract_event_draft_from_url,
    )

    result = asyncio.run(tools.apply_suggestion(suggestion.id))

    assert result.ok is True
    assert result.data["suggestion"]["id"] == suggestion.id
    assert result.data["draft"]["public_id"] == "agent-test.42"
    assert result.data["next_step"] == (
        "create_event with source_suggestion_id after moderator review"
    )
    assert get_event_suggestion(suggestion.id, database_url=url).status == "pending"


def test_moderator_agent_create_event_can_convert_suggestion(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    tools = ModeratorAgentTools(database_url=url)
    suggestion = add_event_suggestion(suggestion_payload(), database_url=url)

    result = tools.create_event(event_fields(), source_suggestion_id=suggestion.id)

    assert result.ok is True
    assert get_event_suggestion(suggestion.id, database_url=url).status == "converted"
