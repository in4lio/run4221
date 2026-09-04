from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from run4221.bot.formatting import research_outcome_headline
from run4221.db.repository import (
    EventCreate,
    EventUpdate,
    EventWriteError,
    add_event_from_suggestion,
    approve_proposed_event_update,
    count_event_suggestions,
    count_proposed_event_updates,
    find_event,
    get_event_suggestion,
    get_proposed_event_update,
    list_archived_events,
    list_event_suggestions,
    list_events,
    list_events_by_tag,
    list_open_events,
    list_proposed_event_updates,
    partial_apply_proposed_event_update,
    reject_proposed_event_update,
    search_events,
    update_event_suggestion_status,
)
from run4221.db.repository import (
    add_event as repo_add_event,
)
from run4221.db.repository import (
    archive_event as repo_archive_event,
)
from run4221.db.repository import (
    delete_event as repo_delete_event,
)
from run4221.db.repository import (
    restore_event as repo_restore_event,
)
from run4221.db.repository import (
    update_event as repo_update_event,
)
from run4221.events import TrackedEvent, normalize_event_id

if TYPE_CHECKING:
    from run4221.researcher.engine import ResearchEngine
    from run4221.researcher.service import ProfileJobResult, ResearchJobResult

JsonObject = dict[str, Any]


@dataclass(frozen=True)
class AgentToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    restricted: bool = False

    @classmethod
    def success(cls, data: Any = None) -> AgentToolResult:
        return cls(ok=True, data=data)

    @classmethod
    def failure(cls, error: str, *, restricted: bool = False) -> AgentToolResult:
        return cls(ok=False, error=error, restricted=restricted)

    def to_dict(self) -> JsonObject:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "restricted": self.restricted,
        }


@dataclass(frozen=True)
class ModeratorAgentToolSpec:
    name: str
    description: str
    destructive: bool = False
    requires_confirmation_code: bool = False

    def to_dict(self) -> JsonObject:
        return asdict(self)


MODERATOR_AGENT_TOOL_SPECS = (
    ModeratorAgentToolSpec("todo", "Show pending moderator work counts."),
    ModeratorAgentToolSpec("list_events", "List tracked events, optionally filtered by tag."),
    ModeratorAgentToolSpec("search_events", "Search tracked events."),
    ModeratorAgentToolSpec("show_event", "Show one tracked event by ID."),
    ModeratorAgentToolSpec("discover_event_profile", "Extract a draft event profile from a URL."),
    ModeratorAgentToolSpec("create_event", "Create a tracked event from confirmed fields."),
    ModeratorAgentToolSpec("edit_event", "Edit tracked event fields without changing public ID."),
    ModeratorAgentToolSpec("update_event", "Run a researcher registration check for an event."),
    ModeratorAgentToolSpec("archive_event", "Archive an active event."),
    ModeratorAgentToolSpec("list_archive", "List archived events."),
    ModeratorAgentToolSpec("restore_event", "Restore an archived event."),
    ModeratorAgentToolSpec(
        "delete_event",
        "Permanently delete an event; requires a configured delete confirmation code.",
        destructive=True,
        requires_confirmation_code=True,
    ),
    ModeratorAgentToolSpec("list_updates", "List pending proposed updates."),
    ModeratorAgentToolSpec("next_update", "Show the oldest pending proposed update."),
    ModeratorAgentToolSpec("show_update", "Show one pending proposed update."),
    ModeratorAgentToolSpec("apply_update", "Apply all fields from a pending proposed update."),
    ModeratorAgentToolSpec("partial_apply_update", "Apply selected fields from a proposed update."),
    ModeratorAgentToolSpec("reject_update", "Reject a pending proposed update."),
    ModeratorAgentToolSpec("list_suggestions", "List pending subscriber suggestions."),
    ModeratorAgentToolSpec("next_suggestion", "Show the oldest pending subscriber suggestion."),
    ModeratorAgentToolSpec("show_suggestion", "Show one pending subscriber suggestion."),
    ModeratorAgentToolSpec(
        "apply_suggestion",
        "Prepare a pending suggestion for event creation by extracting a draft from its URL.",
    ),
    ModeratorAgentToolSpec("reject_suggestion", "Reject a pending subscriber suggestion."),
)


class ModeratorAgentTools:
    """Telegram-independent moderator operations for an AI agent.

    The methods return JSON-like data and avoid Telegram message objects, FSM state, HTML,
    and inline keyboards. A UI layer can wrap these functions for Telegram, CLI, or agent
    function calling.
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        delete_confirmation_code: str | None = None,
        engine: ResearchEngine | None = None,
    ) -> None:
        self.database_url = database_url
        self.delete_confirmation_code = delete_confirmation_code
        self.engine = engine

    def _acquire_engine(self) -> ResearchEngine:
        """Lazily build and cache the researcher engine; fail closed on errors."""

        if self.engine is None:
            from run4221.researcher.engine import build_engine

            self.engine = build_engine()
        return self.engine

    def tool_specs(self) -> tuple[JsonObject, ...]:
        return moderator_agent_tool_specs()

    def tool_functions(self) -> dict[str, Any]:
        return {
            "todo": self.todo,
            "list_events": self.list_events,
            "search_events": self.search_events,
            "show_event": self.show_event,
            "discover_event_profile": self.discover_event_profile,
            "create_event": self.create_event,
            "edit_event": self.edit_event,
            "update_event": self.update_event,
            "archive_event": self.archive_event,
            "list_archive": self.list_archive,
            "restore_event": self.restore_event,
            "delete_event": self.delete_event,
            "list_updates": self.list_updates,
            "next_update": self.next_update,
            "show_update": self.show_update,
            "apply_update": self.apply_update,
            "partial_apply_update": self.partial_apply_update,
            "reject_update": self.reject_update,
            "list_suggestions": self.list_suggestions,
            "next_suggestion": self.next_suggestion,
            "show_suggestion": self.show_suggestion,
            "apply_suggestion": self.apply_suggestion,
            "reject_suggestion": self.reject_suggestion,
        }

    def todo(self) -> AgentToolResult:
        return AgentToolResult.success(
            {
                "pending_updates": count_proposed_event_updates(
                    status="pending",
                    database_url=self.database_url,
                ),
                "pending_suggestions": count_event_suggestions(
                    status="pending",
                    database_url=self.database_url,
                ),
            }
        )

    def list_events(
        self,
        *,
        tag: str | None = None,
        open_only: bool = False,
        limit: int = 10,
    ) -> AgentToolResult:
        if open_only:
            events = list_open_events(tag=tag, limit=limit, database_url=self.database_url)
        elif tag:
            events = list_events_by_tag(tag, limit=limit, database_url=self.database_url)
        else:
            events = list_events(limit=limit, database_url=self.database_url)
        return AgentToolResult.success([serialize_event(event) for event in events])

    def search_events(self, query: str, *, limit: int = 10) -> AgentToolResult:
        events = search_events(query, database_url=self.database_url)[:limit]
        return AgentToolResult.success([serialize_event(event) for event in events])

    def show_event(self, event_id: str) -> AgentToolResult:
        event = find_event(event_id, database_url=self.database_url)
        if event is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")
        return AgentToolResult.success(serialize_event(event))

    async def discover_event_profile(self, url: str) -> AgentToolResult:
        try:
            engine = self._acquire_engine()
        except Exception as error:
            return AgentToolResult.failure(f"Researcher engine is not configured: {error}")
        try:
            result = await engine.profile(url)
        except Exception as error:
            return AgentToolResult.failure(f"Could not discover event profile: {error}")
        if result.draft is None:
            failure = research_outcome_headline(result)
            if result.status.detail:
                failure = f"{failure} {result.status.detail}"
            return AgentToolResult.failure(f"{failure} (run {result.run_id})")
        return AgentToolResult.success(serialize_profile_result(result))

    def create_event(
        self,
        fields: Mapping[str, Any],
        *,
        source_suggestion_id: int | None = None,
    ) -> AgentToolResult:
        try:
            event_create = event_create_from_fields(fields)
            if source_suggestion_id is None:
                event = repo_add_event(event_create, self.database_url)
            else:
                event = add_event_from_suggestion(
                    event_create,
                    source_suggestion_id,
                    database_url=self.database_url,
                )
        except (EventWriteError, KeyError, TypeError, ValueError) as error:
            return AgentToolResult.failure(f"Could not create event: {error}")
        return AgentToolResult.success(serialize_event(event))

    def edit_event(self, event_id: str, fields: Mapping[str, Any]) -> AgentToolResult:
        existing = find_event(event_id, database_url=self.database_url)
        if existing is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")

        try:
            update = event_update_from_fields(existing, fields)
            event = repo_update_event(event_id, update, database_url=self.database_url)
        except (EventWriteError, TypeError, ValueError) as error:
            return AgentToolResult.failure(f"Could not edit event: {error}")

        if event is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")
        return AgentToolResult.success(serialize_event(event))

    async def update_event(self, event_id: str) -> AgentToolResult:
        event = find_event(event_id, database_url=self.database_url)
        if event is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")

        try:
            engine = self._acquire_engine()
        except Exception as error:
            return AgentToolResult.failure(f"Researcher engine is not configured: {error}")
        try:
            result = await engine.refresh_source(event.id)
        except ValueError as error:
            return AgentToolResult.failure(str(error))
        except Exception as error:
            return AgentToolResult.failure(f"Could not update event: {error}")
        return AgentToolResult.success(serialize_refresh_result(result))

    def archive_event(self, event_id: str) -> AgentToolResult:
        event = repo_archive_event(event_id, database_url=self.database_url)
        if event is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")
        return AgentToolResult.success(serialize_event(event))

    def list_archive(self, *, limit: int = 10) -> AgentToolResult:
        archived = list_archived_events(limit=limit, database_url=self.database_url)
        return AgentToolResult.success([serialize_archived_event(row) for row in archived])

    def restore_event(self, event_id: str) -> AgentToolResult:
        event = repo_restore_event(event_id, database_url=self.database_url)
        if event is None:
            return AgentToolResult.failure(f"Archived event not found: {event_id}")
        return AgentToolResult.success(serialize_event(event))

    def delete_event(
        self,
        event_id: str,
        *,
        confirmation_code: str | None = None,
        confirm_event_id: str | None = None,
    ) -> AgentToolResult:
        guard_error = self._delete_guard_error(
            event_id,
            confirmation_code=confirmation_code,
            confirm_event_id=confirm_event_id,
        )
        if guard_error is not None:
            return AgentToolResult.failure(guard_error, restricted=True)

        event = repo_delete_event(event_id, database_url=self.database_url)
        if event is None:
            return AgentToolResult.failure(f"Event not found: {event_id}")
        return AgentToolResult.success(serialize_event(event))

    def list_updates(self, *, limit: int = 10) -> AgentToolResult:
        updates = list_proposed_event_updates(limit=limit, database_url=self.database_url)
        return AgentToolResult.success([serialize_proposed_update(update) for update in updates])

    def next_update(self) -> AgentToolResult:
        updates = list_proposed_event_updates(limit=1, database_url=self.database_url)
        if not updates:
            return AgentToolResult.failure("No pending updates.")
        return AgentToolResult.success(serialize_proposed_update(updates[0]))

    def show_update(self, update_id: int) -> AgentToolResult:
        update = get_proposed_event_update(
            update_id,
            status="pending",
            database_url=self.database_url,
        )
        if update is None:
            return AgentToolResult.failure(f"Pending update not found: #{update_id}")
        return AgentToolResult.success(serialize_proposed_update(update))

    def apply_update(
        self,
        update_id: int,
        *,
        reviewer_user_id: str | None = "moderator-agent",
    ) -> AgentToolResult:
        try:
            result = approve_proposed_event_update(
                update_id,
                reviewer_user_id=reviewer_user_id,
                database_url=self.database_url,
            )
        except EventWriteError as error:
            return AgentToolResult.failure(f"Could not apply update: {error}")
        if result is None:
            return AgentToolResult.failure(f"Pending update not found: #{update_id}")
        return AgentToolResult.success(
            {
                "update": serialize_proposed_update(result.update),
                "event": serialize_event(result.event),
            }
        )

    def partial_apply_update(
        self,
        update_id: int,
        *,
        selected_fields: tuple[str, ...] | list[str],
        reviewer_user_id: str | None = "moderator-agent",
    ) -> AgentToolResult:
        try:
            result = partial_apply_proposed_event_update(
                update_id,
                selected_fields=tuple(selected_fields),
                reviewer_user_id=reviewer_user_id,
                database_url=self.database_url,
            )
        except EventWriteError as error:
            return AgentToolResult.failure(f"Could not partially apply update: {error}")
        if result is None:
            return AgentToolResult.failure(f"Pending update not found: #{update_id}")
        return AgentToolResult.success(
            {
                "update": serialize_proposed_update(result.update),
                "event": serialize_event(result.event),
                "follow_up_update": (
                    serialize_proposed_update(result.follow_up_update)
                    if result.follow_up_update is not None
                    else None
                ),
                "applied_fields": list(result.applied_fields),
                "remaining_fields": list(result.remaining_fields),
            }
        )

    def reject_update(
        self,
        update_id: int,
        *,
        reviewer_user_id: str | None = "moderator-agent",
    ) -> AgentToolResult:
        update = reject_proposed_event_update(
            update_id,
            reviewer_user_id=reviewer_user_id,
            database_url=self.database_url,
        )
        if update is None:
            return AgentToolResult.failure(f"Pending update not found: #{update_id}")
        return AgentToolResult.success(serialize_proposed_update(update))

    def list_suggestions(self, *, limit: int = 10) -> AgentToolResult:
        suggestions = list_event_suggestions(limit=limit, database_url=self.database_url)
        return AgentToolResult.success(
            [serialize_suggestion(suggestion) for suggestion in suggestions]
        )

    def next_suggestion(self) -> AgentToolResult:
        suggestions = list_event_suggestions(limit=1, database_url=self.database_url)
        if not suggestions:
            return AgentToolResult.failure("No pending suggestions.")
        return AgentToolResult.success(serialize_suggestion(suggestions[0]))

    def show_suggestion(self, suggestion_id: int) -> AgentToolResult:
        suggestion = get_event_suggestion(
            suggestion_id,
            status="pending",
            database_url=self.database_url,
        )
        if suggestion is None:
            return AgentToolResult.failure(f"Pending suggestion not found: #{suggestion_id}")
        return AgentToolResult.success(serialize_suggestion(suggestion))

    async def apply_suggestion(self, suggestion_id: int) -> AgentToolResult:
        suggestion = get_event_suggestion(
            suggestion_id,
            status="pending",
            database_url=self.database_url,
        )
        if suggestion is None:
            return AgentToolResult.failure(f"Pending suggestion not found: #{suggestion_id}")
        if not suggestion.url:
            return AgentToolResult.failure(
                "Suggestion has no URL. Reject it or create an event manually."
            )

        draft_result = await self.discover_event_profile(suggestion.url)
        if not draft_result.ok:
            return draft_result
        return AgentToolResult.success(
            {
                "suggestion": serialize_suggestion(suggestion),
                "draft": draft_result.data["draft"],
                "run_id": draft_result.data["run_id"],
                "next_step": "create_event with source_suggestion_id after moderator review",
            }
        )

    def reject_suggestion(self, suggestion_id: int) -> AgentToolResult:
        suggestion = update_event_suggestion_status(
            suggestion_id,
            "removed",
            database_url=self.database_url,
        )
        if suggestion is None:
            return AgentToolResult.failure(f"Suggestion not found: #{suggestion_id}")
        return AgentToolResult.success(serialize_suggestion(suggestion))

    def _delete_guard_error(
        self,
        event_id: str,
        *,
        confirmation_code: str | None,
        confirm_event_id: str | None,
    ) -> str | None:
        if not self.delete_confirmation_code:
            return "Permanent delete is disabled: no delete confirmation code is configured."
        if confirmation_code != self.delete_confirmation_code:
            return "Permanent delete requires the configured delete confirmation code."
        if normalize_event_id(confirm_event_id or "") != normalize_event_id(event_id):
            return "Permanent delete requires confirm_event_id to exactly match the event ID."
        return None


def moderator_agent_tool_specs() -> tuple[JsonObject, ...]:
    return tuple(spec.to_dict() for spec in MODERATOR_AGENT_TOOL_SPECS)


def create_moderator_agent_tools(
    *,
    database_url: str | None = None,
    delete_confirmation_code: str | None = None,
    engine: ResearchEngine | None = None,
) -> ModeratorAgentTools:
    return ModeratorAgentTools(
        database_url=database_url,
        delete_confirmation_code=delete_confirmation_code,
        engine=engine,
    )


def event_create_from_fields(fields: Mapping[str, Any]) -> EventCreate:
    return EventCreate(
        public_id=required_text(fields, "public_id"),
        name=required_text(fields, "name"),
        city=required_text(fields, "city"),
        country=required_text(fields, "country"),
        timezone=required_text(fields, "timezone"),
        distances=string_tuple(fields.get("distances")),
        regions=string_tuple(fields.get("regions")),
        official_url=required_text(fields, "official_url"),
        registration_url=optional_string(fields.get("registration_url")),
        event_date=optional_string(fields.get("event_date")),
        registration_status=optional_string(fields.get("registration_status")) or "unknown",
        registration_open_at=optional_string(fields.get("registration_open_at")),
        registration_open_precision=(
            optional_string(fields.get("registration_open_precision")) or "unknown"
        ),
        registration_close_at=optional_string(fields.get("registration_close_at")),
    )


def event_update_from_fields(event: TrackedEvent, fields: Mapping[str, Any]) -> EventUpdate:
    return EventUpdate(
        name=optional_string(fields.get("name")) or event.name,
        city=optional_string(fields.get("city")) or event.city,
        country=optional_string(fields.get("country")) or event.country,
        timezone=optional_string(fields.get("timezone")) or event.timezone,
        distances=string_tuple(fields.get("distances"), fallback=event.distances),
        regions=string_tuple(fields.get("regions"), fallback=event.regions),
        official_url=optional_string(fields.get("official_url")) or event.official_url,
        registration_url=optional_string(
            fields.get("registration_url"),
            keep_empty_as_none=True,
            fallback=event.registration_url,
        ),
        event_date=optional_string(
            fields.get("event_date"),
            keep_empty_as_none=True,
            fallback=event.event_date,
        ),
        registration_status=(
            optional_string(fields.get("registration_status")) or event.registration_status
        ),
        registration_open_at=optional_string(
            fields.get("registration_open_at"),
            keep_empty_as_none=True,
            fallback=event.registration_open_at,
        ),
        registration_open_precision=(
            optional_string(fields.get("registration_open_precision"))
            or event.registration_open_precision
        ),
        registration_close_at=optional_string(
            fields.get("registration_close_at"),
            keep_empty_as_none=True,
            fallback=event.registration_close_at,
        ),
    )


def serialize_event(event: TrackedEvent) -> JsonObject:
    return {
        "id": event.id,
        "public_id": event.public_id,
        "name": event.name,
        "city": event.city,
        "country": event.country,
        "timezone": event.timezone,
        "distances": list(event.distances),
        "regions": list(event.regions),
        "collections": list(event.collections),
        "tags": list(event.tags),
        "event_date": event.event_date,
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at,
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at,
        "official_url": event.official_url,
        "registration_url": event.registration_url,
    }


def serialize_archived_event(archived) -> JsonObject:
    return {
        "event": serialize_event(archived.event),
        "removed_at": archived.removed_at,
    }


def serialize_proposed_update(update) -> JsonObject:
    return {
        "id": update.id,
        "handle": f"#{update.id}",
        "event_id": update.event_id,
        "update_type": update.update_type,
        "current_fields": dict(update.current_fields),
        "proposed_fields": dict(update.proposed_fields),
        "evidence": list(update.evidence),
        "confidence": update.confidence,
        "status": update.status,
        "change_summary": update.change_summary,
    }


def serialize_suggestion(suggestion) -> JsonObject:
    return {
        "id": suggestion.id,
        "handle": f"#{suggestion.id}",
        "status": suggestion.status,
        "event_name": suggestion.event_name,
        "url": suggestion.url,
        "event_date": suggestion.event_date,
        "location": suggestion.location,
        "region_tags": list(suggestion.region_tags),
        "distances": list(suggestion.distances),
        "note": suggestion.note,
        "submitter_user_id": suggestion.submitter_user_id,
        "submitter_username": suggestion.submitter_username,
        "submitter_display_name": suggestion.submitter_display_name,
    }


def serialize_profile_result(result: ProfileJobResult) -> JsonObject:
    return {
        "run_id": result.run_id,
        "status": str(result.status.status),
        "outcome": str(result.status.outcome),
        "detail": result.status.detail,
        "message": research_outcome_headline(result),
        "draft": (
            result.draft.model_dump(mode="json") if result.draft is not None else None
        ),
    }


def serialize_refresh_result(result: ResearchJobResult) -> JsonObject:
    return {
        "run_id": result.run_id,
        "status": str(result.status.status),
        "outcome": str(result.status.outcome),
        "detail": result.status.detail,
        "queue_reference": result.queue_reference,
        "conflicting_update_id": result.conflicting_update_id,
        "message": research_outcome_headline(result),
    }


def serialize_dataclass(value) -> JsonObject:
    data = asdict(value)
    return json_ready(data)


def json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    return value


def required_text(fields: Mapping[str, Any], key: str) -> str:
    value = optional_string(fields.get(key))
    if value is None:
        raise KeyError(key)
    return value


def optional_string(
    value: Any,
    *,
    keep_empty_as_none: bool = False,
    fallback: str | None = None,
) -> str | None:
    if value is None:
        return fallback
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    if not stripped or stripped == "-":
        return None if keep_empty_as_none else fallback
    return stripped


def string_tuple(value: Any, *, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return fallback
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return tuple(str(part).strip() for part in parts if str(part).strip())
