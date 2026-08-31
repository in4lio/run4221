from __future__ import annotations

from enum import StrEnum
from pathlib import PurePath
from typing import Annotated, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ShortText = Annotated[str, Field(min_length=1, max_length=240)]
SummaryText = Annotated[str, Field(min_length=1, max_length=1_000)]
EvidenceText = Annotated[str, Field(min_length=1, max_length=1_000)]
EvidenceList = Annotated[list[EvidenceText], Field(max_length=8)]
RESEARCHER_MAX_PENDING_SUGGESTIONS = 20


class ResearchSchema(BaseModel):
    """Closed, framework-neutral base for data that crosses the model boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def validate_http_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be an absolute HTTP(S) URL.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL credentials are not allowed.")
    return value


class ResearchCandidate(ResearchSchema):
    source_url: Annotated[str, Field(min_length=1, max_length=2_048)]
    title: ShortText
    snippet: EvidenceText
    discovery_query: Annotated[str, Field(min_length=1, max_length=500)] | None = None
    event_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None
    location: Annotated[str, Field(min_length=1, max_length=240)] | None = None
    region_tags: Annotated[tuple[ShortText, ...], Field(max_length=12)] = ()
    distances: Annotated[tuple[ShortText, ...], Field(max_length=12)] = ()

    _validate_source_url = field_validator("source_url")(validate_http_url)


class AssessmentVerdict(StrEnum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ResearchAssessment(ResearchSchema):
    verdict: AssessmentVerdict
    summary: SummaryText
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: EvidenceList = Field(default_factory=list)


class RegistrationStatus(StrEnum):
    UNKNOWN = "unknown"
    ANNOUNCED = "announced"
    OPEN = "open"
    WAITLIST = "waitlist"
    CLOSED = "closed"
    SOLD_OUT = "sold_out"


class RegistrationOpenPrecision(StrEnum):
    UNKNOWN = "unknown"
    DATE_ONLY = "date_only"
    DATETIME = "datetime"
    MONTH_ONLY = "month_only"
    ESTIMATED = "estimated"


class ProposedEventChanges(ResearchSchema):
    registration_status: RegistrationStatus | None = None
    registration_open_at: Annotated[str, Field(max_length=40)] | None = None
    registration_open_precision: RegistrationOpenPrecision | None = None
    registration_close_at: Annotated[str, Field(max_length=40)] | None = None
    registration_url: Annotated[str, Field(min_length=1, max_length=2_048)] | None = None
    event_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None

    @field_validator("registration_url")
    @classmethod
    def validate_registration_url(cls, value: str | None) -> str | None:
        return None if value is None else validate_http_url(value)


class ArtifactReference(ResearchSchema):
    run_id: str
    artifact_name: Annotated[str, Field(min_length=1, max_length=180)]
    source_url: Annotated[str, Field(min_length=1, max_length=2_048)]
    content_hash: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]

    _validate_source_url = field_validator("source_url")(validate_http_url)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        UUID(value)
        return value

    @field_validator("artifact_name")
    @classmethod
    def validate_artifact_name(cls, value: str) -> str:
        if PurePath(value).name != value or value in {".", ".."}:
            raise ValueError("Artifact name must be a basename.")
        return value


class DecisionAction(StrEnum):
    NO_CHANGE = "no_change"
    SUGGEST_EVENT = "suggest_event"
    PROPOSE_UPDATE = "propose_update"
    INCONCLUSIVE = "inconclusive"


class ResearchDecision(ResearchSchema):
    action: DecisionAction
    summary: SummaryText
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate: ResearchCandidate | None = None
    proposed_fields: ProposedEventChanges | None = None
    evidence: Annotated[list[ArtifactReference], Field(max_length=8)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_action_payload(self) -> Self:
        if self.action is DecisionAction.SUGGEST_EVENT:
            if self.candidate is None:
                raise ValueError("suggest_event requires a candidate.")
            if self.proposed_fields is not None:
                raise ValueError("suggest_event cannot include proposed_fields.")
        if self.action is DecisionAction.PROPOSE_UPDATE:
            if self.proposed_fields is None:
                raise ValueError("propose_update requires proposed_fields.")
            if self.candidate is not None:
                raise ValueError("propose_update cannot include a candidate.")
        if self.action not in {
            DecisionAction.SUGGEST_EVENT,
            DecisionAction.PROPOSE_UPDATE,
        } and (self.candidate is not None or self.proposed_fields is not None):
            raise ValueError("Non-persisting actions cannot include a queue payload.")
        return self


class ResearchBudget(ResearchSchema):
    max_events_per_cycle: int = Field(default=5, ge=1, le=100)
    max_candidates_per_cycle: int = Field(default=3, ge=1, le=100)
    max_agent_turns_per_job: int = Field(default=6, ge=1, le=20)
    max_web_searches_per_job: int = Field(default=2, ge=0, le=20)
    max_static_pages_per_job: int = Field(default=4, ge=1, le=50)
    max_rendered_pages_per_job: int = Field(default=0, ge=0, le=10)
    max_retries_per_job: int = Field(default=2, ge=0, le=10)
    max_output_tokens_per_job: int = Field(default=4_000, ge=128, le=16_000)
    max_wall_time_seconds_per_job: int = Field(default=90, ge=10, le=900)
    max_pending_suggestions: int = Field(
        default=RESEARCHER_MAX_PENDING_SUGGESTIONS,
        ge=0,
        le=RESEARCHER_MAX_PENDING_SUGGESTIONS,
    )
    max_pending_updates: int = Field(default=50, ge=0, le=500)


class RunState(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CAPPED = "capped"
    SKIPPED = "skipped"


class RunOutcome(StrEnum):
    NO_CHANGE = "no_change"
    SUGGESTION_CREATED = "suggestion_created"
    PROPOSAL_CREATED = "proposal_created"
    INCONCLUSIVE = "inconclusive"


class ResearchRunStatus(ResearchSchema):
    status: RunState
    outcome: RunOutcome
    detail: Annotated[str, Field(min_length=1, max_length=1_000)] | None = None
