from __future__ import annotations

from enum import StrEnum
from pathlib import PurePath
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ShortText = Annotated[str, Field(min_length=1, max_length=240)]
SummaryText = Annotated[str, Field(min_length=1, max_length=1_000)]
EvidenceText = Annotated[str, Field(min_length=1, max_length=1_000)]
EvidenceList = Annotated[list[EvidenceText], Field(max_length=8)]
EvidenceKey = Annotated[str, Field(pattern=r"^E[1-8]$")]
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


class EventUpdateField(StrEnum):
    REGISTRATION_STATUS = "registration_status"
    REGISTRATION_OPEN_AT = "registration_open_at"
    REGISTRATION_OPEN_PRECISION = "registration_open_precision"
    REGISTRATION_CLOSE_AT = "registration_close_at"
    REGISTRATION_URL = "registration_url"
    EVENT_DATE = "event_date"


class ProposedEventChanges(ResearchSchema):
    registration_status: RegistrationStatus | None = None
    registration_open_at: Annotated[str, Field(max_length=40)] | None = None
    registration_open_precision: RegistrationOpenPrecision | None = None
    registration_close_at: Annotated[str, Field(max_length=40)] | None = None
    registration_url: Annotated[str, Field(min_length=1, max_length=2_048)] | None = None
    event_date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None
    clear_fields: Annotated[
        tuple[
            Literal[
                "registration_open_at",
                "registration_close_at",
                "registration_url",
                "event_date",
            ],
            ...,
        ],
        Field(max_length=4),
    ] = ()

    @field_validator("registration_url")
    @classmethod
    def validate_registration_url(cls, value: str | None) -> str | None:
        return None if value is None else validate_http_url(value)

    @model_validator(mode="after")
    def validate_clear_fields(self) -> Self:
        if len(set(self.clear_fields)) != len(self.clear_fields):
            raise ValueError("clear_fields cannot contain duplicates.")
        conflicts = [field for field in self.clear_fields if getattr(self, field) is not None]
        if conflicts:
            raise ValueError("A field cannot be set and cleared in the same decision.")
        if not self.changed_fields:
            raise ValueError("At least one event field must be set or cleared.")
        return self

    @property
    def changed_fields(self) -> frozenset[EventUpdateField]:
        fields = {
            field for field in EventUpdateField if getattr(self, field.value) is not None
        }
        fields.update(EventUpdateField(name) for name in self.clear_fields)
        return frozenset(fields)


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


class EvidenceRequestPurpose(StrEnum):
    REGISTRATION_STATUS = "registration_status"
    REGISTRATION_TIMING = "registration_timing"
    REGISTRATION_URL = "registration_url"
    EVENT_DATE = "event_date"
    EVENT_IDENTITY = "event_identity"
    EVENT_EDITION = "event_edition"
    DISTANCE_CATEGORY = "distance_category"
    CONFLICT_RESOLUTION = "conflict_resolution"


class EvidenceApplicability(ResearchSchema):
    evidence_key: EvidenceKey
    event_identity: AssessmentVerdict
    event_edition: AssessmentVerdict
    distance_category: AssessmentVerdict
    applicable_fields: Annotated[
        tuple[EventUpdateField, ...],
        Field(max_length=len(EventUpdateField)),
    ] = ()

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if len(set(self.applicable_fields)) != len(self.applicable_fields):
            raise ValueError("applicable_fields cannot contain duplicates.")
        return self


class FieldEvidenceSupport(ResearchSchema):
    field: EventUpdateField
    evidence_keys: Annotated[
        tuple[EvidenceKey, ...],
        Field(min_length=1, max_length=8),
    ]

    @model_validator(mode="after")
    def validate_evidence_keys(self) -> Self:
        if len(set(self.evidence_keys)) != len(self.evidence_keys):
            raise ValueError("evidence_keys cannot contain duplicates.")
        return self


class EvidenceConflict(ResearchSchema):
    field: EventUpdateField | None = None
    evidence_keys: Annotated[
        tuple[EvidenceKey, ...],
        Field(min_length=2, max_length=8),
    ]
    summary: EvidenceText

    @model_validator(mode="after")
    def validate_evidence_keys(self) -> Self:
        if len(set(self.evidence_keys)) != len(self.evidence_keys):
            raise ValueError("A conflict must cite distinct evidence keys.")
        return self


class AssessorTerminalDecision(ResearchSchema):
    summary: SummaryText
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: SummaryText | None = None
    applicability: Annotated[
        tuple[EvidenceApplicability, ...],
        Field(max_length=8),
    ] = ()
    conflicts: Annotated[tuple[EvidenceConflict, ...], Field(max_length=8)] = ()

    @model_validator(mode="after")
    def validate_unique_applicability(self) -> Self:
        keys = [item.evidence_key for item in self.applicability]
        if len(set(keys)) != len(keys):
            raise ValueError("applicability must contain at most one item per evidence key.")
        if any(not set(conflict.evidence_keys).issubset(keys) for conflict in self.conflicts):
            raise ValueError("Every conflict key requires an applicability item.")
        return self


class AssessorNoPayloadDecision(AssessorTerminalDecision):
    action: Literal[DecisionAction.NO_CHANGE, DecisionAction.INCONCLUSIVE]


class AssessorSuggestionDecision(AssessorTerminalDecision):
    action: Literal[DecisionAction.SUGGEST_EVENT]
    candidate: ResearchCandidate


class AssessorUpdateDecision(AssessorTerminalDecision):
    action: Literal[DecisionAction.PROPOSE_UPDATE]
    proposed_fields: ProposedEventChanges
    applicability: Annotated[
        tuple[EvidenceApplicability, ...],
        Field(min_length=1, max_length=8),
    ]
    field_support: Annotated[
        tuple[FieldEvidenceSupport, ...],
        Field(min_length=1, max_length=len(EventUpdateField)),
    ]

    @model_validator(mode="after")
    def validate_update_support(self) -> Self:
        support_fields = [item.field for item in self.field_support]
        if len(set(support_fields)) != len(support_fields):
            raise ValueError("Each proposed field must have exactly one support item.")
        if set(support_fields) != set(self.proposed_fields.changed_fields):
            raise ValueError("field_support must exactly match the proposed fields.")

        applicability = {item.evidence_key: item for item in self.applicability}
        for support in self.field_support:
            for evidence_key in support.evidence_keys:
                item = applicability.get(evidence_key)
                if item is None:
                    raise ValueError("Every support key requires an applicability item.")
                if (
                    item.event_identity is not AssessmentVerdict.CONFIRMED
                    or item.event_edition is not AssessmentVerdict.CONFIRMED
                    or item.distance_category is not AssessmentVerdict.CONFIRMED
                    or support.field not in item.applicable_fields
                ):
                    raise ValueError(
                        "Field support requires confirmed identity, edition, "
                        "distance/category, and field applicability."
                    )

        changed_fields = self.proposed_fields.changed_fields
        if any(
            conflict.field is None or conflict.field in changed_fields
            for conflict in self.conflicts
        ):
            raise ValueError("A proposed update cannot source-order conflicting evidence.")
        return self


class EvidenceRequest(ResearchSchema):
    action: Literal["request_evidence"]
    purpose: EvidenceRequestPurpose
    query: Annotated[str, Field(min_length=1, max_length=500)]
    gap: EvidenceText


AssessorDecision = Annotated[
    AssessorNoPayloadDecision | AssessorSuggestionDecision | AssessorUpdateDecision,
    Field(discriminator="action"),
]

AssessorOutcome = Annotated[
    AssessorNoPayloadDecision
    | AssessorSuggestionDecision
    | AssessorUpdateDecision
    | EvidenceRequest,
    Field(discriminator="action"),
]


class ResolvedEvidenceApplicability(ResearchSchema):
    evidence: ArtifactReference
    event_identity: AssessmentVerdict
    event_edition: AssessmentVerdict
    distance_category: AssessmentVerdict
    applicable_fields: Annotated[
        tuple[EventUpdateField, ...],
        Field(max_length=len(EventUpdateField)),
    ] = ()

    @model_validator(mode="after")
    def validate_fields(self) -> Self:
        if len(set(self.applicable_fields)) != len(self.applicable_fields):
            raise ValueError("applicable_fields cannot contain duplicates.")
        return self


class ResolvedFieldEvidenceSupport(ResearchSchema):
    field: EventUpdateField
    evidence: Annotated[
        tuple[ArtifactReference, ...],
        Field(min_length=1, max_length=8),
    ]

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("Field evidence cannot contain duplicates.")
        return self


class ResolvedEvidenceConflict(ResearchSchema):
    field: EventUpdateField | None = None
    evidence: Annotated[
        tuple[ArtifactReference, ...],
        Field(min_length=2, max_length=8),
    ]
    summary: EvidenceText

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("A conflict must cite distinct evidence.")
        return self


class ResearchDecision(ResearchSchema):
    action: DecisionAction
    summary: SummaryText
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate: ResearchCandidate | None = None
    proposed_fields: ProposedEventChanges | None = None
    evidence: Annotated[list[ArtifactReference], Field(max_length=8)] = Field(default_factory=list)
    uncertainty: SummaryText | None = None
    applicability: Annotated[
        tuple[ResolvedEvidenceApplicability, ...],
        Field(max_length=8),
    ] = ()
    field_support: Annotated[
        tuple[ResolvedFieldEvidenceSupport, ...],
        Field(max_length=len(EventUpdateField)),
    ] = ()
    conflicts: Annotated[
        tuple[ResolvedEvidenceConflict, ...],
        Field(max_length=8),
    ] = ()

    @model_validator(mode="after")
    def validate_action_payload(self) -> Self:
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError("Decision evidence cannot contain duplicates.")
        applicability = {item.evidence: item for item in self.applicability}
        if len(applicability) != len(self.applicability):
            raise ValueError("Applicability must be unique per artifact.")
        if any(not set(conflict.evidence).issubset(applicability) for conflict in self.conflicts):
            raise ValueError("Every conflict artifact requires an applicability item.")

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
            if not self.field_support:
                raise ValueError("propose_update requires exact per-field support.")
            support_fields = [item.field for item in self.field_support]
            if len(set(support_fields)) != len(support_fields):
                raise ValueError("Each proposed field must have exactly one support item.")
            if set(support_fields) != set(self.proposed_fields.changed_fields):
                raise ValueError("field_support must exactly match the proposed fields.")

            for support in self.field_support:
                for evidence in support.evidence:
                    item = applicability.get(evidence)
                    if item is None:
                        raise ValueError("Every supported artifact requires an applicability item.")
                    if (
                        item.event_identity is not AssessmentVerdict.CONFIRMED
                        or item.event_edition is not AssessmentVerdict.CONFIRMED
                        or item.distance_category is not AssessmentVerdict.CONFIRMED
                        or support.field not in item.applicable_fields
                    ):
                        raise ValueError("Resolved field support failed an applicability gate.")

            changed_fields = self.proposed_fields.changed_fields
            if any(
                conflict.field is None or conflict.field in changed_fields
                for conflict in self.conflicts
            ):
                raise ValueError("A proposed update cannot source-order conflicting evidence.")
        if self.action not in {
            DecisionAction.SUGGEST_EVENT,
            DecisionAction.PROPOSE_UPDATE,
        } and (self.candidate is not None or self.proposed_fields is not None):
            raise ValueError("Non-persisting actions cannot include a queue payload.")
        if self.action is not DecisionAction.PROPOSE_UPDATE and self.field_support:
            raise ValueError("Only propose_update can include field_support.")
        return self


class ResearchBudget(ResearchSchema):
    max_events_per_cycle: int = Field(default=5, ge=1, le=100)
    max_candidates_per_cycle: int = Field(default=3, ge=1, le=100)
    max_agent_turns_per_job: int = Field(default=6, ge=1, le=20)
    max_web_searches_per_job: int = Field(default=2, ge=0, le=20)
    max_static_pages_per_job: int = Field(default=4, ge=1, le=50)
    max_rendered_pages_per_job: int = Field(default=0, ge=0, le=10)
    max_retries_per_job: int = Field(default=2, ge=0, le=10)
    max_output_tokens_per_job: int = Field(default=2_000, ge=128, le=16_000)
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
