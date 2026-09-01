from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Protocol

from run4221.ai.event_extractor import conflicts_with_event_identity
from run4221.db.research import (
    EventSuggestionCreate,
    ProposedEventUpdateCreate,
    ResearchSourceRecord,
    admit_proposed_update,
    admit_suggestion,
    normalize_url,
)
from run4221.events import TrackedEvent
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageSnapshot,
    blocked_page_reason,
    fetch_page_snapshot,
)
from run4221.researcher.agent import (
    AgentRunState,
    AssessmentRequest,
    AssessmentRunResult,
    CapturedSnapshotEvidence,
    FrozenContextField,
    ScoutRequest,
    ScoutRunResult,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.policy import SourceTrustPolicy, source_domain
from run4221.researcher.schemas import (
    ArtifactReference,
    DecisionAction,
    ProposedEventChanges,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
    ResearchRunStatus,
    RunOutcome,
    RunState,
)

AUDIT_SOURCE_URL = "https://run4221.invalid/researcher"
MAX_REFRESH_FOLLOWUP_PAGES = 2
SUPPORTED_UPDATE_FIELDS = (
    "registration_status",
    "registration_open_at",
    "registration_open_precision",
    "registration_close_at",
    "registration_url",
    "event_date",
)

type SnapshotFetcher = Callable[[str], Awaitable[PageSnapshot]]


class ResearchAgent(Protocol):
    async def scout(self, request: ScoutRequest) -> ScoutRunResult: ...

    async def assess(self, request: AssessmentRequest) -> AssessmentRunResult: ...


@dataclass(frozen=True)
class ResearchJobResult:
    run_id: str
    status: ResearchRunStatus
    terminal_reference: ArtifactReference
    queue_reference: str | None = None


class ResearcherService:
    """Deterministic bridge from captured evidence to proposal-only queues."""

    def __init__(
        self,
        *,
        database_url: str,
        artifacts: ResearchArtifactStore,
        agent: ResearchAgent,
        trust_policy: SourceTrustPolicy,
        budget: ResearchBudget,
        fetch_snapshot: SnapshotFetcher = fetch_page_snapshot,
        persist_queue: bool = True,
        on_run_started: Callable[[str], None] | None = None,
    ) -> None:
        self.database_url = database_url
        self.artifacts = artifacts
        self.agent = agent
        self.trust_policy = trust_policy
        self.budget = budget
        self.fetch_snapshot = fetch_snapshot
        self.persist_queue = persist_queue
        self.on_run_started = on_run_started

    async def refresh(self, source: ResearchSourceRecord) -> ResearchJobResult:
        """Assess one frozen approved source and, at most, enqueue one proposal."""

        frozen_event = source.event
        frozen_fields = _current_fields(frozen_event)
        run_id = self.artifacts.create_run(
            job_type="refresh",
            metadata={
                "event_id": frozen_event.id,
                "source_id": source.source_id,
                "source_url": source.url,
                "frozen_fields": frozen_fields,
            },
        )
        self._announce_run(run_id)
        try:
            captured = await self._capture(run_id, source.url)
        except PageFetchError as error:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.FAILED,
                RunOutcome.INCONCLUSIVE,
                f"Approved source capture failed ({type(error).__name__}).",
            )
        if reason := blocked_page_reason(captured.snapshot):
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                f"Approved source was unusable: {reason}.",
            )

        captures = [captured]
        if self._refresh_search_enabled():
            scout = await self.agent.scout(
                ScoutRequest(
                    mode="refresh",
                    query=_refresh_search_query(frozen_event, source.url),
                    approved_source_url=source.url,
                    context=_event_context(frozen_event, source.url),
                )
            )
            self._record_scout(run_id, source.url, scout)
            if scout.state is not AgentRunState.SUCCEEDED:
                return self._finish_agent_outcome(run_id, source.url, scout)

            followup_limit = min(
                MAX_REFRESH_FOLLOWUP_PAGES,
                self.budget.max_static_pages_per_job - 1,
            )
            captured_final_urls = {normalize_url(captured.snapshot.final_url)}
            for candidate_url in _refresh_candidate_urls(
                scout.candidates,
                event=frozen_event,
                approved_source_url=source.url,
            ):
                if len(captures) - 1 >= followup_limit:
                    break
                try:
                    followup = await self._capture(run_id, candidate_url)
                except PageFetchError:
                    continue
                if blocked_page_reason(followup.snapshot) is not None:
                    continue
                if not _same_source_domain(
                    followup.snapshot.final_url,
                    source.url,
                ):
                    continue
                if conflicts_with_event_identity(
                    followup.snapshot.final_url,
                    followup.snapshot.title or "",
                    frozen_event.distances,
                ):
                    continue
                normalized_final_url = normalize_url(followup.snapshot.final_url)
                if normalized_final_url in captured_final_urls:
                    continue
                captures.append(followup)
                captured_final_urls.add(normalized_final_url)

        assessment = await self.agent.assess(
            AssessmentRequest(
                mode="refresh",
                context=_event_context(frozen_event, source.url),
                evidence=tuple(item.as_agent_evidence() for item in captures),
            )
        )
        self._record_assessment(run_id, source.url, assessment)
        if assessment.state is not AgentRunState.SUCCEEDED or assessment.decision is None:
            return self._finish_agent_outcome(run_id, source.url, assessment)

        decision = assessment.decision
        if decision.action in {DecisionAction.NO_CHANGE, DecisionAction.INCONCLUSIVE}:
            outcome = (
                RunOutcome.NO_CHANGE
                if decision.action is DecisionAction.NO_CHANGE
                else RunOutcome.INCONCLUSIVE
            )
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SUCCEEDED if outcome is RunOutcome.NO_CHANGE else RunState.SKIPPED,
                outcome,
                decision.summary,
            )
        if not self._valid_evidence(
            decision,
            required=tuple(item.reference for item in captures),
        ):
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Decision did not cite exactly captured approved-source evidence.",
            )
        if decision.action is not DecisionAction.PROPOSE_UPDATE:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Refresh decisions cannot create new-event suggestions.",
            )

        changed_fields = _changed_supported_fields(decision, frozen_fields)
        if not changed_fields:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.NO_CHANGE,
                "No supported tracked-event field changed.",
            )
        prepared_decision = decision.model_copy(
            update={
                "proposed_fields": _normalized_proposed_changes(changed_fields),
            }
        )
        if not self.persist_queue:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SUCCEEDED,
                RunOutcome.INCONCLUSIVE,
                "Shadow mode validated a supported update without writing the queue.",
            )
        committed_status = ResearchRunStatus(
            status=RunState.SUCCEEDED,
            outcome=RunOutcome.PROPOSAL_CREATED,
        )
        prepared = self.artifacts.prepare_decision(
            run_id,
            source_url=source.url,
            decision=prepared_decision,
            committed_status=committed_status,
        )
        evidence_lines = _moderation_evidence(
            decision.summary,
            tuple(captures),
            trust_reason=(
                "stored approved event source plus same-domain official search results"
                if len(captures) > 1
                else "stored approved event source"
            ),
            prepared=prepared,
        )
        try:
            admission = admit_proposed_update(
                ProposedEventUpdateCreate(
                    event_id=frozen_event.id,
                    update_type="registration_window",
                    current_fields={key: frozen_fields[key] for key in changed_fields},
                    proposed_fields=changed_fields,
                    evidence=evidence_lines,
                    confidence=decision.confidence,
                    change_summary=decision.summary,
                ),
                max_pending=self.budget.max_pending_updates,
                database_url=self.database_url,
            )
        except Exception:
            return self._finish_rejected_admission(
                prepared,
                RunState.FAILED,
                "Proposal admission failed before commit.",
            )
        if admission.outcome != "admitted" or admission.update is None:
            detail = {
                "conflicting_pending": "A conflicting pending proposal already exists.",
                "queue_full": "The researcher proposal queue is full.",
            }[admission.outcome]
            return self._finish_rejected_admission(
                prepared,
                RunState.CAPPED if admission.outcome == "queue_full" else RunState.SKIPPED,
                detail,
            )

        queue_reference = f"proposed_event_update:{admission.update.id}"
        terminal = self.artifacts.finalize_committed(
            prepared,
            queue_reference=queue_reference,
        )
        return ResearchJobResult(
            run_id,
            committed_status,
            terminal,
            queue_reference=queue_reference,
        )

    async def discover(self, query: str) -> ResearchJobResult:
        """Search and capture bounded candidates, admitting at most one suggestion."""

        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Discovery query cannot be empty.")
        run_id = self.artifacts.create_run(
            job_type="discovery",
            metadata={"query": clean_query},
        )
        self._announce_run(run_id)
        scout = await self.agent.scout(
            ScoutRequest(mode="discovery", query=clean_query)
        )
        self._record_scout(
            run_id,
            scout.candidates[0].source_url if scout.candidates else AUDIT_SOURCE_URL,
            scout,
        )
        if scout.state is not AgentRunState.SUCCEEDED:
            return self._finish_agent_outcome(run_id, AUDIT_SOURCE_URL, scout)
        if not scout.candidates:
            return self._finish_without_queue(
                run_id,
                AUDIT_SOURCE_URL,
                RunState.SUCCEEDED,
                RunOutcome.NO_CHANGE,
                "Search returned no candidates.",
            )

        registry_captures: dict[str, CapturedPage] = {}
        pages_captured = 0
        skipped_reasons: list[str] = []
        no_change_reasons: list[str] = []
        for candidate in scout.candidates[: self.budget.max_candidates_per_cycle]:
            if pages_captured >= self.budget.max_static_pages_per_job:
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.CAPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Static page capture budget was exhausted.",
                )
            try:
                captured = await self._capture(run_id, candidate.source_url)
            except PageFetchError as error:
                skipped_reasons.append(f"capture failed ({type(error).__name__})")
                continue
            pages_captured += 1
            if reason := blocked_page_reason(captured.snapshot):
                skipped_reasons.append(f"candidate page unusable ({reason})")
                continue

            trust = self.trust_policy.evaluate(captured.snapshot)
            if not trust.trusted:
                for registry_url in self.trust_policy.trusted_registry_urls:
                    if registry_url in registry_captures:
                        continue
                    if pages_captured >= self.budget.max_static_pages_per_job:
                        break
                    try:
                        registry_capture = await self._capture(
                            run_id, registry_url
                        )
                    except PageFetchError:
                        continue
                    if blocked_page_reason(registry_capture.snapshot) is not None:
                        continue
                    registry_captures[registry_url] = registry_capture
                    pages_captured += 1
                trust = self.trust_policy.evaluate(
                    captured.snapshot,
                    registry_snapshots=tuple(
                        item.snapshot for item in registry_captures.values()
                    ),
                )
            if not trust.trusted:
                skipped_reasons.append(trust.reason)
                continue

            evidence = [captured]
            if trust.registry_snapshot is not None:
                registry_capture = registry_captures.get(
                    trust.registry_snapshot.source_url
                )
                if registry_capture is None:
                    skipped_reasons.append("trusted registry artifact was unavailable")
                    continue
                evidence.append(registry_capture)
            assessment = await self.agent.assess(
                AssessmentRequest(
                    mode="discovery",
                    context=(FrozenContextField(name="query", value=clean_query),),
                    evidence=tuple(item.as_agent_evidence() for item in evidence),
                )
            )
            self._record_assessment(run_id, candidate.source_url, assessment)
            if assessment.state is not AgentRunState.SUCCEEDED or assessment.decision is None:
                return self._finish_agent_outcome(run_id, candidate.source_url, assessment)
            decision = assessment.decision
            required_references = tuple(item.reference for item in evidence)
            if decision.action is DecisionAction.NO_CHANGE:
                no_change_reasons.append(decision.summary)
                continue
            if decision.action is not DecisionAction.SUGGEST_EVENT or decision.candidate is None:
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    decision.summary,
                )
            if not self._valid_evidence(decision, required=required_references):
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Decision did not cite the complete captured trust evidence.",
                )
            if not _candidate_matches_capture(decision.candidate, captured.snapshot):
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Suggested event URL was not the captured trusted candidate.",
                )
            if not decision.candidate.distances:
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Suggested event profile had no supported distance.",
                )

            committed_status = ResearchRunStatus(
                status=RunState.SUCCEEDED,
                outcome=RunOutcome.SUGGESTION_CREATED,
            )
            if not self.persist_queue:
                return self._finish_without_queue(
                    run_id,
                    candidate.source_url,
                    RunState.SUCCEEDED,
                    RunOutcome.INCONCLUSIVE,
                    "Shadow mode validated an event candidate without writing the queue.",
                )
            prepared = self.artifacts.prepare_decision(
                run_id,
                source_url=candidate.source_url,
                decision=decision,
                committed_status=committed_status,
            )
            note = "\n".join(
                _moderation_evidence(
                    decision.summary,
                    tuple(evidence),
                    trust_reason=trust.reason,
                    prepared=prepared,
                )
            )
            try:
                admission = admit_suggestion(
                    EventSuggestionCreate(
                        event_name=decision.candidate.title,
                        url=decision.candidate.source_url,
                        event_date=decision.candidate.event_date,
                        location=decision.candidate.location,
                        region_tags=decision.candidate.region_tags,
                        distances=decision.candidate.distances,
                        note=note,
                        submitter_user_id=None,
                        submitter_username=None,
                        submitter_display_name=None,
                    ),
                    max_pending=self.budget.max_pending_suggestions,
                    database_url=self.database_url,
                )
            except Exception:
                return self._finish_rejected_admission(
                    prepared,
                    RunState.FAILED,
                    "Suggestion admission failed before commit.",
                )
            if admission.outcome != "admitted" or admission.suggestion is None:
                detail = {
                    "duplicate": "Candidate URL is already tracked or queued; duplicate skipped.",
                    "queue_full": "The researcher suggestion queue is full.",
                }[admission.outcome]
                return self._finish_rejected_admission(
                    prepared,
                    RunState.CAPPED if admission.outcome == "queue_full" else RunState.SKIPPED,
                    detail,
                    outcome=(
                        RunOutcome.INCONCLUSIVE
                        if admission.outcome == "queue_full"
                        else RunOutcome.NO_CHANGE
                    ),
                )

            queue_reference = f"event_suggestion:{admission.suggestion.id}"
            terminal = self.artifacts.finalize_committed(
                prepared,
                queue_reference=queue_reference,
            )
            return ResearchJobResult(
                run_id,
                committed_status,
                terminal,
                queue_reference=queue_reference,
            )

        if no_change_reasons:
            return self._finish_without_queue(
                run_id,
                scout.candidates[0].source_url,
                RunState.SUCCEEDED,
                RunOutcome.NO_CHANGE,
                "; ".join(no_change_reasons)[:1_000],
            )
        return self._finish_without_queue(
            run_id,
            scout.candidates[0].source_url,
            RunState.SKIPPED,
            RunOutcome.INCONCLUSIVE,
            "; ".join(skipped_reasons)[:1_000] or "No trusted candidate was captured.",
        )

    async def _capture(self, run_id: str, url: str) -> CapturedPage:
        try:
            snapshot = await self.fetch_snapshot(url)
        except PageFetchError:
            raise
        except Exception as error:
            raise PageFetchError("Page capture failed.") from error
        reference = self.artifacts.write_page_snapshot(run_id, snapshot)
        return CapturedPage(snapshot=snapshot, reference=reference)

    def _refresh_search_enabled(self) -> bool:
        return (
            self.budget.max_web_searches_per_job > 0
            and self.budget.max_static_pages_per_job > 1
        )

    def _record_scout(
        self,
        run_id: str,
        source_url: str,
        result: ScoutRunResult,
    ) -> ArtifactReference:
        return self.artifacts.write_artifact(
            run_id,
            artifact_type="search_summary",
            source_url=source_url,
            content={
                "state": result.state.value,
                "metadata": asdict(result.metadata),
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in result.candidates
                ],
                "error_code": result.error_code,
                "detail": result.detail,
            },
        )

    def _announce_run(self, run_id: str) -> None:
        if self.on_run_started is not None:
            self.on_run_started(run_id)

    def _record_assessment(
        self,
        run_id: str,
        source_url: str,
        result: AssessmentRunResult,
    ) -> ArtifactReference:
        return self.artifacts.write_artifact(
            run_id,
            artifact_type="assessment",
            source_url=source_url,
            content={
                "state": result.state.value,
                "metadata": asdict(result.metadata),
                "decision": (
                    result.decision.model_dump(mode="json")
                    if result.decision is not None
                    else None
                ),
                "error_code": result.error_code,
                "detail": result.detail,
            },
        )

    def _valid_evidence(
        self,
        decision: ResearchDecision,
        *,
        required: tuple[ArtifactReference, ...],
    ) -> bool:
        if not decision.evidence:
            return False
        required_set = set(required)
        decision_set = set(decision.evidence)
        if required_set != decision_set:
            return False
        try:
            for reference in decision.evidence:
                self.artifacts.verify_artifact(reference)
        except Exception:
            return False
        return True

    def _finish_agent_outcome(
        self,
        run_id: str,
        source_url: str,
        result: ScoutRunResult | AssessmentRunResult,
    ) -> ResearchJobResult:
        state = {
            AgentRunState.SUCCEEDED: RunState.SKIPPED,
            AgentRunState.FAILED: RunState.FAILED,
            AgentRunState.CAPPED: RunState.CAPPED,
            AgentRunState.INCONCLUSIVE: RunState.SKIPPED,
        }[result.state]
        return self._finish_without_queue(
            run_id,
            source_url,
            state,
            RunOutcome.INCONCLUSIVE,
            result.detail or result.error_code or "Agent returned no usable decision.",
        )

    def _finish_without_queue(
        self,
        run_id: str,
        source_url: str,
        state: RunState,
        outcome: RunOutcome,
        detail: str,
    ) -> ResearchJobResult:
        status = ResearchRunStatus(status=state, outcome=outcome, detail=detail[:1_000])
        terminal = self.artifacts.finalize_without_queue(
            run_id,
            source_url=source_url,
            status=status,
        )
        return ResearchJobResult(run_id, status, terminal)

    def _finish_rejected_admission(
        self,
        prepared: ArtifactReference,
        state: RunState,
        detail: str,
        *,
        outcome: RunOutcome = RunOutcome.INCONCLUSIVE,
    ) -> ResearchJobResult:
        status = ResearchRunStatus(status=state, outcome=outcome, detail=detail)
        terminal = self.artifacts.finalize_uncommitted(prepared, status=status)
        return ResearchJobResult(prepared.run_id, status, terminal)


@dataclass(frozen=True)
class CapturedPage:
    snapshot: PageSnapshot
    reference: ArtifactReference

    def as_agent_evidence(self) -> CapturedSnapshotEvidence:
        return CapturedSnapshotEvidence(
            reference=self.reference,
            final_url=self.snapshot.final_url,
            title=self.snapshot.title,
            fetched_at=self.snapshot.fetched_at,
            normalized_text=self.snapshot.normalized_text,
            text_hash=self.snapshot.text_hash,
        )


def _current_fields(event: TrackedEvent) -> dict[str, object]:
    return {
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at,
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at,
        "registration_url": event.registration_url,
        "event_date": event.event_date,
    }


def _event_context(event: TrackedEvent, approved_source_url: str) -> tuple[FrozenContextField, ...]:
    values = {
        "event_id": event.id,
        "name": event.name,
        "city": event.city,
        "country": event.country,
        "timezone": event.timezone,
        "distances": ", ".join(event.distances),
        "event_date": event.event_date or "",
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at or "",
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at or "",
        "registration_url": event.registration_url or "",
        "official_url": event.official_url,
        "approved_source_url": approved_source_url,
    }
    return tuple(FrozenContextField(name=name, value=str(value)) for name, value in values.items())


def _refresh_search_query(event: TrackedEvent, approved_source_url: str) -> str:
    year = event.event_date[:4] if event.event_date else "current"
    distances = " ".join(event.distances).replace("_", " ")
    query = (
        f'site:{source_domain(approved_source_url)} "{event.name}" {year} {distances} '
        "standard public registration lottery status opening closing dates"
    )
    return query[:500]


def _refresh_candidate_urls(
    candidates: tuple[ResearchCandidate, ...],
    *,
    event: TrackedEvent,
    approved_source_url: str,
) -> tuple[str, ...]:
    approved = normalize_url(approved_source_url)
    selected: list[str] = []
    selected_normalized: set[str] = {approved}
    for candidate in candidates:
        if not _same_source_domain(candidate.source_url, approved_source_url):
            continue
        if (
            event.event_date is not None
            and candidate.event_date is not None
            and candidate.event_date != event.event_date
        ):
            continue
        if conflicts_with_event_identity(
            candidate.source_url,
            f"{candidate.title} {candidate.snippet}",
            event.distances,
        ):
            continue
        normalized = normalize_url(candidate.source_url)
        if normalized in selected_normalized:
            continue
        selected.append(candidate.source_url)
        selected_normalized.add(normalized)
    return tuple(selected)


def _same_source_domain(candidate_url: str, approved_source_url: str) -> bool:
    return source_domain(candidate_url) == source_domain(approved_source_url)


def _changed_supported_fields(
    decision: ResearchDecision,
    current_fields: dict[str, object],
) -> dict[str, object]:
    if decision.proposed_fields is None:
        return {}
    proposed = decision.proposed_fields.model_dump(
        mode="json",
        exclude={"clear_fields"},
        exclude_none=True,
    )
    proposed.update({field: None for field in decision.proposed_fields.clear_fields})
    return {
        field: proposed[field]
        for field in SUPPORTED_UPDATE_FIELDS
        if field in proposed
        and (proposed[field] is None or proposed[field] not in {"", "unknown"})
        and proposed[field] != current_fields[field]
    }


def _normalized_proposed_changes(
    changed_fields: dict[str, object],
) -> ProposedEventChanges:
    return ProposedEventChanges.model_validate(
        {
            **{field: value for field, value in changed_fields.items() if value is not None},
            "clear_fields": tuple(
                field for field, value in changed_fields.items() if value is None
            ),
        }
    )


def _candidate_matches_capture(candidate: ResearchCandidate, snapshot: PageSnapshot) -> bool:
    candidate_url = normalize_url(candidate.source_url)
    return candidate_url in {
        normalize_url(snapshot.source_url),
        normalize_url(snapshot.final_url),
    }


def _moderation_evidence(
    summary: str,
    captures: tuple[CapturedPage, ...],
    *,
    trust_reason: str,
    prepared: ArtifactReference,
) -> tuple[str, ...]:
    lines = [
        f"Researcher worker: {summary[:500]}",
        f"Source check: {trust_reason[:120]}.",
        decision_queue_marker(prepared),
    ]
    lines.extend(
        "researcher-evidence:v1 "
        f"run={capture.reference.run_id} artifact={capture.reference.artifact_name} "
        f"sha256={capture.reference.content_hash[:12]} "
        f"source={capture.snapshot.final_url} "
        f"captured_at={capture.snapshot.fetched_at.isoformat()}"
        for capture in captures
    )
    return tuple(lines)


def decision_queue_marker(prepared: ArtifactReference) -> str:
    return (
        "researcher-decision:v1 "
        f"run={prepared.run_id} artifact={prepared.artifact_name} "
        f"sha256={prepared.content_hash}"
    )
