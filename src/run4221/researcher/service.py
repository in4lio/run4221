from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Protocol
from urllib.parse import urlparse

from pydantic import ValidationError

from run4221.db.models import RESEARCH_DECISION_MARKER_PREFIX
from run4221.db.research import (
    ProposedEventUpdateCreate,
    ResearchSourceRecord,
    admit_proposed_update,
    normalize_url,
)
from run4221.events import TrackedEvent
from run4221.ingestion.event_identity import (
    conflicts_with_event_identity,
    event_identity_tokens,
)
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageSnapshot,
    blocked_page_reason,
    fetch_enriched_page_snapshot,
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
from run4221.researcher.policy import source_domain
from run4221.researcher.schemas import (
    ArtifactReference,
    AssessmentVerdict,
    DecisionAction,
    EventProfileDraft,
    EventUpdateField,
    EvidenceRequest,
    EvidenceRequestPurpose,
    ProposedEventChanges,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
    ResearchRunStatus,
    RunOutcome,
    RunState,
)

AUDIT_SOURCE_URL = "https://run4221.invalid/researcher"
MAX_REFRESH_CONTINUATIONS = 2
MAX_REFRESH_CAPTURES = 4
MAX_REFRESH_WEB_SEARCHES = 2

_ALTERNATIVE_ENTRY_LINK_TERMS = frozenset(
    {
        "10k",
        "5k",
        "charity",
        "children",
        "extras",
        "handbike",
        "handbiker",
        "inline",
        "inlineskating",
        "jubilee",
        "junior",
        "kid",
        "kids",
        "mini",
        "relay",
        "school",
        "skating",
        "tour",
        "wheelchair",
        "youth",
    }
)

_REFRESH_LINK_TERM_WEIGHTS = {
    EvidenceRequestPurpose.REGISTRATION_STATUS: {
        "registration": 100,
        "register": 90,
        "status": 60,
        "information": 50,
        "info": 50,
        "results": 40,
        "lottery": 40,
        "entry": 35,
        "waitlist": 35,
        "sold": 35,
    },
    EvidenceRequestPurpose.REGISTRATION_TIMING: {
        "registration": 100,
        "register": 90,
        "deadline": 70,
        "dates": 60,
        "date": 60,
        "information": 50,
        "info": 50,
        "lottery": 40,
        "entry": 35,
    },
    EvidenceRequestPurpose.REGISTRATION_URL: {
        "registration": 100,
        "register": 90,
        "signup": 70,
        "application": 60,
        "entry": 50,
        "information": 40,
        "info": 40,
        "lottery": 25,
    },
    EvidenceRequestPurpose.EVENT_DATE: {
        "date": 100,
        "dates": 100,
        "calendar": 90,
        "schedule": 90,
        "information": 50,
        "info": 50,
        "event": 30,
        "race": 30,
    },
    EvidenceRequestPurpose.EVENT_IDENTITY: {
        "information": 70,
        "info": 70,
        "about": 60,
        "event": 50,
        "race": 40,
        "marathon": 30,
    },
    EvidenceRequestPurpose.EVENT_EDITION: {
        "edition": 100,
        "information": 60,
        "info": 60,
        "event": 50,
        "race": 40,
        "marathon": 30,
    },
    EvidenceRequestPurpose.DISTANCE_CATEGORY: {
        "distance": 100,
        "course": 80,
        "marathon": 60,
        "race": 40,
        "information": 30,
        "info": 30,
    },
    EvidenceRequestPurpose.CONFLICT_RESOLUTION: {
        "information": 70,
        "info": 70,
        "results": 60,
        "status": 60,
        "event": 40,
        "race": 40,
        "marathon": 30,
    },
}


class SnapshotFetcher(Protocol):
    def __call__(
        self,
        url: str,
        *,
        allowed_origin: str | None = None,
    ) -> Awaitable[PageSnapshot]: ...


class EnrichedSnapshotFetcher(Protocol):
    def __call__(
        self,
        url: str,
        *,
        max_linked_pages: int,
    ) -> Awaitable[PageSnapshot]: ...


class ResearchAgent(Protocol):
    async def scout(self, request: ScoutRequest) -> ScoutRunResult: ...

    async def assess(self, request: AssessmentRequest) -> AssessmentRunResult: ...


@dataclass(frozen=True)
class ResearchJobResult:
    run_id: str
    status: ResearchRunStatus
    terminal_reference: ArtifactReference
    queue_reference: str | None = None
    conflicting_update_id: int | None = None


@dataclass(frozen=True)
class ProfileJobResult:
    """One profile run's terminal truth: a cited draft or a typed failure."""

    run_id: str
    status: ResearchRunStatus
    terminal_reference: ArtifactReference
    draft: EventProfileDraft | None = None
    # True when the draft came from a page located via web search rather than
    # from the moderator-supplied URL; callers must surface that provenance.
    located: bool = False


class ResearcherService:
    """Deterministic bridge from captured evidence to proposal-only queues."""

    def __init__(
        self,
        *,
        database_url: str,
        artifacts: ResearchArtifactStore,
        agent: ResearchAgent,
        budget: ResearchBudget,
        fetch_snapshot: SnapshotFetcher = fetch_page_snapshot,
        fetch_enriched_snapshot: EnrichedSnapshotFetcher = fetch_enriched_page_snapshot,
        persist_queue: bool = True,
        on_run_started: Callable[[str], None] | None = None,
    ) -> None:
        self.database_url = database_url
        self.artifacts = artifacts
        self.agent = agent
        self.budget = budget
        self.fetch_snapshot = fetch_snapshot
        self.fetch_enriched_snapshot = fetch_enriched_snapshot
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
            async with asyncio.timeout(self.budget.max_wall_time_seconds_per_job):
                return await self._refresh_started(
                    run_id,
                    source,
                    frozen_event,
                    frozen_fields,
                )
        except TimeoutError:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.CAPPED,
                RunOutcome.INCONCLUSIVE,
                "Refresh wall-time budget was exhausted.",
            )
        except ValidationError:
            # Request construction failed (e.g. an oversized context value):
            # fail closed with a terminal artifact instead of leaking the
            # error and leaving the run directory without terminal.json.
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.FAILED,
                RunOutcome.INCONCLUSIVE,
                "Refresh request construction failed schema validation.",
            )

    async def _refresh_started(
        self,
        run_id: str,
        source: ResearchSourceRecord,
        frozen_event: TrackedEvent,
        frozen_fields: dict[str, object],
    ) -> ResearchJobResult:
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
        if not _same_source_domain(captured.snapshot.final_url, source.url):
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Approved source redirected outside its stored domain.",
            )

        captures = [captured]
        context = _event_context(frozen_event, source.url)
        captured_final_urls = {normalize_url(captured.snapshot.final_url)}
        completed_purposes: set[EvidenceRequestPurpose] = set()
        normalized_queries: set[str] = set()
        successful_captures = 1
        completed_web_searches = 0

        while True:
            assessment = await self.agent.assess(
                AssessmentRequest(
                    mode="refresh",
                    context=context,
                    evidence=tuple(item.as_agent_evidence() for item in captures),
                )
            )
            self._record_assessment(run_id, source.url, assessment)
            if assessment.state is not AgentRunState.SUCCEEDED:
                return self._finish_agent_outcome(run_id, source.url, assessment)
            if assessment.decision is not None:
                try:
                    decision = ResearchDecision.model_validate(
                        assessment.decision.model_dump(mode="python")
                    )
                except Exception:
                    return self._finish_without_queue(
                        run_id,
                        source.url,
                        RunState.SKIPPED,
                        RunOutcome.INCONCLUSIVE,
                        "Terminal decision failed host schema validation.",
                    )
                break

            evidence_request = assessment.evidence_request
            if not isinstance(evidence_request, EvidenceRequest):
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Assessment returned neither a terminal decision nor a valid evidence request.",
                )
            normalized_query = _normalize_refresh_query(evidence_request.query)
            purpose = evidence_request.purpose
            if len(completed_purposes) >= MAX_REFRESH_CONTINUATIONS:
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.CAPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Refresh continuation budget was exhausted.",
                )
            if normalized_query in normalized_queries or purpose in completed_purposes:
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Assessment repeated an already completed evidence query or purpose.",
                )
            capture_limit = min(
                MAX_REFRESH_CAPTURES,
                self.budget.max_static_pages_per_job,
            )
            if successful_captures >= capture_limit:
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.CAPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Refresh page-capture budget was exhausted.",
            )

            normalized_queries.add(normalized_query)
            linked_candidate_urls = _refresh_link_candidate_urls(
                tuple(captures),
                request=evidence_request,
                event=frozen_event,
                approved_source_url=source.url,
                captured_final_urls=frozenset(captured_final_urls),
            )
            candidate_urls = linked_candidate_urls
            if not candidate_urls:
                if completed_web_searches >= min(
                    MAX_REFRESH_WEB_SEARCHES,
                    self.budget.max_web_searches_per_job,
                ):
                    return self._finish_without_queue(
                        run_id,
                        source.url,
                        RunState.CAPPED,
                        RunOutcome.INCONCLUSIVE,
                        "Refresh web-search budget was exhausted.",
                    )
                scout = await self.agent.scout(
                    ScoutRequest(
                        mode="refresh",
                        query=_targeted_refresh_query(
                            frozen_event,
                            source.url,
                            evidence_request,
                        ),
                        approved_source_url=source.url,
                        context=context,
                    )
                )
                self._record_scout(run_id, source.url, scout)
                if scout.state is not AgentRunState.SUCCEEDED:
                    return self._finish_agent_outcome(run_id, source.url, scout)
                if scout.metadata.web_search_calls > 1:
                    return self._finish_without_queue(
                        run_id,
                        source.url,
                        RunState.CAPPED,
                        RunOutcome.INCONCLUSIVE,
                        "Refresh scout exceeded its one-search provider cap.",
                    )
                completed_web_searches += scout.metadata.web_search_calls
                candidate_urls = _refresh_candidate_urls(
                    scout.candidates,
                    event=frozen_event,
                    approved_source_url=source.url,
                )
            if not candidate_urls:
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "No same-domain qualified search or captured-link candidate URL was found.",
                )

            qualified: CapturedPage | None = None
            rejected_candidates: list[str] = []
            for candidate_url in candidate_urls:
                if successful_captures >= capture_limit:
                    return self._finish_without_queue(
                        run_id,
                        source.url,
                        RunState.CAPPED,
                        RunOutcome.INCONCLUSIVE,
                        "Refresh page-capture budget was exhausted.",
                    )
                try:
                    followup = await self._capture(
                        run_id,
                        candidate_url,
                        allowed_origin=source.url,
                    )
                except PageFetchError as error:
                    rejected_candidates.append(
                        f"capture failed ({type(error).__name__})"
                    )
                    continue
                successful_captures += 1
                if reason := blocked_page_reason(followup.snapshot):
                    rejected_candidates.append(f"page unusable ({reason})")
                    continue
                if not _same_source_domain(followup.snapshot.final_url, source.url):
                    rejected_candidates.append("cross-domain redirect")
                    continue
                normalized_final_url = normalize_url(followup.snapshot.final_url)
                if normalized_final_url in captured_final_urls:
                    rejected_candidates.append("duplicate final URL")
                    continue
                if _conflicts_with_tracked_event(followup.snapshot, frozen_event):
                    rejected_candidates.append("wrong event or distance/category")
                    continue
                if not _matches_tracked_event_identity(followup.snapshot, frozen_event):
                    rejected_candidates.append("wrong event identity or edition")
                    continue
                qualified = followup
                break
            if qualified is None:
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    (
                        "Targeted search produced no new qualified captured evidence: "
                        + "; ".join(rejected_candidates)[:700]
                    ),
                )

            captures.append(qualified)
            captured_final_urls.add(normalize_url(qualified.snapshot.final_url))
            completed_purposes.add(purpose)

        if decision.action in {DecisionAction.NO_CHANGE, DecisionAction.INCONCLUSIVE}:
            if decision.action is DecisionAction.NO_CHANGE and not self._valid_refresh_no_change(
                decision,
                captures=tuple(captures),
                event=frozen_event,
            ):
                return self._finish_without_queue(
                    run_id,
                    source.url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "No-change decision lacked confirmed applicable current-field evidence.",
                )
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
        if decision.action is not DecisionAction.PROPOSE_UPDATE:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Refresh decisions cannot create new-event suggestions.",
            )
        if not self._valid_cited_references(decision, captures=tuple(captures)):
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Update decision referenced uncaptured or unverifiable evidence.",
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
        if _proposes_stale_event_date(changed_fields, frozen_event):
            # Stale-edition guard: a page proposing a date strictly before the
            # stored event date is evidence about an older edition, never a
            # forward correction.
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Proposed event date is older than the stored event date.",
            )
        prepared_decision = self._validated_refresh_update(
            decision,
            captures=tuple(captures),
            event=frozen_event,
            changed_fields=changed_fields,
        )
        if prepared_decision is None:
            return self._finish_without_queue(
                run_id,
                source.url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Update decision failed captured support, applicability, or conflict validation.",
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
            producer_deadline_seconds=self.budget.max_wall_time_seconds_per_job,
        )
        cited_captures = _decision_captures(prepared_decision, tuple(captures))
        evidence_lines = _moderation_evidence(
            decision.summary,
            cited_captures,
            trust_reason=(
                "stored approved event source plus same-domain official captured evidence"
                if len(captures) > 1
                else "stored approved event source"
            ),
            prepared=prepared,
            decision=prepared_decision,
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
                conflicting_update_id=admission.conflicting_update_id,
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

    async def profile(self, url: str) -> ProfileJobResult:
        """Draft one cited event profile from one enriched capture; never touch queues."""

        run_id = self.artifacts.create_run(
            job_type="profile",
            metadata={"source_url": url},
        )
        self._announce_run(run_id)
        try:
            async with asyncio.timeout(self.budget.max_wall_time_seconds_per_profile_job):
                return await self._profile_started(run_id, url)
        except TimeoutError:
            return self._finish_profile(
                run_id,
                url,
                RunState.CAPPED,
                RunOutcome.INCONCLUSIVE,
                "Profile wall-time budget was exhausted.",
            )
        except ValidationError:
            # Request construction failed (e.g. a URL longer than the frozen
            # context allows): fail closed with a terminal artifact instead of
            # leaking the error and leaving the run directory unterminated.
            return self._finish_profile(
                run_id,
                url,
                RunState.FAILED,
                RunOutcome.INCONCLUSIVE,
                "Profile request construction failed schema validation.",
            )

    async def _profile_started(self, run_id: str, url: str) -> ProfileJobResult:
        # The root fetch plus every enrichment sub-fetch of BOTH captures must
        # stay inside the static-page budget for the whole job.
        first_linked_pages = max(0, self.budget.max_static_pages_per_job - 1)
        try:
            captured = await self._capture_enriched(
                run_id,
                url,
                max_linked_pages=first_linked_pages,
            )
        except PageFetchError as error:
            return self._finish_profile(
                run_id,
                url,
                RunState.FAILED,
                RunOutcome.INCONCLUSIVE,
                f"Profile page capture failed ({type(error).__name__}).",
            )
        if reason := blocked_page_reason(captured.snapshot):
            return self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                f"Profile page was unusable: {reason}.",
            )

        decision, failure = await self._profile_assessment(
            run_id,
            url,
            captured,
            context=(FrozenContextField(name="requested_url", value=url),),
            # The first assessment must leave room for a potential locate
            # scout plus a second assessment (mirrors the refresh reserve).
            reserve_continuation=True,
        )
        if failure is not None:
            return failure
        assert decision is not None
        if decision.page_is_event is not False:
            assert decision.draft is not None
            if not _draft_matches_capture(decision.draft, captured.snapshot):
                return self._finish_profile(
                    run_id,
                    url,
                    RunState.SKIPPED,
                    RunOutcome.INCONCLUSIVE,
                    "Draft URLs did not match the captured evidence.",
                )
            return self._finish_profile_draft(run_id, url, decision)

        # The captured page is not the event's own page: run exactly one
        # bounded locating search, capture the official page, assess once more.
        assert decision.draft is not None
        if self.budget.max_web_searches_per_job < 1:
            return self._finish_profile(
                run_id,
                url,
                RunState.CAPPED,
                RunOutcome.INCONCLUSIVE,
                "Profile locate-search budget was exhausted.",
            )
        context = _profile_locate_context(url, decision.draft)
        scout = await self.agent.scout(
            ScoutRequest(
                mode="profile",
                query=_locate_profile_query(decision.draft),
                context=context,
            )
        )
        self._record_scout(run_id, url, scout)
        if scout.state is not AgentRunState.SUCCEEDED:
            return self._finish_profile_agent_outcome(run_id, url, scout)
        if scout.metadata.web_search_calls > 1:
            return self._finish_profile(
                run_id,
                url,
                RunState.CAPPED,
                RunOutcome.INCONCLUSIVE,
                "Profile scout exceeded its one-search provider cap.",
            )
        official_url = _locate_candidate_url(
            scout.candidates,
            captured_urls=frozenset(
                {normalize_url(url), normalize_url(captured.snapshot.final_url)}
            ),
        )
        if official_url is None:
            return self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Locating search produced no new official-page candidate.",
            )
        try:
            # The locate capture may only enrich with whatever linked-page
            # allowance the first enriched capture left unspent.
            official = await self._capture_enriched(
                run_id,
                official_url,
                max_linked_pages=max(
                    0,
                    self.budget.max_static_pages_per_job - 1 - first_linked_pages,
                ),
            )
        except PageFetchError as error:
            return self._finish_profile(
                run_id,
                url,
                RunState.FAILED,
                RunOutcome.INCONCLUSIVE,
                f"Official page capture failed ({type(error).__name__}).",
            )
        if reason := blocked_page_reason(official.snapshot):
            return self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                f"Official page was unusable: {reason}.",
            )
        located_decision, failure = await self._profile_assessment(
            run_id,
            url,
            official,
            context=context,
        )
        if failure is not None:
            return failure
        assert located_decision is not None
        if located_decision.page_is_event is False:
            return self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "The located page still was not the event's own page.",
            )
        assert located_decision.draft is not None
        if not _draft_matches_capture(located_decision.draft, official.snapshot):
            return self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Draft URLs did not match the captured evidence.",
            )
        return self._finish_profile_draft(run_id, url, located_decision, located=True)

    async def _profile_assessment(
        self,
        run_id: str,
        url: str,
        capture: CapturedPage,
        *,
        context: tuple[FrozenContextField, ...],
        reserve_continuation: bool = False,
    ) -> tuple[ResearchDecision | None, ProfileJobResult | None]:
        assessment = await self.agent.assess(
            AssessmentRequest(
                mode="profile",
                context=context,
                evidence=(capture.as_agent_evidence(),),
                reserve_continuation=reserve_continuation,
            )
        )
        self._record_assessment(run_id, url, assessment)
        if assessment.state is not AgentRunState.SUCCEEDED or assessment.decision is None:
            return None, self._finish_profile_agent_outcome(run_id, url, assessment)
        try:
            decision = ResearchDecision.model_validate(
                assessment.decision.model_dump(mode="python")
            )
        except Exception:
            return None, self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Terminal decision failed host schema validation.",
            )
        if decision.action is DecisionAction.INCONCLUSIVE:
            return None, self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                decision.summary,
            )
        if decision.action is not DecisionAction.PROFILE_EVENT or decision.draft is None:
            return None, self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Profile assessments can only profile the captured page.",
            )
        if not self._valid_cited_references(decision, captures=(capture,)):
            return None, self._finish_profile(
                run_id,
                url,
                RunState.SKIPPED,
                RunOutcome.INCONCLUSIVE,
                "Profile decision cited uncaptured or unverifiable evidence.",
            )
        return decision, None

    async def _capture_enriched(
        self,
        run_id: str,
        url: str,
        *,
        max_linked_pages: int,
    ) -> CapturedPage:
        try:
            snapshot = await self.fetch_enriched_snapshot(
                url,
                max_linked_pages=max_linked_pages,
            )
        except PageFetchError:
            raise
        except Exception as error:
            raise PageFetchError("Page capture failed.") from error
        reference = self.artifacts.write_page_snapshot(run_id, snapshot)
        return CapturedPage(snapshot=snapshot, reference=reference)

    def _finish_profile(
        self,
        run_id: str,
        source_url: str,
        state: RunState,
        outcome: RunOutcome,
        detail: str,
    ) -> ProfileJobResult:
        status = ResearchRunStatus(status=state, outcome=outcome, detail=detail[:1_000])
        terminal = self.artifacts.finalize_without_queue(
            run_id,
            source_url=source_url,
            status=status,
        )
        return ProfileJobResult(run_id, status, terminal)

    def _finish_profile_draft(
        self,
        run_id: str,
        source_url: str,
        decision: ResearchDecision,
        *,
        located: bool = False,
    ) -> ProfileJobResult:
        status = ResearchRunStatus(
            status=RunState.SUCCEEDED,
            outcome=RunOutcome.PROFILE_COMPLETED,
            detail=decision.summary[:1_000],
        )
        terminal = self.artifacts.finalize_without_queue(
            run_id,
            source_url=source_url,
            status=status,
        )
        return ProfileJobResult(
            run_id,
            status,
            terminal,
            draft=decision.draft,
            located=located,
        )

    def _finish_profile_agent_outcome(
        self,
        run_id: str,
        source_url: str,
        result: ScoutRunResult | AssessmentRunResult,
    ) -> ProfileJobResult:
        return self._finish_profile(
            run_id,
            source_url,
            _AGENT_TERMINAL_STATES[result.state],
            RunOutcome.INCONCLUSIVE,
            result.detail or result.error_code or "Agent returned no usable decision.",
        )

    async def _capture(
        self,
        run_id: str,
        url: str,
        *,
        allowed_origin: str | None = None,
    ) -> CapturedPage:
        try:
            snapshot = await self.fetch_snapshot(url, allowed_origin=allowed_origin)
        except PageFetchError:
            raise
        except Exception as error:
            raise PageFetchError("Page capture failed.") from error
        reference = self.artifacts.write_page_snapshot(run_id, snapshot)
        return CapturedPage(snapshot=snapshot, reference=reference)

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
                    result.decision.model_dump(mode="json") if result.decision is not None else None
                ),
                "evidence_request": (
                    result.evidence_request.model_dump(mode="json")
                    if result.evidence_request is not None
                    else None
                ),
                "error_code": result.error_code,
                "detail": result.detail,
            },
        )

    def _valid_refresh_no_change(
        self,
        decision: ResearchDecision,
        *,
        captures: tuple[CapturedPage, ...],
        event: TrackedEvent,
    ) -> bool:
        if decision.conflicts or not self._valid_cited_references(decision, captures=captures):
            return False
        cited = set(decision.evidence)
        qualified = _qualified_capture_references(captures, event)
        return any(
            item.evidence in cited
            and item.evidence in qualified
            and item.event_identity is AssessmentVerdict.CONFIRMED
            and item.event_edition is AssessmentVerdict.CONFIRMED
            and item.distance_category is AssessmentVerdict.CONFIRMED
            and bool(item.applicable_fields)
            for item in decision.applicability
        )

    def _validated_refresh_update(
        self,
        decision: ResearchDecision,
        *,
        captures: tuple[CapturedPage, ...],
        event: TrackedEvent,
        changed_fields: dict[str, object],
    ) -> ResearchDecision | None:
        retained_support = tuple(
            item for item in decision.field_support if item.field.value in changed_fields
        )
        if set(item.field.value for item in retained_support) != set(changed_fields):
            return None
        cited = set(decision.evidence)
        if not cited or any(
            not set(item.evidence).issubset(cited) for item in retained_support
        ):
            return None
        qualified = _qualified_capture_references(captures, event)
        if any(not set(item.evidence).issubset(qualified) for item in retained_support):
            return None
        try:
            return ResearchDecision.model_validate(
                {
                    **decision.model_dump(mode="python"),
                    "proposed_fields": _normalized_proposed_changes(changed_fields),
                    "field_support": retained_support,
                }
            )
        except Exception:
            return None

    def _valid_cited_references(
        self,
        decision: ResearchDecision,
        *,
        captures: tuple[CapturedPage, ...],
    ) -> bool:
        captured_references = {item.reference for item in captures}
        referenced = set(decision.evidence)
        referenced.update(item.evidence for item in decision.applicability)
        for support in decision.field_support:
            referenced.update(support.evidence)
        for conflict in decision.conflicts:
            referenced.update(conflict.evidence)
        if not referenced or not referenced.issubset(captured_references):
            return False
        try:
            for reference in referenced:
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
        return self._finish_without_queue(
            run_id,
            source_url,
            _AGENT_TERMINAL_STATES[result.state],
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
        conflicting_update_id: int | None = None,
    ) -> ResearchJobResult:
        status = ResearchRunStatus(status=state, outcome=outcome, detail=detail)
        terminal = self.artifacts.finalize_uncommitted(prepared, status=status)
        return ResearchJobResult(
            prepared.run_id,
            status,
            terminal,
            conflicting_update_id=conflicting_update_id,
        )


_AGENT_TERMINAL_STATES = {
    AgentRunState.SUCCEEDED: RunState.SKIPPED,
    AgentRunState.FAILED: RunState.FAILED,
    AgentRunState.CAPPED: RunState.CAPPED,
    AgentRunState.INCONCLUSIVE: RunState.SKIPPED,
}


def _locate_profile_query(draft: EventProfileDraft) -> str:
    year = draft.event_date[:4] if draft.event_date else ""
    parts = (f'"{draft.name}"', draft.city, draft.country, year, "official event website")
    return " ".join(part for part in parts if part)[:500]


def _profile_locate_context(
    url: str,
    draft: EventProfileDraft,
) -> tuple[FrozenContextField, ...]:
    values = {
        "requested_url": url,
        "name": draft.name,
        "city": draft.city,
        "country": draft.country,
        "event_date": draft.event_date or "",
        "distances": ", ".join(draft.distances),
    }
    return tuple(FrozenContextField(name=name, value=str(value)) for name, value in values.items())


def _draft_matches_capture(draft: EventProfileDraft, snapshot: PageSnapshot) -> bool:
    """A returned draft must describe the page this run actually captured.

    Mirrors the deleted ``_candidate_matches_capture``: the draft's source URL
    must be the captured page (requested or final URL), and an optional
    official URL must stay on the captured page's domain.
    """

    captured_urls = {
        normalize_url(snapshot.source_url),
        normalize_url(snapshot.final_url),
    }
    if normalize_url(draft.source_url) not in captured_urls:
        return False
    return draft.official_url is None or _same_source_domain(
        draft.official_url,
        snapshot.final_url,
    )


def _proposes_stale_event_date(
    changed_fields: dict[str, object],
    event: TrackedEvent,
) -> bool:
    proposed = changed_fields.get("event_date")
    return (
        isinstance(proposed, str)
        and event.event_date is not None
        and proposed < event.event_date
    )


def _locate_candidate_url(
    candidates: tuple[ResearchCandidate, ...],
    *,
    captured_urls: frozenset[str],
) -> str | None:
    for candidate in candidates:
        if normalize_url(candidate.source_url) in captured_urls:
            continue
        return candidate.source_url
    return None


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
            primary_text=self.snapshot.primary_text,
            chrome_text=self.snapshot.chrome_text,
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


def _targeted_refresh_query(
    event: TrackedEvent,
    approved_source_url: str,
    request: EvidenceRequest,
) -> str:
    year = event.event_date[:4] if event.event_date else "current"
    distances = " ".join(event.distances).replace("_", " ")
    requested_query = " ".join(request.query.split())[:180]
    gap = " ".join(request.gap.split())[:160]
    query = (
        f'site:{source_domain(approved_source_url)} "{event.name}" {year} {distances} '
        f"{request.purpose.value} {requested_query} {gap}"
    )
    return query[:500]


def _normalize_refresh_query(query: str) -> str:
    return " ".join(query.casefold().split())


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
            "",
            event.distances,
        ):
            continue
        if event_identity_tokens(candidate.source_url, "") & _ALTERNATIVE_ENTRY_LINK_TERMS:
            continue
        normalized = normalize_url(candidate.source_url)
        if normalized in selected_normalized:
            continue
        selected.append(candidate.source_url)
        selected_normalized.add(normalized)
    return tuple(selected)


def _refresh_link_candidate_urls(
    captures: tuple[CapturedPage, ...],
    *,
    request: EvidenceRequest,
    event: TrackedEvent,
    approved_source_url: str,
    captured_final_urls: frozenset[str],
) -> tuple[str, ...]:
    weights = _REFRESH_LINK_TERM_WEIGHTS[request.purpose]
    ranked: list[tuple[int, int, str]] = []
    seen = set(captured_final_urls)
    order = 0
    for capture in captures:
        for link in capture.snapshot.links:
            order += 1
            if not _same_source_domain(link.url, approved_source_url):
                continue
            normalized = normalize_url(link.url)
            if normalized in seen:
                continue
            tokens = event_identity_tokens(link.url, link.text)
            if tokens & _ALTERNATIVE_ENTRY_LINK_TERMS:
                continue
            if conflicts_with_event_identity(link.url, link.text, event.distances):
                continue
            score = sum(weight for term, weight in weights.items() if term in tokens)
            if score <= 0:
                continue
            path = urlparse(link.url).path.casefold()
            if "/registration/" in path or "/register/" in path:
                score += 30
            ranked.append((-score, order, link.url))
            seen.add(normalized)
    ranked.sort()
    return tuple(url for _, _, url in ranked)


def _same_source_domain(candidate_url: str, approved_source_url: str) -> bool:
    return source_domain(candidate_url) == source_domain(approved_source_url)


_GENERIC_EVENT_IDENTITY_TERMS = frozenset(
    {
        "event",
        "events",
        "lauf",
        "marathon",
        "race",
        "registration",
        "run",
        "running",
    }
)


def _matches_tracked_event_identity(snapshot: PageSnapshot, event: TrackedEvent) -> bool:
    searchable = f"{snapshot.title or ''} {_snapshot_primary_text(snapshot)[:2_000]}"
    captured_tokens = event_identity_tokens(snapshot.final_url, searchable)
    expected_tokens = event_identity_tokens("", event.name) - _GENERIC_EVENT_IDENTITY_TERMS
    event_year = event.event_date[:4] if event.event_date else None
    explicit_identity_years = _explicit_identity_years(snapshot)
    if (
        event_year is not None
        and explicit_identity_years
        and event_year not in explicit_identity_years
    ):
        return False
    if expected_tokens:
        required_name_tokens = min(2, len(expected_tokens))
        return len(expected_tokens & captured_tokens) >= required_name_tokens

    captured_years = frozenset(re.findall(r"\b20\d{2}\b", searchable))
    return event_year is not None and event_year in captured_years


def _conflicts_with_tracked_event(snapshot: PageSnapshot, event: TrackedEvent) -> bool:
    return bool(
        event_identity_tokens(snapshot.final_url, "") & _ALTERNATIVE_ENTRY_LINK_TERMS
    ) or conflicts_with_event_identity(
        snapshot.final_url,
        f"{snapshot.title or ''} {_snapshot_primary_text(snapshot)[:2_000]}",
        event.distances,
    )


def _snapshot_primary_text(snapshot: PageSnapshot) -> str:
    return snapshot.primary_text or snapshot.normalized_text


def _qualified_capture_references(
    captures: tuple[CapturedPage, ...],
    event: TrackedEvent,
) -> frozenset[ArtifactReference]:
    return frozenset(
        capture.reference
        for capture in captures
        if not _conflicts_with_tracked_event(capture.snapshot, event)
        and _matches_tracked_event_identity(capture.snapshot, event)
    )


def _explicit_identity_years(snapshot: PageSnapshot) -> frozenset[str]:
    years = re.findall(r"\b20\d{2}\b", f"{snapshot.final_url} {snapshot.title or ''}")
    primary_text = _snapshot_primary_text(snapshot)[:2_000]
    for pattern in (
        r"\b(20\d{2})\s+(?:edition|event|race)\b",
        r"\b(?:edition|event|race)\s+(?:of\s+)?(20\d{2})\b",
    ):
        years.extend(re.findall(pattern, primary_text, flags=re.IGNORECASE))
    return frozenset(years)


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
        field.value: proposed[field.value]
        for field in EventUpdateField
        if field in decision.proposed_fields.changed_fields
        and (proposed[field.value] is None or proposed[field.value] not in {"", "unknown"})
        and proposed[field.value] != current_fields[field.value]
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


def _decision_captures(
    decision: ResearchDecision,
    captures: tuple[CapturedPage, ...],
) -> tuple[CapturedPage, ...]:
    referenced = set(decision.evidence)
    for support in decision.field_support:
        referenced.update(support.evidence)
    for conflict in decision.conflicts:
        referenced.update(conflict.evidence)
    return tuple(capture for capture in captures if capture.reference in referenced)


def _moderation_evidence(
    summary: str,
    captures: tuple[CapturedPage, ...],
    *,
    trust_reason: str,
    prepared: ArtifactReference,
    decision: ResearchDecision | None = None,
) -> tuple[str, ...]:
    lines = [
        f"Researcher worker: {_single_line(summary, 500)}",
        f"Source check: {_single_line(trust_reason, 120)}.",
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
    if decision is not None:
        lines.extend(_field_support_evidence_lines(decision))
        lines.extend(_conflict_evidence_lines(decision))
    return tuple(lines)


def _field_support_evidence_lines(decision: ResearchDecision) -> tuple[str, ...]:
    return tuple(
        "Researcher field support: "
        f"{support.field.value} <- "
        + ", ".join(
            f"{reference.artifact_name}#{reference.content_hash[:12]}"
            for reference in support.evidence
        )
        for support in decision.field_support
    )


def _conflict_evidence_lines(decision: ResearchDecision) -> tuple[str, ...]:
    return tuple(
        "Researcher conflict: "
        f"{conflict.field.value if conflict.field is not None else 'general'} <- "
        + ", ".join(
            f"{reference.artifact_name}#{reference.content_hash[:12]}"
            for reference in conflict.evidence
        )
        + f" | {_single_line(conflict.summary, 300)}"
        for conflict in decision.conflicts
    )


def _single_line(value: str, max_length: int) -> str:
    return " ".join(value.split())[:max_length]


def decision_queue_marker(prepared: ArtifactReference) -> str:
    return (
        RESEARCH_DECISION_MARKER_PREFIX
        + f"run={prepared.run_id} artifact={prepared.artifact_name} "
        f"sha256={prepared.content_hash}"
    )
