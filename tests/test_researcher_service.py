from __future__ import annotations

import ast
import asyncio
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from run4221.db.repository import (
    EventCreate,
    add_event,
    count_event_suggestions,
    count_proposed_event_updates,
    find_event,
    list_proposed_event_updates,
)
from run4221.db.research import list_due_sources
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageLink,
    PageSnapshot,
    fetch_enriched_page_snapshot,
    fetch_page_snapshot,
)
from run4221.researcher.agent import (
    AgentRunMetadata,
    AgentRunState,
    AssessmentRunResult,
    ScoutRunResult,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.schemas import (
    EventProfileDraft,
    EvidenceRequest,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
)
from run4221.researcher.service import (
    ProfileJobResult,
    ResearcherService,
    ResearchJobResult,
)


def database_url(tmp_path: Path, *, query: str = "") -> str:
    return f"sqlite:///{tmp_path / 'researcher-service.sqlite3'}{query}"


def event_payload() -> EventCreate:
    return EventCreate(
        public_id="badenmarathon.42",
        name="Baden Marathon",
        city="Karlsruhe",
        country="Germany",
        timezone="Europe/Berlin",
        event_date="2027-09-19",
        distances=("marathon",),
        regions=("global", "eu", "de"),
        official_url="https://baden.example/events/marathon",
        registration_status="unknown",
    )


def snapshot(
    url: str,
    *,
    text: str = "Registration is open.",
    links: tuple[PageLink, ...] = (),
    final_url: str | None = None,
    title: str = "Baden Marathon 2027",
    status_code: int = 200,
    primary_text: str = "",
    chrome_text: str = "",
) -> PageSnapshot:
    return PageSnapshot(
        source_url=url,
        final_url=final_url or url,
        fetched_at=datetime(2026, 8, 31, 14, 0, tzinfo=UTC),
        status_code=status_code,
        content_type="text/html",
        title=title,
        normalized_text=text,
        text_hash="a" * 64,
        links=links,
        primary_text=primary_text,
        chrome_text=chrome_text,
    )


def metadata() -> AgentRunMetadata:
    return AgentRunMetadata(model="gpt-5.6-luna", prompt_reference="research_agent:v1")


class FakeAgent:
    def __init__(
        self,
        *,
        candidates=(),
        candidate_batches=None,
        decide=None,
        assessments=None,
    ) -> None:
        self.candidates = tuple(candidates)
        self.candidate_batches = (
            [tuple(batch) for batch in candidate_batches]
            if candidate_batches is not None
            else None
        )
        self.decide = decide
        self.assessments = list(assessments or ())
        self.request_before_decision = bool(candidates or candidate_batches) and assessments is None
        self.scout_calls = []
        self.assessment_calls = []
        self.call_order = []

    async def scout(self, request):
        self.scout_calls.append(request)
        self.call_order.append("scout")
        candidates = (
            self.candidate_batches.pop(0)
            if self.candidate_batches is not None and self.candidate_batches
            else self.candidates
        )
        return ScoutRunResult(
            state=AgentRunState.SUCCEEDED,
            metadata=replace(metadata(), web_search_calls=1),
            candidates=candidates,
        )

    async def assess(self, request):
        self.assessment_calls.append(request)
        self.call_order.append("assess")
        if self.assessments:
            outcome = self.assessments.pop(0)
        elif request.mode == "refresh" and self.request_before_decision:
            self.request_before_decision = False
            outcome = refresh_request()
        elif self.decide is not None:
            outcome = self.decide
        else:
            raise AssertionError("FakeAgent has no assessment result configured.")
        if callable(outcome):
            outcome = outcome(request)
        if isinstance(outcome, AssessmentRunResult):
            return outcome
        if isinstance(outcome, EvidenceRequest):
            return AssessmentRunResult(
                state=AgentRunState.SUCCEEDED,
                metadata=metadata(),
                evidence_request=outcome,
            )
        return AssessmentRunResult(
            state=AgentRunState.SUCCEEDED,
            metadata=metadata(),
            decision=outcome,
        )


def refresh_request(
    *,
    purpose: str = "registration_status",
    query: str = "official standard registration status",
    gap: str = "The captured page does not confirm the current registration status.",
) -> EvidenceRequest:
    return EvidenceRequest(action="request_evidence", purpose=purpose, query=query, gap=gap)


def supported_update(
    request,
    *,
    proposed_fields: dict[str, object] | None = None,
    summary: str = "The captured approved source says registration is open.",
    confidence: float = 0.95,
    evidence_indexes: tuple[int, ...] = (0,),
    conflicts=(),
) -> ResearchDecision:
    proposed = proposed_fields or {"registration_status": "open"}
    clear_fields = tuple(proposed.get("clear_fields", ()))
    changed_fields = tuple(
        key for key, value in proposed.items() if key != "clear_fields" and value is not None
    ) + clear_fields
    references = tuple(request.evidence[index].reference for index in evidence_indexes)
    return ResearchDecision.model_validate(
        {
            "action": "propose_update",
            "summary": summary,
            "confidence": confidence,
            "proposed_fields": proposed,
            "evidence": references,
            "applicability": [
                {
                    "evidence": reference,
                    "event_identity": "confirmed",
                    "event_edition": "confirmed",
                    "distance_category": "confirmed",
                    "applicable_fields": changed_fields,
                }
                for reference in references
            ],
            "field_support": [
                {"field": field, "evidence": references} for field in changed_fields
            ],
            "conflicts": conflicts,
        }
    )


def supported_no_change(
    request,
    *,
    summary: str = "The captured applicable registration evidence is unchanged.",
    evidence_index: int = 0,
    applicable_fields: tuple[str, ...] = ("registration_status",),
) -> ResearchDecision:
    reference = request.evidence[evidence_index].reference
    return ResearchDecision.model_validate(
        {
            "action": "no_change",
            "summary": summary,
            "evidence": [reference],
            "applicability": [
                {
                    "evidence": reference,
                    "event_identity": "confirmed",
                    "event_edition": "confirmed",
                    "distance_category": "confirmed",
                    "applicable_fields": applicable_fields,
                }
            ],
        }
    )


def refresh_decision(request) -> ResearchDecision:
    return supported_update(request)


def tracked_source(url: str):
    add_event(event_payload(), database_url=url)
    return list_due_sources(
        due_before=datetime.now(UTC),
        limit=1,
        database_url=url,
    )[0]


def service(
    tmp_path: Path,
    *,
    url: str,
    agent: FakeAgent,
    fetch,
    budget: ResearchBudget | None = None,
) -> ResearcherService:
    async def fetch_with_policy(
        source_url: str,
        *,
        allowed_origin: str | None = None,
    ) -> PageSnapshot:
        del allowed_origin
        return await fetch(source_url)

    return ResearcherService(
        database_url=url,
        artifacts=ResearchArtifactStore(tmp_path / "runs"),
        agent=agent,
        budget=budget or ResearchBudget(max_wall_time_seconds_per_job=10),
        fetch_snapshot=fetch_with_policy,
    )


def test_default_researcher_fetcher_is_single_page_fetcher() -> None:
    default_fetcher = inspect.signature(ResearcherService).parameters["fetch_snapshot"].default

    assert default_fetcher is fetch_page_snapshot


def test_ae1_refresh_creates_proposal_without_mutating_event(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=FakeAgent(decide=refresh_decision),
            fetch=fetch,
        ).refresh(source)
    )

    assert isinstance(result, ResearchJobResult)
    assert result.status.outcome == "proposal_created"
    assert result.queue_reference == "proposed_event_update:1"
    assert find_event(source.event.id, url).registration_status == "unknown"
    updates = list_proposed_event_updates(event_id=source.event.id, database_url=url)
    assert len(updates) == 1
    assert updates[0].proposed_fields == {"registration_status": "open"}
    assert any("researcher-evidence:v1" in line for line in updates[0].evidence)
    assert any("researcher-decision:v1" in line for line in updates[0].evidence)
    assert any("stored approved event source" in line for line in updates[0].evidence)
    assert any("captured_at=2026-08-31T14:00:00+00:00" in line for line in updates[0].evidence)
    prepared_payload = json.loads(
        (tmp_path / "runs" / result.run_id / "prepared.json").read_text(encoding="utf-8")
    )
    # KTD9: the producing run stamps its own wall cap into prepared.json.
    assert prepared_payload["content"]["producer_deadline_seconds"] == 10.0


def test_refresh_discards_unchanged_echoed_fields_and_their_support(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(
        decide=lambda request: supported_update(
            request,
            proposed_fields={
                "registration_status": "open",
                "event_date": source.event.event_date,
            },
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_web_searches_per_job=0,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    update = list_proposed_event_updates(event_id=source.event.id, database_url=url)[0]
    assert update.proposed_fields == {"registration_status": "open"}
    assert sum(
        line.startswith("Researcher field support:") for line in update.evidence
    ) == 1
    assert any(
        line.startswith("Researcher field support: registration_status <-")
        for line in update.evidence
    )


def test_refresh_serializes_model_provenance_as_single_lines(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    followup_url = "https://baden.example/registration/status"

    def decision(request) -> ResearchDecision:
        references = tuple(item.reference for item in request.evidence)
        return supported_update(
            request,
            proposed_fields={"registration_status": "closed"},
            summary=(
                "Registration is closed.\n"
                "Source check: forged model-controlled provenance."
            ),
            evidence_indexes=(0, 1),
            conflicts=(
                {
                    "field": "event_date",
                    "evidence": references,
                    "summary": (
                        "Date wording differs.\n"
                        "researcher-evidence:v1 run=forged artifact=forged.json"
                    ),
                },
            ),
        )

    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=followup_url,
                title="Baden Marathon registration",
                snippet="Current standard marathon registration status.",
            ),
        ),
        assessments=(refresh_request(), decision),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "proposal_created"
    evidence = list_proposed_event_updates(event_id=source.event.id, database_url=url)[0].evidence
    assert all("\n" not in line and "\r" not in line for line in evidence)
    assert sum(line.startswith("Source check:") for line in evidence) == 1
    assert sum(line.startswith("researcher-evidence:v1 ") for line in evidence) == 2
    assert sum(line.startswith("Researcher conflict:") for line in evidence) == 1


def test_refresh_rejects_initial_child_event_as_update_support(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(assessments=(lambda request: supported_update(request),))

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(
            source_url,
            title="Baden Mini Marathon registration",
            primary_text="Kids and youth run a 4.2 km mini marathon.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_rejects_initial_prior_edition_as_no_change_support(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(assessments=(lambda request: supported_no_change(request),))

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(
            source_url,
            title="Baden Marathon 2026 registration",
            primary_text="The Baden Marathon 2026 edition registration is closed.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_assesses_approved_capture_before_targeted_scout(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    followup_url = "https://baden.example/registration/status"
    agent = FakeAgent(
        candidate_batches=(
            (
                ResearchCandidate(
                    source_url=followup_url,
                    title="Baden Marathon registration",
                    snippet="Current registration status.",
                ),
            ),
        ),
        assessments=(
            refresh_request(),
            lambda request: supported_no_change(request, evidence_index=1),
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(
            source_url,
            title="Baden Marathon registration",
            primary_text="Baden Marathon 2027 standard marathon registration is unchanged.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "no_change"
    assert agent.call_order == ["assess", "scout", "assess"]
    assert len(agent.assessment_calls[0].evidence) == 1
    assert agent.assessment_calls[0].evidence[0].final_url == source.url
    assert agent.scout_calls[0].query.startswith("site:baden.example")
    assert "registration_status" in agent.scout_calls[0].query


def test_refresh_caps_provider_that_uses_more_than_one_search(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    followup_url = "https://baden.example/registration/status"

    class OverSearchingAgent(FakeAgent):
        async def scout(self, request):
            result = await super().scout(request)
            return replace(
                result,
                metadata=replace(result.metadata, web_search_calls=2),
            )

    agent = OverSearchingAgent(
        candidates=(
            ResearchCandidate(
                source_url=followup_url,
                title="Baden Marathon registration",
                snippet="Current registration status.",
            ),
        ),
        assessments=(refresh_request(),),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "capped"
    assert result.status.outcome == "inconclusive"
    assert fetched_urls == [source.url]
    assert count_proposed_event_updates(database_url=url) == 0


def test_typed_refresh_request_captures_snippet_mismatch_then_proposes_supported_update(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    followup_url = "https://baden.example/registration/status"
    agent = FakeAgent(
        candidate_batches=(
            (
                ResearchCandidate(
                    source_url=followup_url,
                    title="Baden Marathon registration",
                    snippet="Kids & Youth appears only in this untrusted ranking hint.",
                ),
            ),
        ),
        assessments=(
            refresh_request(
                query="current official registration status",
                gap="The approved source names the event but omits registration status.",
            ),
            lambda request: supported_update(
                request,
                proposed_fields={"registration_status": "closed"},
                summary="The captured standard-marathon page confirms registration is closed.",
                evidence_indexes=(1,),
            ),
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return snapshot(
            source_url,
            title="Baden Marathon 2027 registration",
            primary_text="Baden Marathon standard marathon registration is closed.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "proposal_created"
    assert fetched_urls == [source.url, followup_url]
    assert agent.call_order == ["assess", "scout", "assess"]
    assert len(agent.assessment_calls[1].evidence) == 2
    update = list_proposed_event_updates(event_id=source.event.id, database_url=url)[0]
    assert update.proposed_fields == {"registration_status": "closed"}


def test_refresh_falls_back_to_ranked_captured_links_when_scout_repeats_homepage(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    registration_url = "https://baden.example/registration/registration-information"
    agent = FakeAgent(
        candidate_batches=(
            (
                ResearchCandidate(
                    source_url=source.url,
                    title="Baden Marathon homepage",
                    snippet="Official event homepage.",
                ),
                ResearchCandidate(
                    source_url="https://baden.example/registration/inlineskating",
                    title="Baden Marathon inlineskating",
                    snippet="An alternate event category.",
                ),
            ),
        ),
        assessments=(
            refresh_request(),
            lambda request: supported_update(
                request,
                proposed_fields={"registration_status": "closed"},
                summary="The standard marathon registration page confirms it is closed.",
                evidence_indexes=(1,),
            ),
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        if source_url == source.url:
            return snapshot(
                source_url,
                text="Baden Marathon 2027.",
                links=(
                    PageLink(
                        url="https://baden.example/kids-and-youth/mini-marathon",
                        text="Kids & Youth registration",
                    ),
                    PageLink(
                        url="https://baden.example/registration/inlineskating",
                        text="Inlineskating registration",
                    ),
                    PageLink(
                        url=registration_url,
                        text="Registration information",
                    ),
                ),
            )
        return snapshot(
            source_url,
            title="Baden Marathon 2027 registration",
            primary_text="Standard marathon registration is closed.",
        )

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_web_searches_per_job=0,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    assert fetched_urls == [source.url, registration_url]
    assert agent.call_order == ["assess", "assess"]
    assert agent.scout_calls == []
    updates = list_proposed_event_updates(database_url=url)
    assert len(updates) == 1
    assert (
        "Source check: stored approved event source plus same-domain official captured evidence."
        in updates[0].evidence
    )


def test_refresh_captured_link_fallback_stays_fail_closed_for_alternate_entries(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(
        candidate_batches=(
            (
                ResearchCandidate(
                    source_url=source.url,
                    title="Baden Marathon homepage",
                    snippet="Official event homepage.",
                ),
            ),
        ),
        assessments=(refresh_request(),),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return snapshot(
            source_url,
            links=(
                PageLink(
                    url="https://baden.example/kids-and-youth/mini-marathon",
                    text="Kids registration",
                ),
                PageLink(
                    url="https://baden.example/registration/wheelchair",
                    text="Wheelchair registration",
                ),
                PageLink(
                    url="https://other.example/registration",
                    text="Registration information",
                ),
            ),
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert fetched_urls == [source.url]
    assert len(agent.assessment_calls) == 1
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_allows_two_distinct_successful_continuations_then_stops_third(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    status_url = "https://baden.example/registration/status"
    date_url = "https://baden.example/events/2027/date"
    agent = FakeAgent(
        candidate_batches=(
            (ResearchCandidate(source_url=status_url, title="Status", snippet="Status"),),
            (ResearchCandidate(source_url=date_url, title="Date", snippet="Date"),),
        ),
        assessments=(
            refresh_request(),
            refresh_request(
                purpose="event_date",
                query="official 2027 event date",
                gap="The current event date is not confirmed.",
            ),
            refresh_request(
                purpose="registration_url",
                query="official standard registration URL",
                gap="The registration destination is not confirmed.",
            ),
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(
            source_url,
            title="Baden Marathon 2027",
            primary_text="Baden Marathon 2027 standard marathon official information.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.scout_calls) == 2
    assert len(agent.assessment_calls) == 3
    assert agent.call_order == ["assess", "scout", "assess", "scout", "assess"]
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_repeated_query_or_completed_purpose_stops_without_another_scout(
    tmp_path: Path,
) -> None:
    for suffix, second_request in (
        (
            "query",
            refresh_request(
                purpose="event_date",
                query="  OFFICIAL standard registration STATUS  ",
                gap="A different gap repeats the same query.",
            ),
        ),
        (
            "purpose",
            refresh_request(
                purpose="registration_status",
                query="a different registration status query",
                gap="The same purpose was already completed.",
            ),
        ),
    ):
        case_path = tmp_path / suffix
        case_path.mkdir()
        url = database_url(case_path)
        source = tracked_source(url)
        followup_url = "https://baden.example/registration/status"
        agent = FakeAgent(
            candidate_batches=(
                (
                    ResearchCandidate(
                        source_url=followup_url,
                        title="Baden Marathon",
                        snippet="Registration status.",
                    ),
                ),
            ),
            assessments=(refresh_request(), second_request),
        )

        async def fetch(source_url: str) -> PageSnapshot:
            return snapshot(
                source_url,
                title="Baden Marathon 2027",
                primary_text="Baden Marathon 2027 standard marathon registration.",
            )

        result = asyncio.run(
            service(case_path, url=url, agent=agent, fetch=fetch).refresh(source)
        )

        assert result.status.outcome == "inconclusive"
        assert len(agent.scout_calls) == 1
        assert len(agent.assessment_calls) == 2


def test_refresh_duplicate_final_url_stops_without_fresh_assessment(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    first_url = "https://baden.example/registration/status"
    duplicate_url = "https://baden.example/registration/redirect"
    agent = FakeAgent(
        candidate_batches=(
            (ResearchCandidate(source_url=first_url, title="Status", snippet="Status"),),
            (ResearchCandidate(source_url=duplicate_url, title="Date", snippet="Date"),),
        ),
        assessments=(
            refresh_request(),
            refresh_request(
                purpose="event_date",
                query="official 2027 event date",
                gap="The event date is not confirmed.",
            ),
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(
            source_url,
            final_url=first_url if source_url == duplicate_url else None,
            title="Baden Marathon 2027",
            primary_text="Baden Marathon 2027 standard marathon information.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.scout_calls) == 2
    assert len(agent.assessment_calls) == 2
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_searches_and_captures_bounded_official_followups(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    registration_url = "https://baden.example/registration/information"
    lottery_url = "https://baden.example/registration/lottery"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url="https://other.example/registration",
                title="Unrelated registration",
                snippet="A different organizer.",
            ),
            ResearchCandidate(
                source_url="https://baden.example/kids-and-youth/mini-marathon",
                title="Mini Marathon registration",
                snippet="Registration for children.",
            ),
            ResearchCandidate(
                source_url=registration_url,
                title="Baden Marathon registration",
                snippet="The standard public marathon registration dates.",
            ),
            ResearchCandidate(
                source_url=lottery_url,
                title="Baden Marathon lottery",
                snippet="Lottery results and closing date.",
            ),
        ),
        decide=lambda request: supported_update(
            request,
            proposed_fields={"registration_status": "closed"},
            summary="The standard marathon lottery closed after the published deadline.",
            confidence=0.98,
            evidence_indexes=tuple(range(len(request.evidence))),
            conflicts=(
                {
                    "field": "event_date",
                    "evidence": tuple(item.reference for item in request.evidence),
                    "summary": (
                        "The approved overview and registration page differ on date wording."
                    ),
                },
            ),
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        texts = {
            source.url: "Baden Marathon 2027.",
            registration_url: "Lottery registration was open until November 6, 2026.",
            lottery_url: "Lottery results were sent at the end of November 2026.",
        }
        return snapshot(
            source_url,
            text=texts[source_url],
            title="Baden Marathon registration",
        )

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_candidates_per_cycle=4,
                max_static_pages_per_job=3,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    assert fetched_urls == [source.url, registration_url]
    assert len(agent.scout_calls) == 1
    assert agent.scout_calls[0].mode == "refresh"
    assert agent.scout_calls[0].approved_source_url == source.url
    assert "Baden Marathon" in agent.scout_calls[0].query
    assert "2027" in agent.scout_calls[0].query
    assert len(agent.assessment_calls) == 2
    assert len(agent.assessment_calls[1].evidence) == 2
    update = list_proposed_event_updates(event_id=source.event.id, database_url=url)[0]
    assert update.proposed_fields == {"registration_status": "closed"}
    assert sum("researcher-evidence:v1" in line for line in update.evidence) == 2
    assert any(
        line.startswith("Researcher field support: registration_status <- ")
        for line in update.evidence
    )
    assert any(
        line.startswith("Researcher conflict: event_date <- ")
        and "differ on date wording" in line
        for line in update.evidence
    )


def test_refresh_search_authority_stays_with_stored_source_host(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    redirected_host_candidate = "https://registration-platform.example/baden"
    stored_host_candidate = "https://baden.example/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=redirected_host_candidate,
                title="Baden Marathon registration",
                snippet="Registration details.",
            ),
            ResearchCandidate(
                source_url=stored_host_candidate,
                title="Baden Marathon registration",
                snippet="Registration details.",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The captured pages do not support a change.",
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        if source_url == source.url:
            return snapshot(
                source_url,
                final_url="https://registration-platform.example/baden",
            )
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert agent.scout_calls == []
    assert fetched_urls == [source.url]


def test_refresh_revalidates_identity_after_same_host_redirect(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    candidate_url = "https://baden.example/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=candidate_url,
                title="Baden Marathon registration",
                snippet="Registration details for the standard marathon.",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The approved event page does not support a change.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == candidate_url:
            return snapshot(
                source_url,
                final_url="https://baden.example/kids-and-youth/mini-marathon",
                title="mini-MARATHON for kids and youth",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.assessment_calls) == 1


def test_refresh_rejects_cross_host_followup_redirect_and_keeps_searching(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    redirected_url = "https://baden.example/registration/redirect"
    working_url = "https://baden.example/registration/lottery"
    agent = FakeAgent(
        candidates=tuple(
            ResearchCandidate(
                source_url=candidate_url,
                title="Baden Marathon registration",
                snippet="Registration details for the standard marathon.",
            )
            for candidate_url in (redirected_url, working_url)
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The accepted official evidence is current.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == redirected_url:
            return snapshot(
                source_url,
                final_url="https://registration-platform.example/baden",
            )
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_static_pages_per_job=3,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "no_change"
    assert len(agent.assessment_calls) == 2
    assert agent.assessment_calls[1].evidence[1].final_url == working_url


def test_refresh_counts_rejected_capture_against_static_page_budget(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    redirected_url = "https://baden.example/registration/redirect"
    later_url = "https://baden.example/registration/lottery"
    agent = FakeAgent(
        candidates=tuple(
            ResearchCandidate(
                source_url=candidate_url,
                title="Baden Marathon registration",
                snippet="Registration details for the standard marathon.",
            )
            for candidate_url in (redirected_url, later_url)
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The accepted official evidence is current.",
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        if source_url == redirected_url:
            return snapshot(
                source_url,
                final_url="https://registration-platform.example/baden",
            )
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_static_pages_per_job=2,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "inconclusive"
    assert fetched_urls == [source.url, redirected_url]
    assert len(agent.assessment_calls) == 1


def test_refresh_tries_later_candidate_after_followup_fetch_failure(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    failing_url = "https://baden.example/registration/temporary"
    working_url = "https://baden.example/registration/lottery"
    agent = FakeAgent(
        candidates=tuple(
            ResearchCandidate(
                source_url=candidate_url,
                title="Baden Marathon registration",
                snippet="Registration details for the standard marathon.",
            )
            for candidate_url in (failing_url, working_url)
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The captured registration evidence is current.",
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        if source_url == failing_url:
            raise PageFetchError("temporary fetch failure")
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_static_pages_per_job=2,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "no_change"
    assert fetched_urls == [source.url, failing_url, working_url]
    assert len(agent.assessment_calls) == 2


def test_refresh_accepts_supported_subset_of_captured_evidence(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    registration_url = "https://baden.example/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=registration_url,
                title="Baden Marathon registration",
                snippet="Registration details for the standard marathon.",
            ),
        ),
        decide=lambda request: supported_update(
            request,
            proposed_fields={"registration_status": "closed"},
            summary="Registration is closed.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "proposal_created"
    assert count_proposed_event_updates(database_url=url) == 1


def test_refresh_without_search_budget_preserves_single_page_assessment(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(decide=refresh_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_web_searches_per_job=0,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    assert agent.scout_calls == []
    assert len(agent.assessment_calls[0].evidence) == 1


def test_refresh_single_page_budget_skips_search(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(decide=refresh_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_web_searches_per_job=2,
                max_static_pages_per_job=1,
                max_wall_time_seconds_per_job=10,
            ),
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    assert agent.scout_calls == []
    assert len(agent.assessment_calls[0].evidence) == 1


def test_refresh_search_failure_is_inconclusive_without_queue(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)

    class FailingScoutAgent(FakeAgent):
        async def scout(self, request):
            self.scout_calls.append(request)
            return ScoutRunResult(
                state=AgentRunState.FAILED,
                metadata=metadata(),
                error_code="provider_error",
                detail="The provider request failed.",
            )

    agent = FailingScoutAgent(assessments=(refresh_request(),))

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "failed"
    assert result.status.outcome == "inconclusive"
    assert len(agent.assessment_calls) == 1
    assert len(agent.assessment_calls[0].evidence) == 1
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_rejects_candidate_from_different_event_edition(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    stale_url = "https://baden.example/archive/registration"
    current_url = "https://baden.example/registration/2027"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=stale_url,
                title="Baden Marathon 2026 registration",
                snippet="Archived registration page.",
                event_date="2026-09-20",
            ),
            ResearchCandidate(
                source_url=current_url,
                title="Baden Marathon 2027 registration",
                snippet="Current registration page.",
                event_date="2027-09-19",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="The current registration evidence is unchanged.",
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return snapshot(
            source_url,
            title="Baden Marathon 2027 registration",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "no_change"
    assert fetched_urls == [source.url, current_url]
    assert len(agent.assessment_calls[1].evidence) == 2


def test_refresh_rejects_prior_edition_after_capture(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    stale_url = "https://baden.example/archive/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=stale_url,
                title="Baden Marathon registration",
                snippet="Official registration information.",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="Only the approved event page is current.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == stale_url:
            return snapshot(
                source_url,
                title="Baden Marathon registration",
                text="Flattened page text.",
                primary_text="The Baden Marathon 2026 edition is closed.",
                chrome_text="Baden Marathon 2027 navigation.",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.assessment_calls) == 1


def test_refresh_rejects_different_same_year_event_after_capture(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    sibling_url = "https://baden.example/spring-marathon/2027/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=sibling_url,
                title="Spring Marathon 2027 registration",
                snippet="Official registration information.",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="Only the approved Baden Marathon page matches the event.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == sibling_url:
            return snapshot(
                source_url,
                title="Spring Marathon 2027 registration",
                text="Registration for the 2027 Spring Marathon is open.",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.assessment_calls) == 1


def test_refresh_rejects_child_event_named_only_in_captured_body(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    child_url = "https://baden.example/registration/children"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=child_url,
                title="Baden Marathon registration",
                snippet="Official registration information.",
            ),
        ),
        decide=lambda request: supported_no_change(
            request,
            summary="Only the approved standard marathon page matches the event.",
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == child_url:
            return snapshot(
                source_url,
                title="Registration",
                text="Flattened legacy text.",
                primary_text="Baden Mini Marathon registration for kids and youth.",
                chrome_text="Baden Marathon navigation and footer.",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert len(agent.assessment_calls) == 1


def test_refresh_enforces_end_to_end_wall_time_budget(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    followup_url = "https://baden.example/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=followup_url,
                title="Baden Marathon registration",
                snippet="Current registration information.",
            ),
        ),
        decide=refresh_decision,
    )
    budget = ResearchBudget(max_wall_time_seconds_per_job=10).model_copy(
        update={"max_wall_time_seconds_per_job": 0.01}
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == followup_url:
            await asyncio.sleep(0.05)
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=budget,
        ).refresh(source)
    )

    assert result.status.status == "capped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Refresh wall-time budget was exhausted."
    assert len(agent.assessment_calls) == 1
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_request_respects_search_and_page_caps_without_queue(tmp_path: Path) -> None:
    for suffix, budget in (
        (
            "search",
            ResearchBudget(max_web_searches_per_job=0, max_wall_time_seconds_per_job=10),
        ),
        (
            "pages",
            ResearchBudget(max_static_pages_per_job=1, max_wall_time_seconds_per_job=10),
        ),
    ):
        case_path = tmp_path / suffix
        case_path.mkdir()
        url = database_url(case_path)
        source = tracked_source(url)
        agent = FakeAgent(assessments=(refresh_request(),))

        async def fetch(source_url: str) -> PageSnapshot:
            return snapshot(source_url, title="Baden Marathon")

        result = asyncio.run(
            service(
                case_path,
                url=url,
                agent=agent,
                fetch=fetch,
                budget=budget,
            ).refresh(source)
        )

        assert result.status.status == "capped"
        assert result.status.outcome == "inconclusive"
        assert agent.scout_calls == []
        assert count_proposed_event_updates(database_url=url) == 0


def test_thin_high_confidence_no_change_is_inconclusive(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(
        assessments=(
            lambda request: ResearchDecision(
                action="no_change",
                summary="No change despite no applicable current-field proof.",
                confidence=1.0,
                evidence=[request.evidence[0].reference],
            ),
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_can_explicitly_clear_invalid_registration_url(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    invalid_url = "https://baden.example/kids-and-youth/mini-marathon"
    event = add_event(
        replace(event_payload(), registration_url=invalid_url),
        database_url=url,
    )
    source = list_due_sources(
        due_before=datetime.now(UTC),
        limit=1,
        database_url=url,
    )[0]

    def decide(request) -> ResearchDecision:
        return supported_update(
            request,
            proposed_fields={
                "registration_status": "closed",
                "clear_fields": ["registration_url"],
            },
            summary="Standard marathon registration is closed; the saved URL is a child event.",
            confidence=0.96,
        )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, text="The standard marathon lottery is closed.")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=FakeAgent(decide=decide),
            fetch=fetch,
        ).refresh(source)
    )

    assert result.status.outcome == "proposal_created"
    assert find_event(event.id, url).registration_url == invalid_url
    updates = list_proposed_event_updates(event_id=event.id, database_url=url)
    assert updates[0].current_fields == {
        "registration_status": "unknown",
        "registration_url": invalid_url,
    }
    assert updates[0].proposed_fields == {
        "registration_status": "closed",
        "registration_url": None,
    }


def test_shadow_mode_audits_supported_finding_without_queue_write(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    async def fetch_with_policy(
        source_url: str,
        *,
        allowed_origin: str | None = None,
    ) -> PageSnapshot:
        del allowed_origin
        return await fetch(source_url)

    shadow = ResearcherService(
        database_url=url,
        artifacts=ResearchArtifactStore(tmp_path / "shadow-runs"),
        agent=FakeAgent(decide=refresh_decision),
        budget=ResearchBudget(max_wall_time_seconds_per_job=10),
        fetch_snapshot=fetch_with_policy,
        persist_queue=False,
    )

    result = asyncio.run(shadow.refresh(source))

    assert result.status.status == "succeeded"
    assert result.status.outcome == "inconclusive"
    assert "shadow" in (result.status.detail or "").casefold()
    assert count_proposed_event_updates(database_url=url) == 0


def test_uncaptured_evidence_and_profile_only_result_never_write_queue(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    def uncaptured(_request) -> ResearchDecision:
        reference = {
            "run_id": "019c6e27-e55b-73d1-87d8-4e01f1f75043",
            "artifact_name": "page_snapshot-other.json",
            "source_url": "https://other.example/page",
            "content_hash": "b" * 64,
        }
        return ResearchDecision.model_validate(
            {
                "action": "propose_update",
                "summary": "Unsupported evidence.",
                "proposed_fields": {"registration_status": "open"},
                "evidence": [reference],
                "applicability": [
                    {
                        "evidence": reference,
                        "event_identity": "confirmed",
                        "event_edition": "confirmed",
                        "distance_category": "confirmed",
                        "applicable_fields": ["registration_status"],
                    }
                ],
                "field_support": [
                    {"field": "registration_status", "evidence": [reference]}
                ],
            }
        )

    bad = asyncio.run(
        service(
            tmp_path / "bad",
            url=url,
            agent=FakeAgent(decide=uncaptured),
            fetch=fetch,
        ).refresh(source)
    )

    def profile_only(_request) -> ResearchDecision:
        return ResearchDecision(
            action="no_change",
            summary="Only unsupported profile fields such as the event name changed.",
        )

    profile = asyncio.run(
        service(
            tmp_path / "profile",
            url=url,
            agent=FakeAgent(decide=profile_only),
            fetch=fetch,
        ).refresh(source)
    )

    assert bad.status.outcome == "inconclusive"
    assert profile.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


def test_concurrent_identical_refreshes_admit_one_pending_proposal(tmp_path: Path) -> None:
    base_url = database_url(tmp_path)
    source = tracked_source(base_url)
    urls = (
        database_url(tmp_path, query="?timeout=1"),
        database_url(tmp_path, query="?timeout=2"),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    def refresh(index: int) -> ResearchJobResult:
        worker = service(
            tmp_path / f"worker-{index}",
            url=urls[index],
            agent=FakeAgent(decide=refresh_decision),
            fetch=fetch,
        )
        return asyncio.run(worker.refresh(source))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(refresh, range(2)))

    assert sorted(result.status.status for result in results) == ["skipped", "succeeded"]
    assert count_proposed_event_updates(database_url=base_url) == 1
    succeeded = next(result for result in results if result.status.status == "succeeded")
    skipped = next(result for result in results if result.status.status == "skipped")
    assert succeeded.queue_reference == "proposed_event_update:1"
    # The losing run names the blocking pending update through the typed field.
    assert skipped.conflicting_update_id == 1
    assert skipped.queue_reference is None


def test_service_boundary_has_no_direct_apply_or_registration_orchestration() -> None:
    import run4221.researcher.service as researcher_service

    path = Path(researcher_service.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert all(not module.startswith("run4221.bot") for module in imported)
    assert "update_registration_window" not in source
    assert "auto_confirm" not in inspect.signature(ResearcherService.refresh).parameters


def test_refresh_accepts_half_marathon_evidence_for_mixed_distance_event(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    add_event(
        replace(event_payload(), distances=("marathon", "half_marathon")),
        database_url=url,
    )
    source = list_due_sources(due_before=datetime.now(UTC), limit=1, database_url=url)[0]
    followup_url = "https://baden.example/halbmarathon/anmeldung"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=followup_url,
                title="Baden Marathon Halbmarathon registration",
                snippet="Current half marathon registration information.",
            ),
        ),
        assessments=(
            refresh_request(),
            lambda request: supported_update(request, evidence_indexes=(1,)),
        ),
    )
    fetched_urls: list[str] = []

    async def fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return snapshot(
            source_url,
            title="Baden Marathon 2027",
            primary_text="Baden Marathon Halbmarathon registration is open.",
        )

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "proposal_created"
    assert fetched_urls == [source.url, followup_url]


def test_refresh_still_rejects_mini_marathon_evidence_for_mixed_distance_event(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    add_event(
        replace(event_payload(), distances=("marathon", "half_marathon")),
        database_url=url,
    )
    source = list_due_sources(due_before=datetime.now(UTC), limit=1, database_url=url)[0]
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url="https://baden.example/mini-marathon/anmeldung",
                title="Baden Mini Marathon registration",
                snippet="Kids race registration.",
            ),
        ),
        assessments=(refresh_request(),),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_rejects_event_date_older_than_stored_event_date(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)  # stored event_date: 2027-09-19
    agent = FakeAgent(
        decide=lambda request: supported_update(
            request,
            proposed_fields={"event_date": "2026-09-20"},
            summary="A stale page still lists the previous edition date.",
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Proposed event date is older than the stored event date."
    assert count_proposed_event_updates(database_url=url) == 0


def test_refresh_accepts_event_date_later_than_stored_event_date(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(
        decide=lambda request: supported_update(
            request,
            proposed_fields={"event_date": "2028-04-16"},
            summary="The official page announces the next edition date.",
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "proposal_created"
    updates = list_proposed_event_updates(event_id=source.event.id, database_url=url)
    assert updates[0].proposed_fields == {"event_date": "2028-04-16"}


PROFILE_URL = "https://news.example/karlsruhe-marathon-report"
OFFICIAL_URL = "https://baden.example/events/marathon"


def profile_draft_payload(source_url: str) -> dict[str, object]:
    return {
        "source_url": source_url,
        "name": "Baden Marathon",
        "public_id": "badenmarathon.42",
        "city": "Karlsruhe",
        "country": "Germany",
        "timezone": "Europe/Berlin",
        "event_date": "2027-09-19",
        "distances": ("marathon",),
        "regions": ("global", "eu", "de"),
        "official_url": OFFICIAL_URL,
        "registration_url": "https://baden.example/registration",
        "registration_url_candidates": (
            {"url": "https://baden.example/registration", "link_text": "Registration"},
        ),
        "summary": "The captured page profiles the Baden Marathon 2027.",
        "confidence": 0.9,
    }


def profile_decision(request, *, page_is_event: bool = True) -> ResearchDecision:
    evidence = request.evidence[0]
    return ResearchDecision.model_validate(
        {
            "action": "profile_event",
            "summary": "The captured page yields a grounded event profile.",
            "confidence": 0.9,
            "draft": profile_draft_payload(evidence.final_url),
            "page_is_event": page_is_event,
            "evidence": [evidence.reference],
        }
    )


def profile_service(
    tmp_path: Path,
    *,
    url: str,
    agent: FakeAgent,
    fetch,
    budget: ResearchBudget | None = None,
    persist_queue: bool = True,
    enriched_calls: list[tuple[str, int]] | None = None,
) -> ResearcherService:
    async def fetch_enriched(source_url: str, *, max_linked_pages: int) -> PageSnapshot:
        if enriched_calls is not None:
            enriched_calls.append((source_url, max_linked_pages))
        return await fetch(source_url)

    return ResearcherService(
        database_url=url,
        artifacts=ResearchArtifactStore(tmp_path / "runs"),
        agent=agent,
        budget=budget
        or ResearchBudget(
            max_wall_time_seconds_per_job=10,
            max_wall_time_seconds_per_profile_job=10,
        ),
        fetch_enriched_snapshot=fetch_enriched,
        persist_queue=persist_queue,
    )


def test_default_profile_fetcher_is_the_dedicated_enriched_seam() -> None:
    parameters = inspect.signature(ResearcherService).parameters

    assert parameters["fetch_enriched_snapshot"].default is fetch_enriched_page_snapshot
    # The dedicated seam leaves the pinned single-page refresh seam untouched.
    assert parameters["fetch_snapshot"].default is fetch_page_snapshot


def test_ae1_profile_grounded_event_page_yields_cited_draft(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=profile_decision)
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            enriched_calls=enriched_calls,
        ).profile(OFFICIAL_URL)
    )

    assert isinstance(result, ProfileJobResult)
    assert result.status.status == "succeeded"
    assert result.status.outcome == "profile_completed"
    assert result.located is False
    assert isinstance(result.draft, EventProfileDraft)
    assert result.draft.name == "Baden Marathon"
    assert result.draft.official_url == OFFICIAL_URL
    assert result.draft.registration_url_candidates[0].as_pair == (
        "https://baden.example/registration",
        "Registration",
    )
    # One merged enriched capture, assessed once, with zero locating searches.
    assert enriched_calls == [(OFFICIAL_URL, 3)]
    assert agent.scout_calls == []
    assert len(agent.assessment_calls) == 1
    assert agent.assessment_calls[0].mode == "profile"
    assert len(agent.assessment_calls[0].evidence) == 1


def test_ae2_blocked_profile_page_terminates_inconclusive_without_draft(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=profile_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, status_code=403, text="Access denied by protection.")

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(OFFICIAL_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert "unusable" in (result.status.detail or "")
    assert result.draft is None
    assert agent.assessment_calls == []


def test_ae2_wrong_event_profile_assessment_terminates_inconclusive_with_reason(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    reason = "The captured page describes a different event than requested."
    agent = FakeAgent(
        decide=lambda request: ResearchDecision(
            action="inconclusive",
            summary=reason,
            evidence=[request.evidence[0].reference],
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Some Other Race")

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(PROFILE_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == reason
    assert result.draft is None


def test_profile_wall_cap_applies_not_refresh_cap(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=profile_decision)
    budget = ResearchBudget(max_wall_time_seconds_per_job=900).model_copy(
        update={"max_wall_time_seconds_per_profile_job": 0.01}
    )

    async def fetch(source_url: str) -> PageSnapshot:
        await asyncio.sleep(0.05)
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=budget,
        ).profile(OFFICIAL_URL)
    )

    assert result.status.status == "capped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Profile wall-time budget was exhausted."
    assert result.draft is None
    assert agent.assessment_calls == []


def test_profile_writes_no_queue_rows_even_with_persist_queue_true(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    add_event(event_payload(), database_url=url)
    agent = FakeAgent(decide=profile_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            persist_queue=True,
        ).profile(OFFICIAL_URL)
    )

    assert result.status.outcome == "profile_completed"
    assert count_proposed_event_updates(database_url=url) == 0
    assert count_event_suggestions(database_url=url) == 0


def test_profile_enrichment_sub_fetches_stay_inside_static_page_budget(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=profile_decision)
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_static_pages_per_job=2,
                max_wall_time_seconds_per_job=10,
                max_wall_time_seconds_per_profile_job=10,
            ),
            enriched_calls=enriched_calls,
        ).profile(OFFICIAL_URL)
    )

    assert result.status.outcome == "profile_completed"
    # The root fetch plus at most one linked sub-fetch: two static pages total.
    assert enriched_calls == [(OFFICIAL_URL, 1)]


def test_profile_non_event_page_locates_official_page_with_one_search(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=OFFICIAL_URL,
                title="Baden Marathon",
                snippet="Official event page.",
            ),
        ),
        assessments=(
            lambda request: profile_decision(request, page_is_event=False),
            lambda request: profile_decision(request),
        ),
    )
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            enriched_calls=enriched_calls,
        ).profile(PROFILE_URL)
    )

    assert result.status.outcome == "profile_completed"
    # The locate hop is typed provenance for the caller to surface.
    assert result.located is True
    assert isinstance(result.draft, EventProfileDraft)
    assert [call_url for call_url, _ in enriched_calls] == [PROFILE_URL, OFFICIAL_URL]
    assert len(agent.scout_calls) == 1
    assert agent.scout_calls[0].mode == "profile"
    assert agent.scout_calls[0].approved_source_url is None
    assert "Baden Marathon" in agent.scout_calls[0].query
    assert len(agent.assessment_calls) == 2
    # Only the first assessment reserves budget for the potential locate hop.
    assert agent.assessment_calls[0].reserve_continuation is True
    assert agent.assessment_calls[1].reserve_continuation is False
    assert agent.assessment_calls[1].evidence[0].final_url == OFFICIAL_URL


def test_profile_locate_runs_at_most_once_then_terminates_inconclusive(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=OFFICIAL_URL,
                title="Baden Marathon",
                snippet="Still an aggregator listing.",
            ),
        ),
        assessments=(
            lambda request: profile_decision(request, page_is_event=False),
            lambda request: profile_decision(request, page_is_event=False),
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(PROFILE_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.draft is None
    assert len(agent.scout_calls) == 1
    assert len(agent.assessment_calls) == 2


def test_profile_locate_without_search_budget_is_capped_without_draft(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        assessments=(lambda request: profile_decision(request, page_is_event=False),)
    )

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=lambda source_url: _snapshot_async(source_url),
            budget=ResearchBudget(
                max_web_searches_per_job=0,
                max_wall_time_seconds_per_job=10,
                max_wall_time_seconds_per_profile_job=10,
            ),
        ).profile(PROFILE_URL)
    )

    assert result.status.status == "capped"
    assert result.status.outcome == "inconclusive"
    assert result.draft is None
    assert agent.scout_calls == []


async def _snapshot_async(source_url: str) -> PageSnapshot:
    return snapshot(source_url)


def test_profile_rejects_refresh_style_decision_without_draft(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=refresh_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(OFFICIAL_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.draft is None
    assert count_proposed_event_updates(database_url=url) == 0


def draft_profile_decision(request, **draft_overrides) -> ResearchDecision:
    evidence = request.evidence[0]
    payload = profile_draft_payload(evidence.final_url)
    payload.update(draft_overrides)
    return ResearchDecision.model_validate(
        {
            "action": "profile_event",
            "summary": "The captured page yields a grounded event profile.",
            "confidence": 0.9,
            "draft": payload,
            "page_is_event": True,
            "evidence": [evidence.reference],
        }
    )


def test_profile_rejects_draft_source_url_that_is_not_the_captured_page(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        decide=lambda request: draft_profile_decision(
            request,
            source_url="https://elsewhere.example/other-page",
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(OFFICIAL_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Draft URLs did not match the captured evidence."
    assert result.draft is None


def test_profile_rejects_cross_domain_official_url(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        decide=lambda request: draft_profile_decision(
            request,
            official_url="https://other.example/marathon",
        )
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(OFFICIAL_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Draft URLs did not match the captured evidence."
    assert result.draft is None


def test_profile_accepts_matching_draft_and_absent_official_url(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        decide=lambda request: draft_profile_decision(request, official_url=None)
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(OFFICIAL_URL)
    )

    assert result.status.outcome == "profile_completed"
    assert result.draft is not None
    assert result.draft.official_url is None


def test_profile_locate_capture_shares_the_static_page_budget(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=OFFICIAL_URL,
                title="Baden Marathon",
                snippet="Official event page.",
            ),
        ),
        assessments=(
            lambda request: profile_decision(request, page_is_event=False),
            lambda request: profile_decision(request),
        ),
    )
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon 2027")

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            budget=ResearchBudget(
                max_static_pages_per_job=3,
                max_wall_time_seconds_per_job=10,
                max_wall_time_seconds_per_profile_job=10,
            ),
            enriched_calls=enriched_calls,
        ).profile(PROFILE_URL)
    )

    assert result.status.outcome == "profile_completed"
    # The locate capture only gets the linked-page allowance the first
    # enriched capture left unspent.
    assert enriched_calls == [(PROFILE_URL, 2), (OFFICIAL_URL, 0)]


def test_profile_locate_accepts_candidate_linked_from_first_capture(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    located_url = "https://runfest.example/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=located_url,
                title="Run Fest",
                snippet="Official event page.",
            ),
        ),
        assessments=(
            lambda request: profile_decision(request, page_is_event=False),
            # The located domain differs from the draft's official URL domain,
            # so the second draft must not carry a cross-domain official URL.
            lambda request: draft_profile_decision(request, official_url=None),
        ),
    )
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        links = (
            (PageLink(url="https://runfest.example/home", text="event site"),)
            if source_url == PROFILE_URL
            else ()
        )
        return snapshot(source_url, links=links)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            enriched_calls=enriched_calls,
        ).profile(PROFILE_URL)
    )

    # "runfest.example" shares no event-name token; the first capture's link
    # to that domain is what corroborates the located page.
    assert result.status.outcome == "profile_completed"
    assert result.located is True
    assert [call_url for call_url, _ in enriched_calls] == [PROFILE_URL, located_url]


def test_profile_locate_accepts_candidate_domain_sharing_event_name_token(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    located_url = "https://baden-events.example/marathon"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=located_url,
                title="Baden Marathon",
                snippet="Official event page.",
            ),
        ),
        assessments=(
            lambda request: profile_decision(request, page_is_event=False),
            lambda request: draft_profile_decision(request, official_url=None),
        ),
    )
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            enriched_calls=enriched_calls,
        ).profile(PROFILE_URL)
    )

    # No link and no mention of the domain in the first capture: the shared
    # significant name token ("baden") corroborates; "marathon" alone is too
    # generic to count.
    assert result.status.outcome == "profile_completed"
    assert result.located is True
    assert [call_url for call_url, _ in enriched_calls] == [PROFILE_URL, located_url]


def test_profile_locate_rejects_uncorroborated_candidate_before_second_capture(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url="https://unrelated.example/race",
                title="Some Race",
                snippet="Search listing.",
            ),
        ),
        assessments=(lambda request: profile_decision(request, page_is_event=False),),
    )
    enriched_calls: list[tuple[str, int]] = []

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            enriched_calls=enriched_calls,
        ).profile(PROFILE_URL)
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == (
        "Located page could not be corroborated by the captured evidence."
    )
    assert result.draft is None
    # Rejected deterministically before any second capture or assessment.
    assert [call_url for call_url, _ in enriched_calls] == [PROFILE_URL]
    assert len(agent.assessment_calls) == 1


def test_profile_with_oversized_url_finalizes_failed_terminal(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    agent = FakeAgent(decide=profile_decision)
    long_url = "https://example.com/" + "a" * 1_100

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(
        profile_service(tmp_path, url=url, agent=agent, fetch=fetch).profile(long_url)
    )

    assert result.status.status == "failed"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Profile request construction failed schema validation."
    assert result.draft is None
    assert agent.assessment_calls == []
    # The run directory is terminated instead of being left half-written.
    assert (tmp_path / "runs" / result.run_id / "terminal.json").is_file()


def test_refresh_skips_http_error_source_before_assessment(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    agent = FakeAgent(decide=refresh_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, status_code=404)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert result.status.detail == "Approved source was unusable: unusable HTTP status 404."
    assert agent.assessment_calls == []
    assert count_proposed_event_updates(database_url=url) == 0
