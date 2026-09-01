from __future__ import annotations

import ast
import asyncio
import inspect
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
    list_event_suggestions,
    list_events,
    list_proposed_event_updates,
)
from run4221.db.research import list_due_sources
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageLink,
    PageSnapshot,
    fetch_page_snapshot,
)
from run4221.researcher.agent import (
    AgentRunMetadata,
    AgentRunState,
    AssessmentRunResult,
    ScoutRunResult,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.policy import SourceTrustPolicy
from run4221.researcher.schemas import ResearchBudget, ResearchCandidate, ResearchDecision
from run4221.researcher.service import ResearcherService, ResearchJobResult


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
    title: str = "Example Marathon",
    status_code: int = 200,
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
    )


def metadata() -> AgentRunMetadata:
    return AgentRunMetadata(model="gpt-5.6-luna", prompt_reference="research_agent:v1")


class FakeAgent:
    def __init__(self, *, candidates=(), decide) -> None:
        self.candidates = tuple(candidates)
        self.decide = decide
        self.scout_calls = []
        self.assessment_calls = []

    async def scout(self, request):
        self.scout_calls.append(request)
        return ScoutRunResult(
            state=AgentRunState.SUCCEEDED,
            metadata=metadata(),
            candidates=self.candidates,
        )

    async def assess(self, request):
        self.assessment_calls.append(request)
        return AssessmentRunResult(
            state=AgentRunState.SUCCEEDED,
            metadata=metadata(),
            decision=self.decide(request),
        )


def refresh_decision(request) -> ResearchDecision:
    return ResearchDecision(
        action="propose_update",
        summary="The captured approved source says registration is open.",
        confidence=0.95,
        proposed_fields={"registration_status": "open"},
        evidence=[request.evidence[0].reference],
    )


def discovery_decision(request) -> ResearchDecision:
    return ResearchDecision(
        action="suggest_event",
        summary="Captured official event page with marathon details.",
        confidence=0.91,
        candidate=ResearchCandidate(
            source_url=request.evidence[0].final_url,
            title="New City Marathon",
            snippet="Marathon in New City on 2027-10-10.",
            event_date="2027-10-10",
            location="New City, Germany",
            region_tags=("eu", "de"),
            distances=("marathon",),
        ),
        evidence=[item.reference for item in request.evidence],
    )


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
    trusted_domains=("baden.example", "official.example"),
    trusted_registry_urls=(),
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
        trust_policy=SourceTrustPolicy(
            trusted_domains=frozenset(trusted_domains),
            trusted_registry_urls=tuple(trusted_registry_urls),
        ),
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
        decide=lambda request: ResearchDecision(
            action="propose_update",
            summary="The standard marathon lottery closed after the published deadline.",
            confidence=0.98,
            proposed_fields={"registration_status": "closed"},
            evidence=[item.reference for item in request.evidence],
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
    assert fetched_urls == [source.url, registration_url, lottery_url]
    assert len(agent.scout_calls) == 1
    assert agent.scout_calls[0].mode == "refresh"
    assert agent.scout_calls[0].approved_source_url == source.url
    assert "Baden Marathon" in agent.scout_calls[0].query
    assert "2027" in agent.scout_calls[0].query
    assert len(agent.assessment_calls[0].evidence) == 3
    update = list_proposed_event_updates(event_id=source.event.id, database_url=url)[0]
    assert update.proposed_fields == {"registration_status": "closed"}
    assert sum("researcher-evidence:v1" in line for line in update.evidence) == 3


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The captured pages do not support a change.",
            evidence=[item.reference for item in request.evidence],
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

    assert result.status.outcome == "no_change"
    assert agent.scout_calls[0].approved_source_url == source.url
    assert fetched_urls == [source.url, stored_host_candidate]


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The approved event page does not support a change.",
            evidence=[item.reference for item in request.evidence],
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

    assert result.status.outcome == "no_change"
    assert len(agent.assessment_calls[0].evidence) == 1


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The accepted official evidence is current.",
            evidence=[item.reference for item in request.evidence],
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
    assert len(agent.assessment_calls[0].evidence) == 2
    assert agent.assessment_calls[0].evidence[1].final_url == working_url


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The accepted official evidence is current.",
            evidence=[item.reference for item in request.evidence],
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

    assert result.status.outcome == "no_change"
    assert fetched_urls == [source.url, redirected_url]
    assert len(agent.assessment_calls[0].evidence) == 1


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The captured registration evidence is current.",
            evidence=[item.reference for item in request.evidence],
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
    assert len(agent.assessment_calls[0].evidence) == 2


def test_refresh_rejects_incomplete_followup_evidence_set(tmp_path: Path) -> None:
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
        decide=lambda request: ResearchDecision(
            action="propose_update",
            summary="Registration is closed.",
            proposed_fields={"registration_status": "closed"},
            evidence=[request.evidence[0].reference],
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, title="Baden Marathon registration")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "inconclusive"
    assert count_proposed_event_updates(database_url=url) == 0


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


def test_refresh_search_failure_falls_back_to_approved_capture(tmp_path: Path) -> None:
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

    agent = FailingScoutAgent(decide=refresh_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.status == "succeeded"
    assert result.status.outcome == "proposal_created"
    assert len(agent.assessment_calls) == 1
    assert len(agent.assessment_calls[0].evidence) == 1
    assert count_proposed_event_updates(database_url=url) == 1


def test_refresh_rejects_candidate_from_different_event_edition(
    tmp_path: Path,
) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    stale_url = "https://baden.example/archive/2026/registration"
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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="The current registration evidence is unchanged.",
            evidence=[item.reference for item in request.evidence],
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
    assert len(agent.assessment_calls[0].evidence) == 2


def test_refresh_rejects_prior_edition_after_capture(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)
    stale_url = "https://baden.example/archive/2026/registration"
    agent = FakeAgent(
        candidates=(
            ResearchCandidate(
                source_url=stale_url,
                title="Baden Marathon registration",
                snippet="Official registration information.",
            ),
        ),
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="Only the approved event page is current.",
            evidence=[item.reference for item in request.evidence],
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == stale_url:
            return snapshot(
                source_url,
                title="Baden Marathon 2026 registration",
                text="The 2026 edition is closed.",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "no_change"
    assert len(agent.assessment_calls[0].evidence) == 1


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="Only the approved Baden Marathon page matches the event.",
            evidence=[item.reference for item in request.evidence],
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

    assert result.status.outcome == "no_change"
    assert len(agent.assessment_calls[0].evidence) == 1


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
        decide=lambda request: ResearchDecision(
            action="no_change",
            summary="Only the approved standard marathon page matches the event.",
            evidence=[item.reference for item in request.evidence],
        ),
    )

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == child_url:
            return snapshot(
                source_url,
                title="Registration",
                text="Baden Mini Marathon registration for kids and youth.",
            )
        return snapshot(source_url, title="Baden Marathon")

    result = asyncio.run(service(tmp_path, url=url, agent=agent, fetch=fetch).refresh(source))

    assert result.status.outcome == "no_change"
    assert len(agent.assessment_calls[0].evidence) == 1


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
    assert agent.assessment_calls == []
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
        return ResearchDecision(
            action="propose_update",
            summary="Standard marathon registration is closed; the saved URL is a child event.",
            confidence=0.96,
            proposed_fields={
                "registration_status": "closed",
                "clear_fields": ["registration_url"],
            },
            evidence=[request.evidence[0].reference],
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


def test_ae2_discovery_creates_system_suggestion_and_no_event(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    candidate = ResearchCandidate(
        source_url="https://official.example/new-city-marathon",
        title="Search result",
        snippet="Possible event page.",
        discovery_query="Germany marathon 2027",
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, text="New City Marathon, 10 October 2027.")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=FakeAgent(candidates=(candidate,), decide=discovery_decision),
            fetch=fetch,
        ).discover("Germany marathon 2027")
    )

    assert result.status.outcome == "suggestion_created"
    assert result.queue_reference == "event_suggestion:1"
    assert list_events(database_url=url) == ()
    suggestions = list_event_suggestions(database_url=url)
    assert len(suggestions) == 1
    assert suggestions[0].submitter_user_id is None
    assert suggestions[0].distances == ("marathon",)
    assert suggestions[0].note and "researcher-evidence:v1" in suggestions[0].note
    assert "researcher-decision:v1" in suggestions[0].note
    assert "Source check: configured trusted domain." in suggestions[0].note
    assert "captured_at=2026-08-31T14:00:00+00:00" in suggestions[0].note


def test_discovery_continues_after_no_change_candidate(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    candidates = tuple(
        ResearchCandidate(
            source_url=f"https://official.example/{slug}",
            title=f"Search result {slug}",
            snippet="Possible event page.",
        )
        for slug in ("existing-event", "new-event")
    )

    def decide(request) -> ResearchDecision:
        if request.evidence[0].final_url.endswith("/existing-event"):
            return ResearchDecision(
                action="no_change",
                summary="The first candidate is already represented.",
            )
        return discovery_decision(request)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    agent = FakeAgent(candidates=candidates, decide=decide)
    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
        ).discover("Germany marathon 2027")
    )

    assert result.status.outcome == "suggestion_created"
    assert len(agent.assessment_calls) == 2
    assert list_event_suggestions(database_url=url)[0].url == candidates[1].source_url


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
        trust_policy=SourceTrustPolicy(trusted_domains=frozenset({"baden.example"})),
        budget=ResearchBudget(max_wall_time_seconds_per_job=10),
        fetch_snapshot=fetch_with_policy,
        persist_queue=False,
    )

    result = asyncio.run(shadow.refresh(source))

    assert result.status.status == "succeeded"
    assert result.status.outcome == "inconclusive"
    assert "shadow" in (result.status.detail or "").casefold()
    assert count_proposed_event_updates(database_url=url) == 0


def test_repeated_discovery_is_audited_duplicate_skip(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    candidate = ResearchCandidate(
        source_url="https://official.example/new-city-marathon",
        title="Search result",
        snippet="Possible event page.",
    )

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    first = service(
        tmp_path / "first",
        url=url,
        agent=FakeAgent(candidates=(candidate,), decide=discovery_decision),
        fetch=fetch,
    )
    second = service(
        tmp_path / "second",
        url=url,
        agent=FakeAgent(candidates=(candidate,), decide=discovery_decision),
        fetch=fetch,
    )

    assert asyncio.run(first.discover("marathon")).status.outcome == "suggestion_created"
    duplicate = asyncio.run(second.discover("marathon"))

    assert duplicate.status.status == "skipped"
    assert duplicate.status.outcome == "no_change"
    assert "duplicate" in (duplicate.status.detail or "").casefold()
    assert count_event_suggestions(database_url=url) == 1
    terminal = second.artifacts.read_artifact(duplicate.terminal_reference)
    assert terminal["content"]["queue_state"] == "absent"


def test_untrusted_page_cannot_self_declare_official(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    candidate = ResearchCandidate(
        source_url="https://unknown.example/marathon",
        title="Official Marathon Website",
        snippet="This page says it is official.",
    )
    agent = FakeAgent(candidates=(candidate,), decide=discovery_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url, text="We are the official marathon organizer.")

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            trusted_domains=("registry.example",),
        ).discover("marathon")
    )

    assert result.status.status == "skipped"
    assert result.status.outcome == "inconclusive"
    assert count_event_suggestions(database_url=url) == 0
    assert agent.assessment_calls == []


def test_registry_link_chain_requires_and_persists_both_artifacts(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    candidate_url = "https://organizer.example/new-city-marathon"
    registry_url = "https://registry.example/events"
    candidate = ResearchCandidate(
        source_url=candidate_url,
        title="Search result",
        snippet="Possible event page.",
    )
    agent = FakeAgent(candidates=(candidate,), decide=discovery_decision)

    async def fetch(source_url: str) -> PageSnapshot:
        if source_url == registry_url:
            return snapshot(
                source_url,
                text="Trusted event registry.",
                links=(PageLink(url=candidate_url, text="New City Marathon"),),
            )
        return snapshot(source_url)

    result = asyncio.run(
        service(
            tmp_path,
            url=url,
            agent=agent,
            fetch=fetch,
            trusted_domains=("registry.example",),
            trusted_registry_urls=(registry_url,),
        ).discover("marathon")
    )

    assert result.status.outcome == "suggestion_created"
    assert len(agent.assessment_calls[0].evidence) == 2
    note = list_event_suggestions(database_url=url)[0].note or ""
    assert "Source check: captured trusted registry link." in note
    for evidence in agent.assessment_calls[0].evidence:
        assert evidence.reference.artifact_name in note
        assert evidence.reference.content_hash[:12] in note


def test_uncaptured_evidence_and_profile_only_result_never_write_queue(tmp_path: Path) -> None:
    url = database_url(tmp_path)
    source = tracked_source(url)

    async def fetch(source_url: str) -> PageSnapshot:
        return snapshot(source_url)

    def uncaptured(_request) -> ResearchDecision:
        return ResearchDecision(
            action="propose_update",
            summary="Unsupported evidence.",
            proposed_fields={"registration_status": "open"},
            evidence=[
                {
                    "run_id": "019c6e27-e55b-73d1-87d8-4e01f1f75043",
                    "artifact_name": "page_snapshot-other.json",
                    "source_url": "https://other.example/page",
                    "content_hash": "b" * 64,
                }
            ],
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
    assert profile.status.outcome == "no_change"
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


def test_service_boundary_has_no_direct_apply_or_registration_orchestration() -> None:
    import run4221.researcher.service as researcher_service

    path = Path(researcher_service.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "run4221.ai.registration_window" not in imported
    assert "update_registration_window" not in source
    assert "auto_confirm" not in inspect.signature(ResearcherService.refresh).parameters
    assert "auto_confirm" not in inspect.signature(ResearcherService.discover).parameters
