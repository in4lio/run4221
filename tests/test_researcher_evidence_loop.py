from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from run4221.ai.event_extractor import event_identity_tokens
from run4221.db.repository import (
    EventCreate,
    add_event,
    count_event_suggestions,
    count_proposed_event_updates,
    find_event,
)
from run4221.db.research import list_due_sources
from run4221.ingestion.page_snapshot import PageSnapshot, fetch_page_snapshot
from run4221.researcher.agent import (
    AgentRunMetadata,
    AgentRunState,
    AssessmentRequest,
    AssessmentRunResult,
    ScoutRunResult,
)
from run4221.researcher.artifacts import ResearchArtifactStore
from run4221.researcher.policy import SourceTrustPolicy
from run4221.researcher.schemas import (
    EvidenceRequest,
    ResearchBudget,
    ResearchCandidate,
    ResearchDecision,
)
from run4221.researcher.service import ResearcherService, ResearchJobResult

FIXTURES = Path(__file__).parent / "fixtures" / "researcher"
GENERIC_NAME_TOKENS = {"event", "marathon", "race", "run", "running"}


async def _public_resolver(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


async def _parse(url: str, html: str, status_code: int = 200) -> PageSnapshot:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await fetch_page_snapshot(
            url,
            client=client,
            resolve_host=_public_resolver,
        )


class FixtureFetcher:
    def __init__(self, pages: dict[str, str | tuple[str, int]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(
        self,
        url: str,
        *,
        allowed_origin: str | None = None,
    ) -> PageSnapshot:
        self.calls.append((url, allowed_origin))
        page = self.pages[url]
        html, status = page if isinstance(page, tuple) else (page, 200)
        return await _parse(url, html, status)


class CorpusAgent:
    def __init__(
        self,
        assess: Callable[[AssessmentRequest], ResearchDecision | EvidenceRequest],
        *,
        scout_batches: tuple[tuple[ResearchCandidate, ...], ...] = (),
    ) -> None:
        self.assess_fn = assess
        self.scout_batches = list(scout_batches)
        self.assessment_requests: list[AssessmentRequest] = []
        self.scout_requests = []
        self.decisions: list[ResearchDecision] = []

    async def assess(self, request: AssessmentRequest) -> AssessmentRunResult:
        self.assessment_requests.append(request)
        outcome = self.assess_fn(request)
        if isinstance(outcome, EvidenceRequest):
            return AssessmentRunResult(
                AgentRunState.SUCCEEDED,
                _metadata(),
                evidence_request=outcome,
            )
        self.decisions.append(outcome)
        return AssessmentRunResult(
            AgentRunState.SUCCEEDED,
            _metadata(),
            decision=outcome,
        )

    async def scout(self, request) -> ScoutRunResult:
        self.scout_requests.append(request)
        candidates = self.scout_batches.pop(0) if self.scout_batches else ()
        return ScoutRunResult(
            AgentRunState.SUCCEEDED,
            replace(_metadata(), web_search_calls=1),
            candidates=candidates,
        )


@dataclass(frozen=True)
class Run:
    result: ResearchJobResult
    agent: CorpusAgent
    fetcher: FixtureFetcher
    artifacts: ResearchArtifactStore
    before: tuple[object, int, int]
    after: tuple[object, int, int]


@dataclass(frozen=True)
class Case:
    name: str
    fixture: str
    event_name: str
    city: str
    country: str
    event_date: str
    action: str
    field: str | None
    value: str | None
    verdicts: tuple[str, str, str]
    primary_contains: tuple[str, ...]
    primary_excludes: tuple[str, ...] = ()


def _metadata() -> AgentRunMetadata:
    return AgentRunMetadata(model="deterministic-corpus", prompt_reference="u4:test")


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _context(request: AssessmentRequest) -> dict[str, str]:
    return {item.name: item.value for item in request.context}


def _applicability(
    request: AssessmentRequest,
    index: int,
    field: str | None = None,
) -> dict[str, object]:
    evidence = request.evidence[index]
    searchable = f"{evidence.title or ''} {evidence.primary_text}".casefold()
    context = _context(request)
    expected = event_identity_tokens("", context["name"]) - GENERIC_NAME_TOKENS
    captured = event_identity_tokens("", searchable)
    required = min(2, len(expected))
    identity = (
        "confirmed"
        if required and len(expected & captured) >= required
        else "rejected"
    )
    years = set(re.findall(r"\b20\d{2}\b", searchable))
    event_year = context["event_date"][:4]
    edition = (
        "confirmed"
        if event_year in years
        else "rejected"
        if years
        else "inconclusive"
    )
    negative_distance = any(
        term in searchable
        for term in ("mini marathon", "youth race", "4.2 km", "half marathon", "21k")
    )
    positive_distance = any(
        term in searchable
        for term in ("42.195 km", "42k", "full marathon", "standard marathon")
    )
    distance = (
        "rejected"
        if negative_distance
        else "confirmed"
        if positive_distance
        else "inconclusive"
    )
    return {
        "evidence": evidence.reference,
        "event_identity": identity,
        "event_edition": edition,
        "distance_category": distance,
        "applicable_fields": (field,) if field else (),
    }


def _decision(
    request: AssessmentRequest,
    action: str,
    field: str | None = None,
    value: str | None = None,
) -> ResearchDecision:
    reference = request.evidence[0].reference
    payload: dict[str, object] = {
        "action": action,
        "summary": "Deterministic corpus assessment.",
        "evidence": [reference],
        "applicability": [_applicability(request, 0, field)],
    }
    if action == "propose_update":
        assert field is not None and value is not None
        payload.update(
            proposed_fields={field: value},
            field_support=[{"field": field, "evidence": [reference]}],
            confidence=0.95,
        )
    return ResearchDecision.model_validate(payload)


def _event(case: Case, source_url: str, public_id: str | None = None) -> EventCreate:
    return EventCreate(
        public_id=public_id or f"{case.name}.42",
        name=case.event_name,
        city=case.city,
        country=case.country,
        timezone="Europe/Berlin",
        distances=("marathon",),
        regions=("global",),
        official_url=source_url,
        event_date=case.event_date,
        registration_status="unknown",
    )


def _state(database_url: str, event_id: str) -> tuple[object, int, int]:
    return (
        find_event(event_id, database_url),
        count_event_suggestions(database_url=database_url),
        count_proposed_event_updates(database_url=database_url),
    )


def _run(
    root: Path,
    event: EventCreate,
    pages: dict[str, str | tuple[str, int]],
    agent: CorpusAgent,
    *,
    persist_queue: bool = True,
) -> Run:
    root.mkdir(parents=True)
    database_url = f"sqlite:///{root / 'corpus.sqlite3'}"
    tracked = add_event(event, database_url)
    source = list_due_sources(
        due_before=datetime.now(UTC),
        limit=1,
        database_url=database_url,
    )[0]
    artifacts = ResearchArtifactStore(root / "runs")
    fetcher = FixtureFetcher(pages)
    hostname = urlsplit(source.url).hostname
    assert hostname is not None
    service = ResearcherService(
        database_url=database_url,
        artifacts=artifacts,
        agent=agent,
        trust_policy=SourceTrustPolicy(trusted_domains=frozenset({hostname})),
        budget=ResearchBudget(
            max_wall_time_seconds_per_job=10,
            max_rendered_pages_per_job=0,
        ),
        fetch_snapshot=fetcher,
        persist_queue=persist_queue,
    )
    before = _state(database_url, tracked.id)
    result = asyncio.run(service.refresh(source))
    return Run(
        result,
        agent,
        fetcher,
        artifacts,
        before,
        _state(database_url, tracked.id),
    )


CASES = (
    Case(
        "berlin",
        "berlin-navigation-noise.html",
        "BMW Berlin Marathon",
        "Berlin",
        "Germany",
        "2026-09-27",
        "no_change",
        "event_date",
        None,
        ("confirmed", "confirmed", "confirmed"),
        ("BMW Berlin Marathon", "42.195 km"),
        ("Kids & Youth Mini Marathon",),
    ),
    Case(
        "baden-mini",
        "baden-mini-marathon.html",
        "Baden Marathon",
        "Karlsruhe",
        "Germany",
        "2027-09-19",
        "inconclusive",
        None,
        None,
        ("confirmed", "inconclusive", "rejected"),
        ("Baden Mini Marathon", "4.2 km"),
    ),
    Case(
        "valencia",
        "valencia-waitlist.html",
        "Valencia Marathon",
        "Valencia",
        "Spain",
        "2026-12-06",
        "propose_update",
        "registration_status",
        "sold_out",
        ("confirmed", "confirmed", "confirmed"),
        ("sold out", "waiting list"),
        ("21K registration is open",),
    ),
    Case(
        "amsterdam",
        "amsterdam-mixed-faq.html",
        "Amsterdam Marathon",
        "Amsterdam",
        "Netherlands",
        "2026-10-18",
        "propose_update",
        "registration_close_at",
        "2026-09-30",
        ("confirmed", "confirmed", "confirmed"),
        ("full marathon", "planned closing date"),
        ("Kids race registration is open",),
    ),
    Case(
        "copenhagen",
        "copenhagen-replacement.html",
        "Copenhagen Marathon",
        "Copenhagen",
        "Denmark",
        "2026-05-10",
        "inconclusive",
        None,
        None,
        ("confirmed", "confirmed", "confirmed"),
        ("normal event is paused", "Harbor 42K Replacement", "17 May 2027"),
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_fixture_corpus_replays_through_shadow_service(case: Case, tmp_path: Path) -> None:
    source_url = f"https://{case.name}.example/current"
    agent = CorpusAgent(
        lambda request: _decision(request, case.action, case.field, case.value)
    )
    run = _run(
        tmp_path / case.name,
        _event(case, source_url),
        {source_url: _fixture(case.fixture)},
        agent,
        persist_queue=case.action != "propose_update",
    )

    evidence = agent.assessment_requests[0].evidence[0]
    assert all(text in evidence.primary_text for text in case.primary_contains)
    assert all(text not in evidence.primary_text for text in case.primary_excludes)
    applicability = agent.decisions[0].applicability[0]
    assert (
        applicability.event_identity,
        applicability.event_edition,
        applicability.distance_category,
    ) == case.verdicts
    assert run.after == run.before
    if case.action == "propose_update":
        proposed = agent.decisions[0].proposed_fields
        assert proposed is not None
        assert proposed.model_dump(mode="json", exclude_none=True)[case.field] == case.value
        if case.name == "amsterdam":
            assert proposed.registration_status is None
        assert run.result.status.detail.startswith("Shadow mode validated")
    elif case.action == "inconclusive":
        assert run.result.status.outcome == "inconclusive"
        assert run.result.queue_reference is None
    else:
        assert run.result.status.outcome == "no_change"

    stored = run.artifacts.read_artifact(evidence.reference)["content"]
    assert stored["primary_text"] == evidence.primary_text
    assert stored["chrome_text"] == evidence.chrome_text

    if case.name == "berlin":
        reordered = _fixture(case.fixture).replace(
            '<nav><a href="/kids-youth">Kids &amp; Youth Mini Marathon</a></nav>',
            (
                '<nav><a href="/half">Berlin Half Marathon</a>'
                '<a href="/kids-youth">Kids &amp; Youth Mini Marathon</a></nav>'
            ),
        )
        variant_agent = CorpusAgent(
            lambda request: _decision(request, case.action, case.field, case.value)
        )
        variant = _run(
            tmp_path / "berlin-reordered",
            _event(case, source_url, "berlin-reordered.42"),
            {source_url: reordered},
            variant_agent,
        )
        variant_evidence = variant_agent.assessment_requests[0].evidence[0]
        assert variant_evidence.primary_text == evidence.primary_text
        variant_applicability = variant_agent.decisions[0].applicability[0]
        assert (
            variant_applicability.event_identity,
            variant_applicability.event_edition,
            variant_applicability.distance_category,
            variant_applicability.applicable_fields,
        ) == (
            applicability.event_identity,
            applicability.event_edition,
            applicability.distance_category,
            applicability.applicable_fields,
        )
        assert variant.result.status.outcome == run.result.status.outcome


def test_baden_overview_requests_distance_specific_evidence(tmp_path: Path) -> None:
    source_url = "https://baden.example/2027/overview"
    candidate_url = "https://baden.example/2027/registration"

    def request_evidence(_request: AssessmentRequest) -> EvidenceRequest:
        return EvidenceRequest(
            action="request_evidence",
            purpose="registration_status",
            query="full marathon standard registration status",
            gap="The multi-distance overview has no current standard marathon status.",
        )

    candidate = ResearchCandidate(
        source_url=candidate_url,
        title="Baden registration details",
        snippet="Distance-specific registration page.",
        event_date="2027-09-19",
        distances=("marathon",),
    )
    agent = CorpusAgent(request_evidence, scout_batches=((candidate,),))
    case = replace(CASES[1], name="baden-overview", fixture="baden-overview.html")
    run = _run(
        tmp_path / case.name,
        _event(case, source_url),
        {
            source_url: _fixture("baden-overview.html"),
            candidate_url: _fixture("baden-mini-marathon.html"),
        },
        agent,
    )

    assert run.result.status.outcome == "inconclusive"
    assert run.after == run.before
    assert len(agent.assessment_requests) == len(agent.scout_requests) == 1
    assert "2027 marathon registration_status" in agent.scout_requests[0].query
    assert run.fetcher.calls[1] == (candidate_url, source_url)
    assert "wrong event or distance/category" in run.result.status.detail


@pytest.mark.parametrize(("first", "second"), (("en", "tr"), ("tr", "en")))
def test_istanbul_conflict_is_explicit_and_order_independent(
    first: str,
    second: str,
    tmp_path: Path,
) -> None:
    source_url = f"https://istanbul.example/{first}/2026/marathon"
    candidate_url = f"https://istanbul.example/{second}/2026/marathon"

    def assess(request: AssessmentRequest) -> ResearchDecision | EvidenceRequest:
        if len(request.evidence) == 1:
            return EvidenceRequest(
                action="request_evidence",
                purpose="conflict_resolution",
                query="official 42K status other language",
                gap="A second official-language page is needed.",
            )
        references = tuple(item.reference for item in request.evidence)
        return ResearchDecision.model_validate(
            {
                "action": "inconclusive",
                "summary": "Official current 42K status claims conflict.",
                "evidence": references,
                "applicability": [
                    _applicability(request, index, "registration_status")
                    for index in range(2)
                ],
                "conflicts": [
                    {
                        "field": "registration_status",
                        "evidence": references,
                        "summary": "Official pages conflict: open versus sold out.",
                    }
                ],
            }
        )

    candidate = ResearchCandidate(
        source_url=candidate_url,
        title="Istanbul Marathon 2026 official status",
        snippet="Official current 42K registration status.",
        event_date="2026-11-01",
        distances=("marathon",),
    )
    agent = CorpusAgent(assess, scout_batches=((candidate,),))
    case = Case(
        f"istanbul-{first}",
        f"istanbul-conflict-{first}.html",
        "Istanbul Marathon",
        "Istanbul",
        "Türkiye",
        "2026-11-01",
        "inconclusive",
        None,
        None,
        ("confirmed", "confirmed", "confirmed"),
        (),
    )
    run = _run(
        tmp_path / first,
        _event(case, source_url),
        {
            source_url: _fixture(case.fixture),
            candidate_url: _fixture(f"istanbul-conflict-{second}.html"),
        },
        agent,
    )

    assert run.result.status.outcome == "inconclusive"
    assert run.after == run.before
    conflict = agent.decisions[0].conflicts[0]
    assert conflict.field == "registration_status"
    assert {reference.source_url for reference in conflict.evidence} == {
        source_url,
        candidate_url,
    }


def test_identity_edition_distance_and_status_mutate_one_at_a_time(
    tmp_path: Path,
) -> None:
    source_url = "https://berlin.example/marathon"
    base = _fixture("berlin-navigation-noise.html").replace(
        "</main>",
        (
            "<p>Standard 42K registration for the BMW Berlin Marathon 2026 "
            "is closed.</p></main>"
        ),
    )
    mutations = {
        "baseline": base,
        "identity": base.replace("BMW Berlin Marathon", "Hamburg Marathon"),
        "edition": base.replace("2026", "2025"),
        "distance": base.replace("42.195 km marathon", "4.2 km youth race"),
        "status": base.replace("is closed", "is open"),
    }
    observed = {}
    berlin = CASES[0]

    for label, html in mutations.items():
        def assess(request: AssessmentRequest) -> ResearchDecision:
            applicability = _applicability(request, 0, "registration_status")
            verdicts = tuple(
                applicability[name]
                for name in ("event_identity", "event_edition", "distance_category")
            )
            if verdicts != ("confirmed", "confirmed", "confirmed"):
                return _decision(request, "inconclusive", "registration_status")
            status = "open" if "is open" in request.evidence[0].primary_text else "closed"
            return _decision(request, "propose_update", "registration_status", status)

        agent = CorpusAgent(assess)
        run = _run(
            tmp_path / label,
            _event(berlin, source_url, f"berlin-{label}.42"),
            {source_url: html},
            agent,
        )
        decision = agent.decisions[0]
        applicability = decision.applicability[0]
        status = (
            decision.proposed_fields.registration_status
            if decision.proposed_fields is not None
            else None
        )
        observed[label] = (
            (
                applicability.event_identity,
                applicability.event_edition,
                applicability.distance_category,
            ),
            decision.action,
            status,
        )
        assert run.after[0] == run.before[0]
        assert run.after[1] == run.before[1]
        if label in {"identity", "edition", "distance"}:
            assert run.after == run.before
            assert run.result.status.outcome == "inconclusive"
        else:
            assert run.result.status.outcome == "proposal_created"

    assert observed == {
        "baseline": (("confirmed", "confirmed", "confirmed"), "propose_update", "closed"),
        "identity": (("rejected", "confirmed", "confirmed"), "inconclusive", None),
        "edition": (("confirmed", "rejected", "confirmed"), "inconclusive", None),
        "distance": (("confirmed", "confirmed", "rejected"), "inconclusive", None),
        "status": (("confirmed", "confirmed", "confirmed"), "propose_update", "open"),
    }


def test_javascript_challenge_is_unusable_with_renderer_disabled(tmp_path: Path) -> None:
    source_url = "https://shell.example/marathon"
    challenge = """
    <html><head><title>Just a moment</title></head>
    <body><main>Enable JavaScript and cookies to continue.</main></body></html>
    """

    def should_not_assess(_request: AssessmentRequest) -> ResearchDecision:
        raise AssertionError("Blocked shells must not reach the assessor.")

    case = replace(
        CASES[0],
        name="shell",
        event_name="Shell Marathon",
        city="Shell City",
        event_date="2026-10-04",
    )
    agent = CorpusAgent(should_not_assess)
    run = _run(
        tmp_path / "shell",
        _event(case, source_url),
        {source_url: challenge},
        agent,
    )

    assert run.result.status.outcome == "inconclusive"
    assert run.after == run.before
    assert not agent.assessment_requests
    assert "site protection challenge page" in run.result.status.detail
