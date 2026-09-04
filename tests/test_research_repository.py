from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import Barrier

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.db.models import EventSource, ProposedEventUpdate
from run4221.db.repository import (
    EventCreate,
    EventSuggestionCreate,
    ProposedEventUpdateCreate,
    add_event,
    add_event_suggestion,
    count_event_suggestions,
    count_proposed_event_updates,
    update_event_suggestion_status,
)
from run4221.db.research import (
    RESEARCHER_MAX_PENDING_SUGGESTIONS,
    admit_proposed_update,
    admit_suggestion,
    find_research_queue_reference,
    get_refresh_source,
    list_due_sources,
    mark_source_checked,
)
from run4221.db.session import session_scope


def database_url(tmp_path, *, query: str = "") -> str:
    return f"sqlite:///{tmp_path / 'research.sqlite3'}{query}"


def event_payload(
    *,
    public_id: str = "zurich.42",
    official_url: str = "https://example.com/zurich-marathon",
) -> EventCreate:
    return EventCreate(
        public_id=public_id,
        name="Zurich Marathon",
        city="Zurich",
        country="Switzerland",
        timezone="Europe/Zurich",
        event_date="2027-04-18",
        distances=("marathon",),
        regions=("global", "eu", "ch"),
        official_url=official_url,
    )


def suggestion_payload(index: int, *, url: str | None = None) -> EventSuggestionCreate:
    return EventSuggestionCreate(
        event_name=f"Research Marathon {index}",
        url=url or f"https://example.com/research-{index}",
        event_date="2027-09-20",
        location="Karlsruhe, Germany",
        region_tags=("eu", "de"),
        distances=("marathon",),
        note="research-run:test",
        submitter_user_id=None,
        submitter_username=None,
        submitter_display_name=None,
    )


def proposed_update_payload(
    event_id: str,
    *,
    field: str = "registration_status",
    value: str = "open",
) -> ProposedEventUpdateCreate:
    return ProposedEventUpdateCreate(
        event_id=event_id,
        update_type="registration_window",
        current_fields={field: "unknown" if field == "registration_status" else None},
        proposed_fields={field: value},
        evidence=("research-run:test",),
        confidence=0.9,
        change_summary=f"Propose {field}.",
    )


def test_research_facade_lists_due_sources_and_marks_last_check(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(event_payload(), database_url=url)
    now = datetime.now(UTC)

    due = list_due_sources(due_before=now, limit=10, database_url=url)

    assert [(item.event.id, item.url) for item in due] == [
        (event.id, "https://example.com/zurich-marathon")
    ]
    assert mark_source_checked(due[0].source_id, checked_at=now, database_url=url)
    assert list_due_sources(
        due_before=now - timedelta(seconds=1),
        limit=10,
        database_url=url,
    ) == ()


def test_get_refresh_source_prefers_active_registration_page(tmp_path) -> None:
    url = database_url(tmp_path)
    registration_url = "https://register.example/zurich-marathon"
    event = add_event(
        replace(event_payload(), registration_url=registration_url),
        database_url=url,
    )

    preferred = get_refresh_source(event.id, database_url=url)

    assert preferred is not None
    assert preferred.url == registration_url
    assert preferred.event.id == event.id

    with session_scope(url) as session:
        for source in session.query(EventSource).all():
            if source.source_type == "registration_page":
                source.active = False

    fallback = get_refresh_source(event.id, database_url=url)

    assert fallback is not None
    assert fallback.url == "https://example.com/zurich-marathon"
    assert get_refresh_source("missing.42", database_url=url) is None


def test_get_refresh_source_falls_back_to_official_site(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(event_payload(), database_url=url)

    source = get_refresh_source(event.id, database_url=url)

    assert source is not None
    assert source.url == "https://example.com/zurich-marathon"


@pytest.mark.parametrize("existing_status", ["tracked", "pending", "converted", "removed"])
def test_research_suggestion_deduplicates_urls_across_all_existing_states(
    tmp_path,
    existing_status: str,
) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    existing_url = "https://example.com/already-known"
    if existing_status == "tracked":
        add_event(
            event_payload(official_url=existing_url),
            database_url=url,
        )
    else:
        existing = add_event_suggestion(
            suggestion_payload(1, url=existing_url),
            database_url=url,
        )
        if existing_status != "pending":
            update_event_suggestion_status(existing.id, existing_status, database_url=url)

    result = admit_suggestion(
        suggestion_payload(2, url="HTTPS://EXAMPLE.COM/already-known/"),
        max_pending=0,
        database_url=url,
    )

    assert result.outcome == "duplicate"
    assert result.suggestion is None


def test_research_suggestion_reserves_subscriber_queue_capacity(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url, seed_initial_events=False)
    for index in range(RESEARCHER_MAX_PENDING_SUGGESTIONS):
        result = admit_suggestion(suggestion_payload(index), database_url=url)
        assert result.outcome == "admitted"

    capped = admit_suggestion(suggestion_payload(99), database_url=url)
    subscriber = add_event_suggestion(
        replace(suggestion_payload(100), submitter_user_id="subscriber-100"),
        database_url=url,
    )

    assert capped.outcome == "queue_full"
    assert capped.suggestion is None
    assert subscriber.id == RESEARCHER_MAX_PENDING_SUGGESTIONS + 1
    assert count_event_suggestions(database_url=url) == 21


@pytest.mark.parametrize("existing_status", ["pending", "applying"])
def test_research_proposal_skips_conflicting_active_update(
    tmp_path,
    existing_status: str,
) -> None:
    url = database_url(tmp_path)
    event = add_event(event_payload(), database_url=url)

    first = admit_proposed_update(
        proposed_update_payload(event.id),
        database_url=url,
    )
    assert first.update is not None
    if existing_status == "applying":
        with session_scope(url) as session:
            stored = session.get(ProposedEventUpdate, first.update.id)
            assert stored is not None
            stored.status = "applying"

    conflict = admit_proposed_update(
        proposed_update_payload(
            event.id,
            field="registration_url",
            value="https://example.com/register",
        ),
        database_url=url,
    )

    assert first.outcome == "admitted"
    assert conflict.outcome == "conflicting_pending"
    assert conflict.update is None
    assert count_proposed_event_updates(status=existing_status, database_url=url) == 1


def test_research_proposal_respects_pending_queue_cap(tmp_path) -> None:
    url = database_url(tmp_path)
    first_event = add_event(event_payload(), database_url=url)
    second_event = add_event(
        event_payload(
            public_id="basel.42",
            official_url="https://example.com/basel-marathon",
        ),
        database_url=url,
    )
    first = admit_proposed_update(
        proposed_update_payload(first_event.id),
        max_pending=1,
        database_url=url,
    )

    capped = admit_proposed_update(
        proposed_update_payload(second_event.id),
        max_pending=1,
        database_url=url,
    )

    assert first.outcome == "admitted"
    assert capped.outcome == "queue_full"
    assert capped.update is None


def test_research_queue_resolver_finds_json_proposal_evidence(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(event_payload(), database_url=url)
    marker = "researcher-decision:v1 run=test artifact=prepared.json sha256=abc"
    admitted = admit_proposed_update(
        replace(proposed_update_payload(event.id), evidence=(marker,)),
        database_url=url,
    )

    assert admitted.update is not None
    assert find_research_queue_reference(
        "propose_update",
        decision_marker=marker,
        database_url=url,
    ) == f"proposed_event_update:{admitted.update.id}"


def test_research_suggestion_admission_serializes_independent_engines(tmp_path) -> None:
    base_url = database_url(tmp_path)
    initialize_database(base_url, seed_initial_events=False)
    engine_urls = (
        database_url(tmp_path, query="?timeout=1"),
        database_url(tmp_path, query="?timeout=2"),
    )
    ready = Barrier(2)

    def admit(index: int):
        ready.wait(timeout=5)
        return admit_suggestion(
            suggestion_payload(index, url="https://example.com/one-discovery"),
            database_url=engine_urls[index],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(admit, range(2)))

    assert sorted(result.outcome for result in results) == ["admitted", "duplicate"]
    assert count_event_suggestions(database_url=base_url) == 1


def test_research_proposal_admission_serializes_independent_engines(tmp_path) -> None:
    base_url = database_url(tmp_path)
    event = add_event(event_payload(), database_url=base_url)
    engine_urls = (
        database_url(tmp_path, query="?timeout=1"),
        database_url(tmp_path, query="?timeout=2"),
    )
    payloads = (
        proposed_update_payload(event.id),
        proposed_update_payload(
            event.id,
            field="registration_url",
            value="https://example.com/register",
        ),
    )
    ready = Barrier(2)

    def admit(index: int):
        ready.wait(timeout=5)
        return admit_proposed_update(payloads[index], database_url=engine_urls[index])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(admit, range(2)))

    assert sorted(result.outcome for result in results) == [
        "admitted",
        "conflicting_pending",
    ]
    assert count_proposed_event_updates(database_url=base_url) == 1
