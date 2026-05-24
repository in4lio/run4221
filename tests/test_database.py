import pytest

from run4221.db.bootstrap import initialize_database
from run4221.db.models import RegistrationWindow
from run4221.db.repository import (
    EVENT_SUGGESTION_MAX_PENDING_PER_USER,
    EVENT_SUGGESTION_MAX_PENDING_TOTAL,
    EventCreate,
    EventSuggestionCreate,
    EventUpdate,
    EventWriteError,
    ProposedEventUpdateCreate,
    add_event,
    add_event_suggestion,
    approve_proposed_event_update,
    archive_event,
    count_event_suggestions,
    count_proposed_event_updates,
    create_proposed_event_update,
    delete_event,
    find_event,
    find_event_by_url,
    get_event_suggestion,
    get_events,
    list_archived_events,
    list_event_suggestions,
    list_events,
    list_events_by_tag,
    list_events_by_url,
    list_open_events,
    list_proposed_event_updates,
    partial_apply_proposed_event_update,
    reject_proposed_event_update,
    resolve_event_lookup,
    restore_event,
    search_events,
    update_event,
    update_event_suggestion_status,
)
from run4221.db.seed import seed_initial_data
from run4221.db.session import session_scope
from tests.seed_fixtures import sample_seed_events


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-test.sqlite3'}"


def initialize_sample_database(url: str) -> None:
    initialize_database(url, seed_initial_events=False)
    with session_scope(url) as session:
        seed_initial_data(session, sample_seed_events())


def suggestion_payload(
    index: int = 1,
    *,
    submitter_user_id: str | None = "42",
    submitter_is_moderator: bool = False,
) -> EventSuggestionCreate:
    return EventSuggestionCreate(
        event_name=f"Baden Marathon {index}",
        url=f"https://www.badenmarathon.de/{index}",
        event_date="2026-09-20",
        location="Karlsruhe, Germany",
        region_tags=("eu", "de"),
        distances=("marathon",),
        note="Registration page should be checked.",
        submitter_user_id=submitter_user_id,
        submitter_username="runner",
        submitter_display_name="Test Runner",
        submitter_is_moderator=submitter_is_moderator,
    )


def test_database_seed_imports_initial_events(tmp_path) -> None:
    url = database_url(tmp_path)

    initialize_sample_database(url)
    events = get_events(url)

    assert len(events) == 8
    assert find_event("berlin.42", url) is not None
    assert find_event("berlin-half-marathon", url) is not None


def test_database_event_search_and_lookup_fallback(tmp_path) -> None:
    url = database_url(tmp_path)

    initialize_sample_database(url)
    berlin = search_events("berlin", database_url=url)
    lookup = resolve_event_lookup("edp lisbon", database_url=url)

    assert {event.public_id for event in berlin} == {"berlin.42", "berlin.21"}
    assert lookup.exact is None
    assert [event.public_id for event in lookup.suggestions] == ["lisbon.21"]
    assert len(list_events_by_tag("major", database_url=url, limit=20)) == 4
    assert len(list_events_by_tag("superhalf", database_url=url, limit=20)) == 4


def test_database_lists_sorted_events_and_open_empty_state(tmp_path) -> None:
    url = database_url(tmp_path)

    initialize_sample_database(url)

    assert [event.public_id for event in list_events(limit=3, database_url=url)] == [
        "lisbon.21",
        "prague.21",
        "copenhagen.21",
    ]
    assert list_open_events(database_url=url) == ()


def test_database_open_events_include_open_registration_windows(tmp_path) -> None:
    url = database_url(tmp_path)

    initialize_sample_database(url)
    with session_scope(url) as session:
        session.add(
            RegistrationWindow(
                event_id="boston-marathon",
                status="open",
                registration_open_precision="unknown",
            )
        )

    open_events = list_open_events(database_url=url)

    assert [event.public_id for event in open_events] == ["boston.42"]
    assert [event.public_id for event in list_open_events(database_url=url, tag="major")] == [
        "boston.42"
    ]
    assert list_open_events(database_url=url, tag="de") == ()


def test_database_approves_proposed_registration_update(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )
    proposed = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={
                "registration_status": "unknown",
                "registration_open_at": None,
                "registration_open_precision": "unknown",
                "registration_close_at": None,
                "registration_url": None,
                "event_date": "2027-04-18",
            },
            proposed_fields={
                "registration_status": "open",
                "registration_open_at": "2026-10-01",
                "registration_open_precision": "date_only",
                "registration_close_at": None,
                "registration_url": "https://example.com/zurich-marathon/register",
                "event_date": "2027-04-18",
            },
            evidence=("Registration is open.",),
            confidence=0.93,
            change_summary="Registration update proposed: registration_status.",
        ),
        database_url=url,
    )

    result = approve_proposed_event_update(proposed.id, reviewer_user_id="42", database_url=url)

    assert result is not None
    assert result.update.status == "applied"
    assert result.event.registration_status == "open"
    assert result.event.registration_url == "https://example.com/zurich-marathon/register"
    assert [event.public_id for event in list_open_events(database_url=url)] == ["zurich.42"]
    assert list_proposed_event_updates(database_url=url) == ()


def test_database_adds_event_with_registration_window_fields(tmp_path) -> None:
    url = database_url(tmp_path)

    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
            registration_url="https://example.com/zurich-marathon/register",
            registration_status="announced",
            registration_open_at="2026-10-01",
            registration_open_precision="date_only",
            registration_close_at="2027-03-01",
        ),
        database_url=url,
    )

    found = find_event("zurich.42", url)

    assert event.registration_status == "announced"
    assert event.registration_open_at == "2026-10-01"
    assert event.registration_open_precision == "date_only"
    assert event.registration_close_at == "2027-03-01"
    assert found is not None
    assert found.registration_open_at == "2026-10-01"
    assert found.registration_open_precision == "date_only"
    assert found.registration_close_at == "2027-03-01"


def test_database_rejects_proposed_registration_update(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )
    proposed = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "unknown"},
            proposed_fields={"registration_status": "sold_out"},
            evidence=("Old page says sold out.",),
            confidence=0.75,
            change_summary="Registration update proposed: registration_status.",
        ),
        database_url=url,
    )

    rejected = reject_proposed_event_update(proposed.id, reviewer_user_id="42", database_url=url)

    assert rejected is not None
    assert rejected.status == "rejected"
    assert list_proposed_event_updates(database_url=url) == ()
    assert find_event("zurich.42", url).registration_status == "unknown"


def test_database_partially_applies_proposed_registration_update(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )
    proposed = create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={
                "registration_status": "unknown",
                "registration_url": None,
                "event_date": "2027-04-18",
            },
            proposed_fields={
                "registration_status": "open",
                "registration_url": "https://example.com/zurich-marathon/register",
                "event_date": "2027-04-19",
            },
            evidence=("Registration is open.",),
            confidence=0.93,
            change_summary="Registration update proposed: registration_status.",
        ),
        database_url=url,
    )

    result = partial_apply_proposed_event_update(
        proposed.id,
        selected_fields=("registration_status",),
        reviewer_user_id="42",
        database_url=url,
    )

    assert result is not None
    assert result.update.status == "applied_partial"
    assert result.applied_fields == ("registration_status",)
    assert result.remaining_fields == ("registration_url", "event_date")
    assert result.event.registration_status == "open"
    assert result.event.registration_url is None
    assert result.event.event_date == "2027-04-18"
    assert result.follow_up_update is not None
    assert result.follow_up_update.status == "pending"
    assert result.follow_up_update.proposed_fields == {
        "registration_url": "https://example.com/zurich-marathon/register",
        "event_date": "2027-04-19",
    }
    assert result.follow_up_update.evidence[-1] == (
        f"Created from partial apply of update #{proposed.id}."
    )
    assert list_proposed_event_updates(database_url=url) == (result.follow_up_update,)


def test_database_counts_pending_updates_and_suggestions(tmp_path) -> None:
    url = database_url(tmp_path)
    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )
    add_event_suggestion(suggestion_payload(1), database_url=url)
    converted = add_event_suggestion(suggestion_payload(2), database_url=url)
    update_event_suggestion_status(converted.id, "converted", database_url=url)
    create_proposed_event_update(
        ProposedEventUpdateCreate(
            event_id=event.id,
            update_type="registration_window",
            current_fields={"registration_status": "unknown"},
            proposed_fields={"registration_status": "open"},
            evidence=("Registration is open.",),
            confidence=0.91,
            change_summary="Registration update proposed: registration_status.",
        ),
        database_url=url,
    )

    assert count_event_suggestions(database_url=url) == 1
    assert count_proposed_event_updates(database_url=url) == 1


def test_database_adds_and_soft_removes_moderator_event(tmp_path) -> None:
    url = database_url(tmp_path)

    event = add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
            registration_url="https://example.com/zurich-marathon/register",
        ),
        database_url=url,
    )

    assert event.public_id == "zurich.42"
    assert find_event("zurich.42", url) is not None
    assert [match.public_id for match in search_events("zurich.42", database_url=url)] == [
        "zurich.42"
    ]
    assert find_event_by_url("https://example.com/zurich-marathon/", url) is not None
    assert find_event_by_url("https://example.com/zurich-marathon/register", url) is not None

    removed = archive_event("zurich.42", url)

    assert removed is not None
    assert removed.public_id == "zurich.42"
    assert find_event("zurich.42", url) is None
    assert find_event_by_url("https://example.com/zurich-marathon", url) is None
    assert not search_events("zurich.42", database_url=url)


def test_database_lists_and_restores_archived_events(tmp_path) -> None:
    url = database_url(tmp_path)
    add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )

    assert archive_event("zurich.42", url) is not None

    archived = list_archived_events(database_url=url)

    assert len(archived) == 1
    assert archived[0].event.public_id == "zurich.42"
    assert archived[0].removed_at is not None
    assert find_event("zurich.42", url) is None

    restored = restore_event("zurich.42", url)

    assert restored is not None
    assert restored.public_id == "zurich.42"
    assert find_event("zurich.42", url) is not None
    assert list_archived_events(database_url=url) == ()


def test_database_permanently_deletes_active_and_archived_events(tmp_path) -> None:
    url = database_url(tmp_path)
    add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )
    add_event(
        EventCreate(
            public_id="basel.21",
            name="Basel Half Marathon",
            city="Basel",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-05-09",
            distances=("half_marathon",),
            regions=("eu", "ch"),
            official_url="https://example.com/basel-half",
        ),
        database_url=url,
    )
    assert archive_event("basel.21", url) is not None

    deleted_active = delete_event("zurich.42", url)
    deleted_archived = delete_event("basel.21", url)

    assert deleted_active is not None
    assert deleted_active.public_id == "zurich.42"
    assert deleted_archived is not None
    assert deleted_archived.public_id == "basel.21"
    assert find_event("zurich.42", url) is None
    assert find_event("basel.21", url) is None
    assert list_archived_events(database_url=url) == ()
    assert delete_event("zurich.42", url) is None


def test_database_can_readd_soft_removed_event_without_keyword_conflict(tmp_path) -> None:
    url = database_url(tmp_path)
    first_event = EventCreate(
        public_id="badenmarathon.42",
        name="Baden Marathon",
        city="Karlsruhe",
        country="Germany",
        timezone="Europe/Berlin",
        event_date="2026-09-20",
        distances=("marathon",),
        regions=("global", "eu", "de"),
        official_url="https://example.com/badenmarathon",
        registration_url="https://example.com/badenmarathon/register",
    )
    second_event = EventCreate(
        public_id="badenmarathon.42",
        name="Baden Marathon",
        city="Karlsruhe",
        country="Germany",
        timezone="Europe/Berlin",
        event_date="2027-09-19",
        distances=("marathon",),
        regions=("global", "eu", "de"),
        official_url="https://example.com/badenmarathon",
        registration_url="https://example.com/badenmarathon/registration",
    )

    add_event(first_event, database_url=url)
    assert archive_event("badenmarathon.42", url) is not None

    restored = add_event(second_event, database_url=url)

    assert restored.public_id == "badenmarathon.42"
    assert restored.event_date == "2027-09-19"
    assert restored.registration_url == "https://example.com/badenmarathon/registration"
    assert [
        match.public_id
        for match in search_events("42", database_url=url)
        if match.public_id == "badenmarathon.42"
    ] == ["badenmarathon.42"]


def test_database_updates_event_without_rewriting_public_id(tmp_path) -> None:
    url = database_url(tmp_path)
    add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
            registration_url="https://example.com/zurich-marathon/register",
            registration_status="announced",
            registration_open_at="2026-10-01",
            registration_open_precision="date_only",
        ),
        database_url=url,
    )

    updated = update_event(
        "zurich.42",
        EventUpdate(
            name="Zurich City Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            event_date="2027-04-25",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-city-marathon",
            registration_url=None,
            registration_status="open",
            registration_open_at="2026-10-02 09:00",
            registration_open_precision="datetime",
            registration_close_at="2027-03-01",
        ),
        database_url=url,
    )

    assert updated is not None
    assert updated.public_id == "zurich.42"
    assert updated.name == "Zurich City Marathon"
    assert updated.event_date == "2027-04-25"
    assert updated.registration_url is None
    assert updated.registration_status == "open"
    assert updated.registration_open_at == "2026-10-02 09:00"
    assert updated.registration_open_precision == "datetime"
    assert updated.registration_close_at == "2027-03-01"
    assert find_event("zurich.42", url).name == "Zurich City Marathon"
    assert find_event("zurich.42", url).registration_open_at == "2026-10-02 09:00"
    assert find_event_by_url("https://example.com/zurich-city-marathon", url) is not None
    assert find_event_by_url("https://example.com/zurich-marathon", url) is None


def test_database_update_keeps_public_id_distance_valid(tmp_path) -> None:
    url = database_url(tmp_path)
    add_event(
        EventCreate(
            public_id="zurich.42",
            name="Zurich Marathon",
            city="Zurich",
            country="Switzerland",
            timezone="Europe/Zurich",
            distances=("marathon",),
            regions=("global", "eu", "ch"),
            official_url="https://example.com/zurich-marathon",
        ),
        database_url=url,
    )

    with pytest.raises(EventWriteError):
        update_event(
            "zurich.42",
            EventUpdate(
                name="Zurich Half Marathon",
                city="Zurich",
                country="Switzerland",
                timezone="Europe/Zurich",
                distances=("half_marathon",),
                regions=("global", "eu", "ch"),
                official_url="https://example.com/zurich-half-marathon",
            ),
            database_url=url,
        )


def test_database_lists_all_active_events_sharing_url(tmp_path) -> None:
    url = database_url(tmp_path)
    shared_url = "https://example.com/shared-event"

    add_event(
        EventCreate(
            public_id="shared.42",
            name="Shared Marathon",
            city="Shared",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2027-04-18",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url=shared_url,
        ),
        database_url=url,
    )
    add_event(
        EventCreate(
            public_id="shared.21",
            name="Shared Half Marathon",
            city="Shared",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2027-04-18",
            distances=("half_marathon",),
            regions=("global", "eu", "de"),
            official_url=f"{shared_url}/",
        ),
        database_url=url,
    )

    matches = list_events_by_url(shared_url, url)

    assert {event.public_id for event in matches} == {"shared.42", "shared.21"}


def test_database_stores_pending_event_suggestions(tmp_path) -> None:
    url = database_url(tmp_path)

    suggestion = add_event_suggestion(
        suggestion_payload(),
        database_url=url,
    )

    pending = list_event_suggestions(database_url=url)

    assert suggestion.id == 1
    assert [item.event_name for item in pending] == ["Baden Marathon 1"]
    assert pending[0].url == "https://www.badenmarathon.de/1"
    assert pending[0].region_tags == ("eu", "de")
    assert pending[0].distances == ("marathon",)
    assert pending[0].submitter_username == "runner"

    converted = update_event_suggestion_status(
        suggestion.id,
        "converted",
        database_url=url,
    )

    assert converted is not None
    assert converted.status == "converted"
    assert list_event_suggestions(database_url=url) == ()
    assert get_event_suggestion(suggestion.id, database_url=url).status == "converted"
    assert get_event_suggestion(suggestion.id, status="pending", database_url=url) is None


def test_database_rejects_unknown_event_suggestion_status(tmp_path) -> None:
    url = database_url(tmp_path)

    with pytest.raises(EventWriteError):
        list_event_suggestions(status="archived", database_url=url)


def test_database_limits_pending_event_suggestions_per_user(tmp_path) -> None:
    url = database_url(tmp_path)

    suggestions = [
        add_event_suggestion(suggestion_payload(index), database_url=url)
        for index in range(1, EVENT_SUGGESTION_MAX_PENDING_PER_USER + 1)
    ]

    with pytest.raises(EventWriteError, match="3 pending suggestions"):
        add_event_suggestion(suggestion_payload(99), database_url=url)

    update_event_suggestion_status(suggestions[0].id, "converted", database_url=url)
    accepted = add_event_suggestion(suggestion_payload(100), database_url=url)

    assert accepted.id == EVENT_SUGGESTION_MAX_PENDING_PER_USER + 1


def test_database_allows_moderator_to_exceed_per_user_pending_limit(tmp_path) -> None:
    url = database_url(tmp_path)

    for index in range(EVENT_SUGGESTION_MAX_PENDING_PER_USER + 1):
        add_event_suggestion(
            suggestion_payload(index, submitter_is_moderator=True),
            database_url=url,
        )

    pending = list_event_suggestions(database_url=url, limit=10)

    assert len(pending) == EVENT_SUGGESTION_MAX_PENDING_PER_USER + 1


def test_database_limits_total_pending_event_suggestions(tmp_path) -> None:
    url = database_url(tmp_path)

    for index in range(EVENT_SUGGESTION_MAX_PENDING_TOTAL):
        add_event_suggestion(
            suggestion_payload(index, submitter_user_id=str(index)),
            database_url=url,
        )

    with pytest.raises(EventWriteError, match="Suggestion queue is full"):
        add_event_suggestion(
            suggestion_payload(99, submitter_user_id="overflow"),
            database_url=url,
        )
