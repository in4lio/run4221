from run4221.ingestion.event_identity import (
    conflicts_with_event_identity,
    select_registration_url_for_distances,
)


def test_registration_url_selection_uses_confirmed_distance() -> None:
    candidates = (
        ("https://www.badenmarathon.de/wettbewerbe/halbmarathon", "Halbmarathon"),
        ("https://www.badenmarathon.de/wettbewerbe/marathon", "Marathon"),
    )

    assert (
        select_registration_url_for_distances(
            candidates,
            ("marathon",),
            fallback="https://www.badenmarathon.de/wettbewerbe/halbmarathon",
        )
        == "https://www.badenmarathon.de/wettbewerbe/marathon"
    )
    assert (
        select_registration_url_for_distances(
            candidates,
            ("half_marathon",),
            fallback="https://www.badenmarathon.de/wettbewerbe/marathon",
        )
        == "https://www.badenmarathon.de/wettbewerbe/halbmarathon"
    )
    assert (
        select_registration_url_for_distances(
            (
                (
                    "https://example.com/kids-and-youth/mini-marathon",
                    "Mini Marathon",
                ),
                ("https://example.com/races/marathon", "Marathon"),
            ),
            ("marathon",),
        )
        == "https://example.com/races/marathon"
    )


def test_registration_url_identity_checks_do_not_match_word_fragments_or_years() -> None:
    marathon_url = "https://example.com/2021/skid-row/marathon-registration"
    half_marathon_url = "https://example.com/2021/halfmarathon-registration"

    assert (
        select_registration_url_for_distances(
            ((marathon_url, "Marathon registration"),),
            ("marathon",),
        )
        == marathon_url
    )
    assert (
        select_registration_url_for_distances(
            ((half_marathon_url, "Half marathon registration"),),
            ("marathon",),
        )
        is None
    )


def test_mixed_distance_event_accepts_its_own_half_and_full_pages() -> None:
    distances = ("marathon", "half_marathon")

    assert not conflicts_with_event_identity(
        "https://race.example/halbmarathon/anmeldung", "Halbmarathon", distances
    )
    assert not conflicts_with_event_identity(
        "https://race.example/marathon/anmeldung", "42 km Marathon", distances
    )
    assert conflicts_with_event_identity(
        "https://race.example/mini-marathon", "Mini Marathon", distances
    )


def test_single_distance_event_still_rejects_other_distance_pages() -> None:
    assert conflicts_with_event_identity(
        "https://race.example/halbmarathon", "", ("marathon",)
    )
    assert conflicts_with_event_identity(
        "https://race.example/42km", "", ("half_marathon",)
    )
