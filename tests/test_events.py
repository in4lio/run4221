from dataclasses import replace

from run4221.bot.formatting import format_event_detail
from run4221.bot.keyboards import (
    CANCEL_BUTTON,
    EVENT_DETAIL_PREFIX,
    dialog_keyboard,
    event_detail_callback,
    event_detail_keyboard,
    remove_dialog_keyboard,
)
from run4221.events import (
    find_event,
    list_events,
    list_events_by_tag,
    list_open_events,
    load_seed_events,
    normalize_event_id,
    normalize_tag,
    resolve_event_lookup,
    search_events,
)
from tests.seed_fixtures import sample_seed_events


def test_seed_events_include_initial_collections() -> None:
    events = sample_seed_events()

    assert len(events) == 8
    assert find_event("berlin.42", events) is not None
    assert find_event("berlin.21", events) is not None


def test_load_seed_events_returns_empty_when_file_is_missing(tmp_path) -> None:
    assert load_seed_events(tmp_path / "missing-events.json") == ()


def test_find_event_accepts_telegram_style_inputs() -> None:
    assert normalize_event_id("/show_event berlin.42") == "berlin.42"
    assert normalize_event_id("berlin_marathon") == "berlin-marathon"
    assert normalize_event_id("berlin-marathon@run4221bot") == "berlin-marathon"
    events = sample_seed_events()
    assert find_event("/show_event berlin.42", events) is not None
    assert find_event("berlin-marathon@run4221bot", events) is not None
    assert find_event("berlin.superhalf.21", events) is not None


def test_event_detail_shows_registration_window_fields() -> None:
    event = replace(
        sample_seed_events()[0],
        registration_status="announced",
        registration_open_at="2026-10-01",
        registration_open_precision="date_only",
        registration_close_at="2027-03-01",
    )

    detail = format_event_detail(event)

    assert "<b>Registration status</b>: announced" in detail
    assert "<b>Registration opens</b>: 2026-10-01" in detail
    assert "<b>Registration open precision</b>: date_only" in detail
    assert "<b>Registration closes</b>: 2027-03-01" in detail


def test_search_by_city_collection_and_distance() -> None:
    events = sample_seed_events()
    berlin = search_events("berlin", events)
    majors = search_events("world marathon majors", events)
    half_marathons = search_events("half marathon", events)
    half_marathon_alias = search_events("21k", events)
    copied_tag_labels = search_events("Germany, Half marathon", events)

    assert {event.public_id for event in berlin} == {"berlin.42", "berlin.21"}
    assert len(majors) == 4
    assert len(half_marathons) == 4
    assert len(half_marathon_alias) == 4
    assert {event.public_id for event in copied_tag_labels} == {"berlin.21"}


def test_search_short_region_codes_match_tokens_only() -> None:
    events = sample_seed_events()
    assert {event.public_id for event in search_events("de", events)} == {
        "berlin.42",
        "berlin.21",
    }
    assert {event.public_id for event in search_events("us", events)} == {
        "boston.42",
        "chicago.42",
        "nyc.42",
    }


def test_events_have_generic_public_tags() -> None:
    events = sample_seed_events()
    berlin_marathon = find_event("berlin.42", events)
    berlin_half = find_event("berlin.21", events)
    assert berlin_marathon is not None
    assert berlin_half is not None

    assert {"global", "eu", "de", "major", "marathon"}.issubset(berlin_marathon.tags)
    assert {"eu", "de", "superhalf", "half_marathon"}.issubset(berlin_half.tags)
    assert "42" not in berlin_marathon.tags
    assert "21" not in berlin_half.tags
    assert berlin_marathon.tag_label == "global, eu, de, major, 42"
    assert berlin_half.tag_label == "eu, de, superhalf, 21"
    assert normalize_tag("#World-Marathon-Majors") == "major"
    assert normalize_tag("superhalfs") == "superhalf"
    assert normalize_tag("21K") == "half_marathon"
    assert normalize_tag("half") == "half_marathon"
    assert normalize_tag("Germany") == "de"
    assert normalize_tag("Deutschland") == "de"
    assert normalize_tag("World Marathon Majors") == "major"


def test_list_events_by_generic_tag() -> None:
    events = sample_seed_events()

    assert len(list_events_by_tag("major", events, limit=20)) == 4
    assert len(list_events_by_tag("superhalf", events, limit=20)) == 4
    assert len(list_events_by_tag("21", events, limit=20)) == 4
    assert len(list_events_by_tag("Half marathon", events, limit=20)) == 4
    assert {event.public_id for event in list_events_by_tag("de", events)} == {
        "berlin.42",
        "berlin.21",
    }
    assert {event.public_id for event in list_events_by_tag("Germany", events)} == {
        "berlin.42",
        "berlin.21",
    }
    assert {event.public_id for event in list_events_by_tag("Deutschland", events)} == {
        "berlin.42",
        "berlin.21",
    }


def test_list_open_events_can_filter_by_tag() -> None:
    events = tuple(
        event
        for event in sample_seed_events()
        if event.public_id in {"berlin.42", "berlin.21"}
    )
    open_events = tuple(replace(event, registration_status="open") for event in events)

    assert {event.public_id for event in list_open_events(open_events, tag="de")} == {
        "berlin.42",
        "berlin.21",
    }
    assert [event.public_id for event in list_open_events(open_events, tag="major")] == [
        "berlin.42"
    ]


def test_seed_data_does_not_use_alpha3_region_variants() -> None:
    forbidden = {"deu", "rus", "usa"}
    events = sample_seed_events()

    for event in events:
        assert forbidden.isdisjoint(event.regions)
        assert forbidden.isdisjoint(event.search_keywords)

    assert not search_events("deu", events)
    assert not search_events("usa", events)


def test_event_lookup_falls_back_to_search_suggestions() -> None:
    events = sample_seed_events()
    exact = resolve_event_lookup("berlin.42", events)
    single_suggestion = resolve_event_lookup("edp lisbon", events)
    multiple_suggestions = resolve_event_lookup("berlin", events)

    assert exact.exact is not None
    assert exact.exact.public_id == "berlin.42"
    assert exact.suggestions == ()

    assert single_suggestion.exact is None
    assert [event.public_id for event in single_suggestion.suggestions] == ["lisbon.21"]

    assert multiple_suggestions.exact is None
    assert {event.public_id for event in multiple_suggestions.suggestions} == {
        "berlin.42",
        "berlin.21",
    }


def test_list_events_sorts_known_dates_first() -> None:
    events = list_events(sample_seed_events(), limit=3)

    assert [event.public_id for event in events] == [
        "lisbon.21",
        "prague.21",
        "copenhagen.21",
    ]


def test_event_detail_keyboard_uses_one_event_public_id() -> None:
    event = find_event("lisbon.21", sample_seed_events())
    assert event is not None

    keyboard = event_detail_keyboard(event)
    button = keyboard.inline_keyboard[0][0]

    assert button.text == "Show"
    assert button.callback_data == event_detail_callback("lisbon.21")
    assert button.callback_data.startswith(EVENT_DETAIL_PREFIX)
    assert len(button.callback_data) <= 64


def test_dialog_keyboard_adds_standard_cancel_button() -> None:
    keyboard = dialog_keyboard("ok", "-")

    assert [button.text for button in keyboard.keyboard[0]] == ["ok", "-", CANCEL_BUTTON]
    assert keyboard.resize_keyboard is True
    assert keyboard.one_time_keyboard is True


def test_remove_dialog_keyboard_hides_standard_input_buttons() -> None:
    keyboard = remove_dialog_keyboard()

    assert keyboard.remove_keyboard is True
