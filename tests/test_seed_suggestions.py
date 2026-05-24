import json

import pytest

from run4221.db.bootstrap import initialize_database
from run4221.db.prompts import get_runtime_prompt
from run4221.db.repository import (
    EventSuggestionCreate,
    EventWriteError,
    add_event_suggestion,
    count_event_suggestions,
    get_events,
    list_event_suggestions,
)
from run4221.db.seed import seed_initial_data
from run4221.db.seed_suggestions import (
    RESET_CONFIRMATION,
    load_suggestion_seed_file,
    main,
    reset_sqlite_database,
    seed_suggestions_from_file,
)
from run4221.db.session import session_scope
from tests.seed_fixtures import sample_seed_events


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-test.sqlite3'}"


def initialize_sample_database(url: str) -> None:
    initialize_database(url, seed_initial_events=False)
    with session_scope(url) as session:
        seed_initial_data(session, sample_seed_events())


def test_initialize_database_can_skip_initial_event_seed(tmp_path) -> None:
    url = database_url(tmp_path)

    initialize_database(url, seed_initial_events=False)

    assert get_events(url) == ()


def test_repository_schema_creation_does_not_seed_events(tmp_path) -> None:
    url = database_url(tmp_path)

    add_event_suggestion(
        EventSuggestionCreate(
            event_name="Stockholm Marathon",
            url="https://www.stockholmmarathon.se/",
            event_date=None,
            location=None,
            region_tags=("eu", "se"),
            distances=("marathon",),
            note=None,
            submitter_user_id="seed:launch",
            submitter_username="run4221_seed",
            submitter_display_name="Run4221 seed",
            submitter_is_moderator=True,
        ),
        database_url=url,
    )

    assert get_events(url) == ()
    assert count_event_suggestions(database_url=url) == 1


def test_startup_does_not_seed_events_when_suggestion_queue_exists(tmp_path) -> None:
    url = database_url(tmp_path)
    add_event_suggestion(
        EventSuggestionCreate(
            event_name="Stockholm Marathon",
            url="https://www.stockholmmarathon.se/",
            event_date=None,
            location=None,
            region_tags=("eu", "se"),
            distances=("marathon",),
            note=None,
            submitter_user_id="seed:launch",
            submitter_username="run4221_seed",
            submitter_display_name="Run4221 seed",
            submitter_is_moderator=True,
        ),
        database_url=url,
    )

    initialize_database(url)

    assert get_events(url) == ()
    assert count_event_suggestions(database_url=url) == 1


def test_seed_suggestions_from_file_creates_pending_suggestions_only(tmp_path) -> None:
    url = database_url(tmp_path)
    input_path = tmp_path / "launch_suggestions.json"
    input_path.write_text(
        json.dumps(
            {
                "suggestions": [
                    {
                        "event_name": "Berlin Marathon",
                        "url": "https://www.bmw-berlin-marathon.com/",
                        "distances": "42",
                        "tags": ["de", "eu", "major"],
                        "note": "Launch queue item.",
                    },
                    {
                        "name": "Lisbon Half Marathon",
                        "official_url": "https://www.lisbon-half-marathon.com/",
                        "distances": ["21K"],
                        "region_tags": "pt,eu,superhalf",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = seed_suggestions_from_file(input_path, database_url=url)
    suggestions = list_event_suggestions(database_url=url)

    assert result.backup_path is None
    assert [suggestion.event_name for suggestion in result.created] == [
        "Berlin Marathon",
        "Lisbon Half Marathon",
    ]
    assert get_events(url) == ()
    assert [suggestion.event_name for suggestion in suggestions] == [
        "Berlin Marathon",
        "Lisbon Half Marathon",
    ]
    assert suggestions[0].distances == ("marathon",)
    assert suggestions[1].distances == ("half_marathon",)
    assert suggestions[1].region_tags == ("pt", "eu", "superhalf")


def test_seed_suggestions_cli_can_seed_prompts(tmp_path, capsys) -> None:
    url = database_url(tmp_path)
    suggestions_path = tmp_path / "launch_suggestions.json"
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    suggestions_path.write_text(
        json.dumps([{"event_name": "Berlin Marathon", "distances": ["42"]}]),
        encoding="utf-8",
    )
    (prompts_dir / "discover_event_profile.instructions.txt").write_text(
        "Discover profile prompt.",
        encoding="utf-8",
    )

    main(
        [
            "--input",
            str(suggestions_path),
            "--database-url",
            url,
            "--prompts-dir",
            str(prompts_dir),
        ]
    )

    assert count_event_suggestions(database_url=url) == 1
    assert get_runtime_prompt("discover_event_profile", database_url=url).content == (
        "Discover profile prompt."
    )
    assert "Seeded 1 active prompt versions." in capsys.readouterr().out


def test_load_suggestion_seed_file_rejects_unknown_distance(tmp_path) -> None:
    input_path = tmp_path / "launch_suggestions.json"
    input_path.write_text(
        json.dumps([{"event_name": "City 10K", "distances": ["10"]}]),
        encoding="utf-8",
    )

    with pytest.raises(EventWriteError, match="Unsupported distance"):
        load_suggestion_seed_file(input_path)


def test_seed_suggestions_from_file_validates_before_reset(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_sample_database(url)
    database_path = tmp_path / "run4221-test.sqlite3"
    input_path = tmp_path / "broken_suggestions.json"
    input_path.write_text(json.dumps([{"distances": ["42"]}]), encoding="utf-8")

    with pytest.raises(EventWriteError, match="requires event_name"):
        seed_suggestions_from_file(
            input_path,
            database_url=url,
            reset_sqlite=True,
            confirm_reset=RESET_CONFIRMATION,
        )

    assert database_path.exists()
    assert len(get_events(url)) == 8


def test_seed_suggestions_from_file_checks_queue_size_before_reset(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_sample_database(url)
    database_path = tmp_path / "run4221-test.sqlite3"
    input_path = tmp_path / "too_many_suggestions.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "event_name": f"Launch Event {index}",
                    "distances": ["42"],
                }
                for index in range(31)
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(EventWriteError, match="too many pending suggestions"):
        seed_suggestions_from_file(
            input_path,
            database_url=url,
            reset_sqlite=True,
            confirm_reset=RESET_CONFIRMATION,
        )

    assert database_path.exists()
    assert len(get_events(url)) == 8


def test_seed_suggestions_from_file_checks_existing_queue_capacity(tmp_path) -> None:
    url = database_url(tmp_path)
    existing_path = tmp_path / "existing_suggestion.json"
    input_path = tmp_path / "launch_suggestions.json"
    existing_path.write_text(
        json.dumps(
            [
                {
                    "event_name": f"Existing Event {index}",
                    "distances": ["42"],
                }
                for index in range(30)
            ]
        ),
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps([{"event_name": "One More Marathon", "distances": ["42"]}]),
        encoding="utf-8",
    )
    seed_suggestions_from_file(existing_path, database_url=url)

    with pytest.raises(EventWriteError, match="does not have enough space"):
        seed_suggestions_from_file(input_path, database_url=url)

    assert count_event_suggestions(database_url=url) == 30


def test_reset_sqlite_database_requires_confirmation(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)

    with pytest.raises(EventWriteError, match=RESET_CONFIRMATION):
        reset_sqlite_database(url, confirm_reset=None)


def test_reset_sqlite_database_backs_up_existing_file(tmp_path) -> None:
    url = database_url(tmp_path)
    initialize_database(url)
    database_path = tmp_path / "run4221-test.sqlite3"

    backup_path = reset_sqlite_database(url, confirm_reset=RESET_CONFIRMATION)
    initialize_database(url, seed_initial_events=False)

    assert backup_path is not None
    assert backup_path.exists()
    assert database_path.exists()
    assert get_events(url) == ()
