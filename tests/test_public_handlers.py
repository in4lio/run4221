import asyncio
from types import SimpleNamespace

from aiogram.filters import CommandObject

from run4221.bot import public
from run4221.events import search_events
from tests.seed_fixtures import sample_seed_events


class FakeMessage:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.answers: list[str] = []
        self.answer_kwargs: list[dict] = []

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append(text)
        self.answer_kwargs.append(kwargs)


class FakeState:
    def __init__(self) -> None:
        self.state = None
        self.cleared = False

    async def set_state(self, state) -> None:
        self.state = state

    async def clear(self) -> None:
        self.state = None
        self.cleared = True


def run_handler(coro) -> None:
    asyncio.run(coro)


def test_search_handler_returns_search_results(monkeypatch) -> None:
    events = sample_seed_events()
    monkeypatch.setattr(public, "search_events", lambda query: search_events(query, events))
    message = FakeMessage()

    run_handler(
        public.handle_search(
            message,
            CommandObject(command="search_events", args="berlin"),
            FakeState(),
        )
    )

    assert message.answers[0] == "<b>✨ Search results for: berlin</b>"
    assert "BMW Berlin Marathon" in message.answers[1]
    assert "Generali Berlin Half Marathon" in message.answers[2]


def test_search_handler_asks_for_missing_query() -> None:
    message = FakeMessage()
    state = FakeState()

    run_handler(
        public.handle_search(
            message,
            CommandObject(command="search_events", args=None),
            state,
        )
    )

    assert state.state == public.SearchEventsStates.query
    assert message.answers == [
        "<b>💬 Search events</b>\n"
        "Send a search term.\n"
        "<b>Example</b>: <i>berlin</i>"
    ]
    keyboard = message.answer_kwargs[0]["reply_markup"]
    assert [button.text for button in keyboard.keyboard[0]] == ["Cancel"]


def test_dialog_cancel_button_clears_public_state() -> None:
    message = FakeMessage(text="Cancel")
    state = FakeState()
    state.state = public.SearchEventsStates.query

    run_handler(public.handle_dialog_cancel_button(message, state))

    assert state.cleared
    assert message.answers == ["Cancelled."]
    assert message.answer_kwargs[0]["reply_markup"].remove_keyboard is True


def test_search_prompt_runs_query_and_clears_state(monkeypatch) -> None:
    events = sample_seed_events()
    monkeypatch.setattr(public, "search_events", lambda query: search_events(query, events))
    message = FakeMessage(text="berlin")
    state = FakeState()

    run_handler(public.handle_search_query(message, state))

    assert state.cleared
    assert message.answer_kwargs[0]["reply_markup"].remove_keyboard is True
    assert message.answers[0] == "<b>✨ Search results for: berlin</b>"
    assert "BMW Berlin Marathon" in message.answers[1]


def test_show_event_handler_asks_for_missing_event_id() -> None:
    message = FakeMessage()
    state = FakeState()

    run_handler(
        public.handle_event(
            message,
            CommandObject(command="show_event", args=None),
            state,
        )
    )

    assert state.state == public.ShowEventStates.event_id
    assert message.answers == [
        "<b>💬 Show event</b>\n"
        "Send an event ID.\n"
        "<b>Example</b>: <i>berlin.42</i>"
    ]
    keyboard = message.answer_kwargs[0]["reply_markup"]
    assert [button.text for button in keyboard.keyboard[0]] == ["Cancel"]


def test_show_event_prompt_shows_detail_and_clears_state(monkeypatch) -> None:
    event = sample_seed_events()[0]
    monkeypatch.setattr(
        public,
        "resolve_event_lookup",
        lambda event_id, limit: SimpleNamespace(exact=event, suggestions=()),
    )
    message = FakeMessage(text="berlin.42")
    state = FakeState()

    run_handler(public.handle_event_id(message, state))

    assert state.cleared
    assert message.answer_kwargs[0]["reply_markup"].remove_keyboard is True
    assert message.answers[0].startswith(f"<b>✨ {event.name}</b>\n\n")
    assert event.name in message.answers[0]


def test_open_handler_returns_open_empty_state(monkeypatch) -> None:
    monkeypatch.setattr(public, "list_open_events", lambda limit, tag=None: ())
    message = FakeMessage()

    run_handler(public.handle_open(message, CommandObject(command="list_open")))

    assert message.answers == ["No tracked registrations are currently open."]
    assert message.answer_kwargs[0] == {}


def test_open_handler_filters_by_tag(monkeypatch) -> None:
    events = sample_seed_events()
    berlin_events = tuple(event for event in events if "de" in event.tags)
    monkeypatch.setattr(public, "list_open_events", lambda limit, tag=None: berlin_events[:limit])
    message = FakeMessage()

    run_handler(public.handle_open(message, CommandObject(command="list_open", args="de")))

    assert message.answers[0] == "<b>✨ Open registrations tagged: de</b>"
    assert "BMW Berlin Marathon" in message.answers[1]


def test_events_handler_filters_by_tag(monkeypatch) -> None:
    events = sample_seed_events()
    superhalf_events = tuple(event for event in events if "superhalf" in event.tags)
    monkeypatch.setattr(public, "list_events_by_tag", lambda tag, limit: superhalf_events[:limit])
    message = FakeMessage()

    run_handler(
        public.handle_events(
            message, CommandObject(command="list_events", args="superhalf")
        )
    )

    assert message.answers[0] == "<b>✨ Events tagged: superhalf</b>"
    assert "Generali Berlin Half Marathon" in message.answers[1]


def test_events_handler_uses_compact_tag_display(monkeypatch) -> None:
    events = tuple(event for event in sample_seed_events() if "half_marathon" in event.tags)
    monkeypatch.setattr(public, "list_events_by_tag", lambda tag, limit: events[:limit])
    message = FakeMessage()

    run_handler(
        public.handle_events(
            message, CommandObject(command="list_events", args="Half marathon")
        )
    )

    assert message.answers[0] == "<b>✨ Events tagged: 21</b>"


def test_start_uses_help_text() -> None:
    message = FakeMessage()

    run_handler(public.handle_start(message))

    assert len(message.answers) == 1
    assert message.answers[0].startswith(
        "Welcome to Full’n’Half running events tracker!\n\n"
        "I can help you track marathon and half marathon registration openings. "
        "Content is managed by AI agents, with public updates posted in"
    )
    assert '<a href="https://t.me/run4221">@run4221</a>' in message.answers[0]
    assert '<a href="https://run4221.com">run4221.com</a>' in message.answers[0]
    assert "You can control me by sending these commands:" in message.answers[0]
    assert "<b>Events</b>" not in message.answers[0]
    assert "<b>Suggestions</b>" not in message.answers[0]
    assert "/list_events &lt;tag&gt; - list events by tag, e.g. <i>major</i>" in message.answers[0]
    assert "/search_events &lt;query&gt;" in message.answers[0]
    assert "/show_event &lt;event_id&gt;" in message.answers[0]
    assert "/suggest - suggest a new event for tracking" in message.answers[0]
    assert "/list_events <tag>" not in message.answers[0]


def test_help_text_includes_moderator_commands_for_moderators() -> None:
    public_help = public.format_help_text(include_moderator=False)
    moderator_help = public.format_help_text(include_moderator=True)

    assert "/add_event" not in public_help
    assert "/archive_event" not in public_help
    assert "/remove_event" not in moderator_help
    assert "<b>Management</b>" in moderator_help
    assert "<b>Events</b>" in moderator_help
    assert "<b>Updates</b>" in moderator_help
    assert "<b>Suggestions</b>" in moderator_help
    assert "<b>Moderator Events</b>" not in moderator_help
    assert "<b>Moderator Queues</b>" not in moderator_help
    assert "<b>Guided Flows</b>" not in moderator_help
    assert "/cancel - cancel current guided flow" not in moderator_help
    assert "/add_event &lt;url&gt; - add event from URL" in moderator_help
    assert "https://example[.]com/race" in moderator_help
    assert "https://example.com/race" not in moderator_help
    assert "/edit_event &lt;event_id&gt; - edit event fields" in moderator_help
    assert "/update_event &lt;event_id&gt; - run registration scan" in moderator_help
    assert "/todo - show pending work" in moderator_help
    assert "/list_updates - list pending updates" in moderator_help
    assert "AI-detected" not in moderator_help
    assert "/next_update - show oldest pending update" in moderator_help
    assert "/show_update &lt;update_id&gt; - show pending update details" in moderator_help
    assert "/apply_update &lt;update_id&gt; - apply a pending update" in moderator_help
    assert "/reject_update &lt;update_id&gt; - reject a pending update" in moderator_help
    assert "/apply_suggestion &lt;suggestion_id&gt; - apply pending suggestion" in moderator_help
    assert "/list_suggestions - list pending suggestions" in moderator_help
    assert "/next_suggestion - show oldest pending suggestion" in moderator_help
    assert "/show_suggestion &lt;suggestion_id&gt; - show pending suggestion" in moderator_help
    assert "/reject_suggestion &lt;suggestion_id&gt; - reject pending suggestion" in moderator_help
    assert "/list_suggestions 5" not in moderator_help
    assert (
        "/archive_event &lt;event_id&gt; - preview and confirm archive"
        in moderator_help
    )
    assert "/list_archive - list archived events" in moderator_help
    assert "/restore_event &lt;event_id&gt; - restore archived event" in moderator_help
    assert "/delete_event &lt;event_id&gt; - permanently delete event" in moderator_help
    assert "/list_open &lt;tag&gt; - list open registrations by tag, e.g. <i>us</i>" in public_help

    assert moderator_help.index("<b>Management</b>") < moderator_help.index("<b>Events</b>")
    assert moderator_help.index("/add_event") < moderator_help.index("/archive_event")
    assert moderator_help.index("/archive_event") < moderator_help.index("/delete_event")
    assert moderator_help.index("/delete_event") < moderator_help.index("/edit_event")
    assert moderator_help.index("/edit_event") < moderator_help.index("/list_archive")
    assert moderator_help.index("/list_archive") < moderator_help.index("/restore_event")
    assert moderator_help.index("/restore_event") < moderator_help.index("/update_event")
    assert moderator_help.index("/apply_update") < moderator_help.index("/list_updates")
    assert moderator_help.index("/list_updates") < moderator_help.index("/next_update")
    assert moderator_help.index("/next_update") < moderator_help.index("/reject_update")
    assert moderator_help.index("/reject_update") < moderator_help.index("/show_update")
    assert moderator_help.index("/apply_suggestion") < moderator_help.index("/list_suggestions")
    assert moderator_help.index("/list_suggestions") < moderator_help.index("/next_suggestion")
    assert moderator_help.index("/next_suggestion") < moderator_help.index("/reject_suggestion")
    assert moderator_help.index("/reject_suggestion") < moderator_help.index("/show_suggestion")
