import asyncio
from types import SimpleNamespace

from run4221.bot import suggestions
from run4221.db.repository import EventWriteError


class FakeMessage:
    def __init__(self, text: str = "", from_user=None) -> None:
        self.text = text
        self.from_user = from_user
        self.answers: list[str] = []
        self.answer_kwargs: list[dict] = []

    async def answer(self, text: str, **kwargs) -> None:
        self.answers.append(text)
        self.answer_kwargs.append(kwargs)


class FakeState:
    def __init__(self) -> None:
        self.state = None
        self.data: dict[str, object] = {}
        self.cleared = False

    async def set_state(self, state) -> None:
        self.state = state

    async def update_data(self, **kwargs) -> None:
        self.data.update(kwargs)

    async def get_data(self) -> dict[str, object]:
        return dict(self.data)

    async def clear(self) -> None:
        self.state = None
        self.data.clear()
        self.cleared = True


def run(coro) -> None:
    asyncio.run(coro)


def test_suggest_flow_stores_pending_request(monkeypatch) -> None:
    captured = {}

    def fake_add_event_suggestion(payload):
        captured["payload"] = payload
        return SimpleNamespace(id=7)

    monkeypatch.setattr(suggestions, "add_event_suggestion", fake_add_event_suggestion)
    monkeypatch.setattr(suggestions, "is_moderator_account", lambda user_id, username: False)
    user = SimpleNamespace(id=42, username="runner", full_name="Test Runner")
    state = FakeState()

    run(suggestions.handle_suggest(FakeMessage(from_user=user), state))
    run(suggestions.handle_suggest_name(FakeMessage("Baden Marathon", user), state))
    url_message = FakeMessage("https://www.badenmarathon.de/", user)
    run(suggestions.handle_suggest_url(url_message, state))
    assert "Supported" not in url_message.answers[0]
    distance_keyboard = url_message.answer_kwargs[0]["reply_markup"]
    assert [button.text for button in distance_keyboard.keyboard[0]] == ["42", "21", "42,21"]
    assert [button.text for button in distance_keyboard.keyboard[1]] == ["Cancel"]
    run(suggestions.handle_suggest_distances(FakeMessage("42", user), state))
    final_message = FakeMessage("Please track registration opening.", user)
    run(suggestions.handle_suggest_note(final_message, state))

    payload = captured["payload"]
    assert payload.event_name == "Baden Marathon"
    assert payload.url == "https://www.badenmarathon.de/"
    assert payload.event_date is None
    assert payload.location is None
    assert payload.region_tags == ()
    assert payload.distances == ("marathon",)
    assert payload.note == "Please track registration opening."
    assert payload.submitter_user_id == "42"
    assert payload.submitter_username == "runner"
    assert payload.submitter_is_moderator is False
    assert state.cleared
    assert final_message.answers == [
        "Suggestion submitted.\n"
        "<b>Request ID</b>: <code>#7</code>\n"
        "A moderator will review it. Publication is not guaranteed."
    ]


def test_suggest_cancel_button_clears_state() -> None:
    message = FakeMessage(text="Cancel")
    state = FakeState()
    state.state = suggestions.SuggestEventStates.name

    run(suggestions.handle_suggest_cancel_button(message, state))

    assert state.cleared
    assert message.answers == ["Cancelled."]
    assert message.answer_kwargs[0]["reply_markup"].remove_keyboard is True


def test_suggest_distance_parser_accepts_known_codes() -> None:
    assert suggestions.parse_distances("42,21") == ("marathon", "half_marathon")


def test_suggest_flow_shows_queue_limit_errors(monkeypatch) -> None:
    def fake_add_event_suggestion(payload):
        raise EventWriteError("You already have 3 pending suggestions.")

    monkeypatch.setattr(suggestions, "add_event_suggestion", fake_add_event_suggestion)
    monkeypatch.setattr(suggestions, "is_moderator_account", lambda user_id, username: False)
    user = SimpleNamespace(id=42, username="runner", full_name="Test Runner")
    state = FakeState()
    state.data = {
        "event_name": "Baden Marathon",
        "url": "https://www.badenmarathon.de/",
        "distances": ("marathon",),
    }
    message = FakeMessage("-", user)

    run(suggestions.handle_suggest_note(message, state))

    assert message.answers == [
        "Could not save suggestion: You already have 3 pending suggestions."
    ]


def test_suggest_flow_marks_moderator_suggestions(monkeypatch) -> None:
    captured = {}

    def fake_add_event_suggestion(payload):
        captured["payload"] = payload
        return SimpleNamespace(id=8)

    monkeypatch.setattr(suggestions, "add_event_suggestion", fake_add_event_suggestion)
    monkeypatch.setattr(suggestions, "is_moderator_account", lambda user_id, username: True)
    user = SimpleNamespace(id=42, username="in4lio", full_name="Moderator")
    state = FakeState()
    state.data = {
        "event_name": "Baden Marathon",
        "url": "https://www.badenmarathon.de/",
        "distances": ("marathon",),
    }

    run(suggestions.handle_suggest_note(FakeMessage("-", user), state))

    assert captured["payload"].submitter_is_moderator is True
