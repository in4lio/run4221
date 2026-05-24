import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from run4221.ai.openai_provider import (
    OpenAIEventExtraction,
    OpenAIExtractorProvider,
    build_snapshot_prompt,
    candidate_links_for_extraction,
    format_openai_error,
)
from run4221.ai.provider_factory import (
    ExtractorProviderConfigError,
    get_extractor_provider,
)
from run4221.config import Settings
from run4221.db.prompts import DISCOVER_EVENT_PROFILE_PROMPT, upsert_active_prompt_version
from run4221.ingestion.page_snapshot import PageLink, PageSnapshot


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    async def parse(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            output_parsed=OpenAIEventExtraction(
                name="Karlsruhe Baden Marathon",
                public_id="badenmarathon.42",
                city="Karlsruhe",
                country="Germany",
                timezone="Europe/Berlin",
                event_date="2026-09-20",
                distances=["marathon"],
                regions=["global", "eu", "de"],
                official_url="https://example.com/badenmarathon",
                registration_url="https://example.com/register",
                confidence=0.91,
                evidence_snippets=["Race date: 20 September 2026"],
            )
        )


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-test.sqlite3'}"


def snapshot() -> PageSnapshot:
    return PageSnapshot(
        source_url="https://example.com/badenmarathon",
        final_url="https://example.com/badenmarathon",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Baden Marathon",
        normalized_text="Baden Marathon is on 20 September 2026. Registration is open.",
        text_hash="c" * 64,
        links=(PageLink(url="https://example.com/register", text="Registration"),),
    )


def test_prompt_includes_ranked_candidate_links() -> None:
    page_snapshot = PageSnapshot(
        source_url="https://www.badenmarathon.de/",
        final_url="https://www.badenmarathon.de/",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Badenmarathon",
        normalized_text="Baden-Marathon in Karlsruhe.",
        text_hash="d" * 64,
        links=(
            PageLink(url="https://www.badenmarathon.de/", text="Home"),
            PageLink(
                url="https://www.badenmarathon.de/wettbewerbe/marathon",
                text="Marathon",
            ),
            PageLink(
                url="https://www.badenmarathon.de/anmeldung",
                text="Anmeldung",
            ),
        ),
    )

    candidates = candidate_links_for_extraction(page_snapshot)
    prompt = build_snapshot_prompt(page_snapshot)

    assert candidates[0]["url"] == "https://www.badenmarathon.de/anmeldung"
    assert "registration-term" in candidates[0]["reasons"]
    assert "candidate_links" in prompt
    assert "https://www.badenmarathon.de/wettbewerbe/marathon" in prompt


def test_openai_provider_parses_structured_response() -> None:
    client = FakeClient()
    provider = OpenAIExtractorProvider(
        api_key="test-api-key",
        model="gpt-test",
        instructions="Extract event metadata.",
        client=client,
    )

    extraction = asyncio.run(provider.extract(snapshot()))

    assert extraction.name == "Karlsruhe Baden Marathon"
    assert extraction.public_id == "badenmarathon.42"
    assert extraction.country == "Germany"
    assert extraction.timezone == "Europe/Berlin"
    assert extraction.event_date == "2026-09-20"
    assert extraction.distances == ("marathon",)
    assert extraction.regions == ("global", "eu", "de")
    assert extraction.registration_url == "https://example.com/register"
    assert extraction.confidence == 0.91
    assert extraction.provider_name == "openai"

    assert client.responses.request is not None
    assert client.responses.request["model"] == "gpt-test"
    assert client.responses.request["instructions"] == "Extract event metadata."
    assert client.responses.request["text_format"] is OpenAIEventExtraction
    assert "Baden Marathon" in str(client.responses.request["input"])


def test_provider_factory_defaults_to_heuristic() -> None:
    settings = Settings(
        telegram_bot_token="test",
        ai_extractor_provider="heuristic",
        openai_api_key=None,
    )

    assert get_extractor_provider(settings) is None


def test_provider_factory_requires_openai_api_key(tmp_path) -> None:
    settings = Settings(
        telegram_bot_token="test",
        database_url=database_url(tmp_path),
        ai_extractor_provider="openai",
        openai_api_key=None,
    )

    with pytest.raises(ExtractorProviderConfigError):
        get_extractor_provider(settings)


def test_provider_factory_requires_prompt_file_by_default(tmp_path) -> None:
    settings = Settings(
        telegram_bot_token="test",
        database_url=database_url(tmp_path),
        ai_extractor_provider="openai",
        openai_api_key="test-api-key",
        run4221_prompt_dir=str(tmp_path / "prompts"),
    )

    with pytest.raises(ExtractorProviderConfigError, match="Prompt directory does not exist"):
        get_extractor_provider(settings)


def test_provider_factory_requires_active_db_prompt_when_configured(tmp_path) -> None:
    settings = Settings(
        telegram_bot_token="test",
        database_url=database_url(tmp_path),
        ai_extractor_provider="openai",
        openai_api_key="test-api-key",
        run4221_prompt_source="db",
    )

    with pytest.raises(ExtractorProviderConfigError, match="No usable DB prompt"):
        get_extractor_provider(settings)


def test_provider_factory_builds_openai_provider_from_db_prompt(tmp_path) -> None:
    url = database_url(tmp_path)
    upsert_active_prompt_version(
        DISCOVER_EVENT_PROFILE_PROMPT,
        "DB instructions.",
        database_url=url,
    )
    settings = Settings(
        telegram_bot_token="test",
        database_url=url,
        ai_extractor_provider="openai",
        openai_api_key="test-api-key",
        openai_extract_model="gpt-test",
        run4221_prompt_source="db",
    )

    provider = get_extractor_provider(settings)

    assert isinstance(provider, OpenAIExtractorProvider)
    assert provider.model == "gpt-test"
    assert provider.instructions == "DB instructions."


def test_provider_factory_builds_openai_provider_from_file_prompt(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "discover_event_profile.instructions.txt").write_text(
        "File instructions.",
        encoding="utf-8",
    )
    settings = Settings(
        telegram_bot_token="test",
        database_url=database_url(tmp_path),
        ai_extractor_provider="openai",
        openai_api_key="test-api-key",
        openai_extract_model="gpt-test",
        run4221_prompt_source="file",
        run4221_prompt_dir=str(prompts_dir),
    )

    provider = get_extractor_provider(settings)

    assert isinstance(provider, OpenAIExtractorProvider)
    assert provider.model == "gpt-test"
    assert provider.instructions == "File instructions."


def test_openai_error_format_does_not_echo_invalid_key() -> None:
    class FakeOpenAIError(Exception):
        status_code = 401

        def __str__(self) -> str:
            return "Incorrect API key provided: test-api-key"

    assert format_openai_error(FakeOpenAIError()) == (
        "OpenAI authentication failed; check OPENAI_API_KEY."
    )
