import asyncio
from datetime import UTC, datetime

from run4221.ai.registration_window import (
    RegistrationWindowExtraction,
    update_registration_window,
)
from run4221.db.repository import (
    EventCreate,
    add_event,
    find_event,
    list_open_events,
    list_proposed_event_updates,
)
from run4221.ingestion.page_snapshot import PageLink, PageSnapshot


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-registration-test.sqlite3'}"


def tracked_event(database_url: str):
    return add_event(
        EventCreate(
            public_id="badenmarathon.42",
            name="Baden Marathon",
            city="Karlsruhe",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2026-09-20",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url="https://www.badenmarathon.de/",
            registration_url="https://www.badenmarathon.de/wettbewerbe/marathon",
        ),
        database_url=database_url,
    )


def barcelona_event(database_url: str):
    return add_event(
        EventCreate(
            public_id="barcelona.42",
            name="Zurich Marató Barcelona",
            city="Barcelona",
            country="Spain",
            timezone="Europe/Madrid",
            event_date="2027-03-14",
            distances=("marathon",),
            regions=("global", "eu", "es"),
            official_url="https://zurichmaratobarcelona.es/en/",
            registration_url=(
                "https://zurichmaratobarcelona.es/en/2026/03/20/"
                "fotyen-tesfay-wins-zurich-barcelona-marathon/"
            ),
        ),
        database_url=database_url,
    )


def snapshot(text: str) -> PageSnapshot:
    return PageSnapshot(
        source_url="https://www.badenmarathon.de/wettbewerbe/marathon",
        final_url="https://www.badenmarathon.de/wettbewerbe/marathon",
        fetched_at=datetime.now(UTC),
        status_code=200,
        content_type="text/html",
        title="Badenmarathon",
        normalized_text=text,
        text_hash="abc123",
        links=(
            PageLink(
                url="https://www.badenmarathon.de/wettbewerbe/marathon",
                text="Marathon",
            ),
        ),
    )


def test_update_registration_window_creates_proposed_update(tmp_path) -> None:
    url = database_url(tmp_path)
    event = tracked_event(url)

    async def fake_fetch(_url: str) -> PageSnapshot:
        return snapshot(
            "Sichere dir jetzt einen Startplatz. "
            "Der 42. Baden-Marathon findet am 20. September 2026 statt."
        )

    result = asyncio.run(
        update_registration_window(
            event,
            fetch_snapshot=fake_fetch,
            store_snapshot=None,
            database_url=url,
        )
    )

    updates = list_proposed_event_updates(event_id=event.id, database_url=url)

    assert result.registration_status == "open"
    assert result.needs_moderator_review
    assert result.proposed_update_id == updates[0].id
    assert updates[0].proposed_fields["registration_status"] == "open"
    assert updates[0].proposed_fields["registration_url"] == event.registration_url


def test_update_registration_window_skips_provider_for_blocked_page(tmp_path) -> None:
    url = database_url(tmp_path)
    event = tracked_event(url)

    blocked_snapshot = PageSnapshot(
        source_url="https://www.badenmarathon.de/wettbewerbe/marathon",
        final_url="https://www.badenmarathon.de/wettbewerbe/marathon",
        fetched_at=datetime.now(UTC),
        status_code=403,
        content_type="text/html",
        title="Just a moment...",
        normalized_text="Checking your browser before accessing the site.",
        text_hash="blocked",
        links=(),
    )

    async def fake_fetch(_url: str) -> PageSnapshot:
        return blocked_snapshot

    class FailingProvider:
        provider_name = "fake-ai"

        async def extract(self, _snapshot: PageSnapshot, _event) -> RegistrationWindowExtraction:
            raise AssertionError("blocked snapshots should not be sent to the provider")

    result = asyncio.run(
        update_registration_window(
            event,
            fetch_snapshot=fake_fetch,
            store_snapshot=None,
            provider=FailingProvider(),
            database_url=url,
        )
    )

    assert result.registration_status == "unknown"
    assert result.registration_url == event.registration_url
    assert result.confidence == 0.05
    assert result.proposed_update_id is None
    assert "Page blocked: site protection challenge page (HTTP 403)." in result.evidence
    assert "Registration extractor provider: fallback." in result.evidence


def test_update_registration_window_can_auto_apply_low_risk_update(tmp_path) -> None:
    url = database_url(tmp_path)
    event = tracked_event(url)

    class FakeProvider:
        provider_name = "fake"

        async def extract(
            self,
            _snapshot: PageSnapshot,
            _event,
        ) -> RegistrationWindowExtraction:
            return RegistrationWindowExtraction(
                registration_status="open",
                registration_open_at=None,
                registration_open_precision="unknown",
                registration_close_at=None,
                registration_url=event.registration_url,
                event_date=event.event_date,
                confidence=0.95,
                evidence_snippets=("Registration is open.",),
                provider_name=self.provider_name,
            )

    async def fake_fetch(_url: str) -> PageSnapshot:
        return snapshot("Registration is open.")

    result = asyncio.run(
        update_registration_window(
            event,
            fetch_snapshot=fake_fetch,
            store_snapshot=None,
            provider=FakeProvider(),
            database_url=url,
            auto_confirm=True,
        )
    )

    updated_event = find_event(event.id, url)

    assert result.applied
    assert result.proposed_update_id is None
    assert updated_event is not None
    assert updated_event.registration_status == "open"
    assert [open_event.public_id for open_event in list_open_events(database_url=url)] == [
        event.public_id
    ]


def test_update_registration_window_ignores_stale_article_registration_url(tmp_path) -> None:
    url = database_url(tmp_path)
    event = barcelona_event(url)
    fetched_urls = []

    async def fake_fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return PageSnapshot(
            source_url=source_url,
            final_url=source_url,
            fetched_at=datetime.now(UTC),
            status_code=200,
            content_type="text/html",
            title="Old Barcelona Marathon Article",
            normalized_text="Sold out. Race date April 30, 2026.",
            text_hash="old",
            links=(),
        )

    result = asyncio.run(
        update_registration_window(
            event,
            fetch_snapshot=fake_fetch,
            store_snapshot=None,
            database_url=url,
        )
    )

    assert fetched_urls == ["https://zurichmaratobarcelona.es/en/"]
    assert result.registration_status == "unknown"
    assert result.event_date is None
    assert result.registration_url is None
    assert result.proposed_update_id is None
    assert "Ignored stored registration URL" in result.evidence
    assert "Ignored stale page date 2026-04-30" in result.evidence
    assert list_proposed_event_updates(event_id=event.id, database_url=url) == ()


def test_update_registration_window_ignores_mini_marathon_source_for_marathon(
    tmp_path,
) -> None:
    url = database_url(tmp_path)
    official_url = "https://www.bmw-berlin-marathon.com/en/"
    mini_url = (
        "https://www.bmw-berlin-marathon.com/anmelden/kids-and-youth/mini-marathon"
    )
    event = add_event(
        EventCreate(
            public_id="berlin.42",
            name="BMW BERLIN-MARATHON",
            city="Berlin",
            country="Germany",
            timezone="Europe/Berlin",
            event_date="2026-09-27",
            distances=("marathon",),
            regions=("global", "eu", "de"),
            official_url=official_url,
            registration_url=mini_url,
        ),
        database_url=url,
    )
    fetched_urls = []

    async def fake_fetch(source_url: str) -> PageSnapshot:
        fetched_urls.append(source_url)
        return PageSnapshot(
            source_url=source_url,
            final_url=source_url,
            fetched_at=datetime(2026, 8, 31, tzinfo=UTC),
            status_code=200,
            content_type="text/html",
            title="BMW BERLIN-MARATHON",
            normalized_text="Lottery results are available.",
            text_hash="berlin",
            links=(),
        )

    result = asyncio.run(
        update_registration_window(
            event,
            fetch_snapshot=fake_fetch,
            store_snapshot=None,
            database_url=url,
        )
    )

    assert fetched_urls == [official_url]
    assert result.registration_status == "unknown"
    assert result.registration_url is None
    assert result.proposed_update_id is None
    assert "Ignored stored registration URL" in result.evidence
    assert list_proposed_event_updates(event_id=event.id, database_url=url) == ()
