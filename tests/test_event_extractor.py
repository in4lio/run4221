import asyncio
from datetime import UTC, datetime
from pathlib import Path

from run4221.ai.event_extractor import (
    extract_event_draft_from_url,
    select_registration_url_for_distances,
)
from run4221.ai.extraction_provider import EventExtraction, ExtractorProviderError
from run4221.ingestion.page_snapshot import PageFetchError, PageLink, PageSnapshot


def extract_url_only(url: str):
    return asyncio.run(extract_event_draft_from_url(url, fetch_snapshot=None))


def test_stub_extractor_derives_marathon_draft_from_url() -> None:
    draft = extract_url_only("https://www.bmw-berlin-marathon.com/en/")

    assert draft.name == "Berlin Marathon"
    assert draft.public_id == "berlin.42"
    assert draft.city == "Berlin"
    assert draft.timezone == "Etc/UTC"
    assert draft.distances == ("marathon",)
    assert draft.regions == ("global",)
    assert draft.official_url == "https://www.bmw-berlin-marathon.com/en/"
    assert draft.registration_url is None
    assert draft.confidence == 0.1


def test_stub_extractor_derives_half_marathon_draft_from_url() -> None:
    draft = extract_url_only("https://berlin-half-marathon.de/register")

    assert draft.name == "Berlin Half Marathon"
    assert draft.public_id == "berlin.21"
    assert draft.distances == ("half_marathon",)


def test_extractor_uses_page_snapshot_for_draft_fields() -> None:
    snapshot = PageSnapshot(
        source_url="https://example.com/zurich",
        final_url="https://example.com/zurich",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Registration | Zurich Marathon 2027",
        normalized_text="Zurich Marathon takes place on April 18, 2027. Registration opens soon.",
        text_hash="a" * 64,
        links=(PageLink(url="https://example.com/register", text="Register now"),),
    )

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://example.com/zurich"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://example.com/zurich",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=lambda _: Path("snapshot.json"),
        )
    )

    assert draft.name == "Zurich Marathon"
    assert draft.public_id == "zurich.42"
    assert draft.city == "Zurich"
    assert draft.event_date == "2027-04-18"
    assert draft.registration_url == "https://example.com/register"
    assert draft.confidence > 0.1
    assert "Text hash: aaaaaaaaaaaa" in draft.evidence
    assert "Extractor provider: heuristic" in draft.evidence
    assert "AI provider is not configured yet" in draft.evidence


def test_extractor_accepts_injected_structured_provider() -> None:
    snapshot = PageSnapshot(
        source_url="https://example.com/race",
        final_url="https://example.com/race",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Messy race page",
        normalized_text="Everything is complicated here.",
        text_hash="b" * 64,
        links=(),
    )

    class FakeExtractorProvider:
        provider_name = "fake-ai"

        async def extract(self, received_snapshot: PageSnapshot) -> EventExtraction:
            assert received_snapshot is snapshot
            return EventExtraction(
                name="Karlsruhe Baden Marathon",
                public_id="badenmarathon.42",
                city="Karlsruhe",
                country="Germany",
                timezone="Europe/Berlin",
                event_date="2026-09-20",
                distances=("marathon",),
                regions=("global", "eu", "de"),
                official_url="https://example.com/race",
                registration_url="https://example.com/register",
                confidence=0.86,
                evidence_snippets=("AI evidence: page says 20 September 2026.",),
                provider_name=self.provider_name,
            )

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://example.com/race"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://example.com/race",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=None,
            extractor_provider=FakeExtractorProvider(),
        )
    )

    assert draft.name == "Karlsruhe Baden Marathon"
    assert draft.public_id == "badenmarathon.42"
    assert draft.country == "Germany"
    assert draft.timezone == "Europe/Berlin"
    assert draft.event_date == "2026-09-20"
    assert draft.registration_url == "https://example.com/register"
    assert draft.confidence == 0.86
    assert "AI evidence: page says 20 September 2026." in draft.evidence
    assert "Extractor provider: fake-ai" in draft.evidence
    assert "AI provider is not configured yet" not in draft.evidence


def test_extractor_skips_provider_for_blocked_challenge_page() -> None:
    snapshot = PageSnapshot(
        source_url="https://maraton.istanbul",
        final_url="https://maraton.istanbul",
        fetched_at=datetime(2026, 5, 24, tzinfo=UTC),
        status_code=403,
        content_type="text/html",
        title="Just a moment...",
        normalized_text="Checking your browser before accessing maraton.istanbul.",
        text_hash="c" * 64,
        links=(),
    )

    class FailingProvider:
        provider_name = "fake-ai"

        async def extract(self, received_snapshot: PageSnapshot) -> EventExtraction:
            raise AssertionError("blocked snapshots should not be sent to AI")

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://maraton.istanbul"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://maraton.istanbul",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=lambda _: Path("snapshot.json"),
            extractor_provider=FailingProvider(),
        )
    )

    assert draft.confidence == 0.03
    assert draft.official_url == "https://maraton.istanbul"
    assert "Page blocked: site protection challenge page (HTTP 403)." in draft.evidence
    assert "Extractor provider: url-fallback." in draft.evidence


def test_extractor_promotes_distance_link_when_ai_omits_registration_url() -> None:
    snapshot = PageSnapshot(
        source_url="https://www.badenmarathon.de/",
        final_url="https://www.badenmarathon.de/",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Badenmarathon - Marathon | Halbmarathon",
        normalized_text="Der 42. Baden-Marathon findet am 20. September 2026 statt.",
        text_hash="c" * 64,
        links=(
            PageLink(url="https://www.badenmarathon.de/", text="Home"),
            PageLink(
                url="https://www.badenmarathon.de/wettbewerbe/marathon",
                text="Marathon",
            ),
            PageLink(
                url="https://www.badenmarathon.de/wettbewerbe/halbmarathon",
                text="Halbmarathon",
            ),
        ),
    )

    class RegistrationUrlOmittingProvider:
        provider_name = "fake-ai"

        async def extract(self, received_snapshot: PageSnapshot) -> EventExtraction:
            assert received_snapshot is snapshot
            return EventExtraction(
                name="Baden-Marathon",
                public_id="badenmarathon.42",
                city="Karlsruhe",
                country="Germany",
                timezone="Etc/UTC",
                event_date="2026-09-20",
                distances=("marathon",),
                regions=("global", "eu", "de"),
                official_url="https://www.badenmarathon.de/",
                registration_url=None,
                confidence=0.97,
                evidence_snippets=(
                    "AI cited event link: https://www.badenmarathon.de/wettbewerbe/marathon",
                ),
                provider_name=self.provider_name,
            )

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://www.badenmarathon.de/"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://www.badenmarathon.de/",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=None,
            extractor_provider=RegistrationUrlOmittingProvider(),
        )
    )

    assert draft.registration_url == "https://www.badenmarathon.de/wettbewerbe/marathon"
    assert (
        "https://www.badenmarathon.de/wettbewerbe/marathon",
        "AI cited event link: https://www.badenmarathon.de/wettbewerbe/marathon",
    ) in draft.registration_url_candidates
    assert (
        "https://www.badenmarathon.de/wettbewerbe/halbmarathon",
        "Halbmarathon",
    ) in draft.registration_url_candidates
    assert draft.timezone == "Europe/Berlin"
    assert "Selected registration URL from page links" in draft.evidence
    assert "Selected timezone from location: Europe/Berlin." in draft.evidence


def test_extractor_rejects_stale_article_registration_url() -> None:
    snapshot = PageSnapshot(
        source_url="https://zurichmaratobarcelona.es/en/",
        final_url="https://zurichmaratobarcelona.es/en/",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Zurich Marató Barcelona",
        normalized_text="Zurich Marató Barcelona takes place on March 14, 2027.",
        text_hash="d" * 64,
        links=(),
    )

    class ArticleUrlProvider:
        provider_name = "fake-ai"

        async def extract(self, received_snapshot: PageSnapshot) -> EventExtraction:
            assert received_snapshot is snapshot
            return EventExtraction(
                name="Zurich Marató Barcelona",
                public_id="barcelona.42",
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
                confidence=0.93,
                evidence_snippets=("AI cited an old news article URL.",),
                provider_name=self.provider_name,
            )

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://zurichmaratobarcelona.es/en/"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://zurichmaratobarcelona.es/en/",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=None,
            extractor_provider=ArticleUrlProvider(),
        )
    )

    assert draft.registration_url is None
    assert draft.event_date == "2027-03-14"


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


def test_extractor_falls_back_when_structured_provider_fails() -> None:
    snapshot = PageSnapshot(
        source_url="https://example.com/zurich",
        final_url="https://example.com/zurich",
        fetched_at=datetime(2026, 5, 18, tzinfo=UTC),
        status_code=200,
        content_type="text/html",
        title="Zurich Marathon 2027",
        normalized_text="Zurich Marathon takes place on April 18, 2027.",
        text_hash="d" * 64,
        links=(),
    )

    class BrokenExtractorProvider:
        provider_name = "broken-ai"

        async def extract(self, received_snapshot: PageSnapshot) -> EventExtraction:
            assert received_snapshot is snapshot
            raise ExtractorProviderError("temporary outage")

    async def fetch_snapshot(url: str) -> PageSnapshot:
        assert url == "https://example.com/zurich"
        return snapshot

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://example.com/zurich",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=None,
            extractor_provider=BrokenExtractorProvider(),
        )
    )

    assert draft.name == "Zurich Marathon"
    assert draft.public_id == "zurich.42"
    assert "Extractor provider broken-ai failed" in draft.evidence
    assert "Extractor provider: heuristic" in draft.evidence


def test_extractor_falls_back_to_url_text_when_fetch_fails() -> None:
    async def fetch_snapshot(url: str) -> PageSnapshot:
        raise PageFetchError(f"offline: {url}")

    draft = asyncio.run(
        extract_event_draft_from_url(
            "https://www.bmw-berlin-marathon.com/en/",
            fetch_snapshot=fetch_snapshot,
            store_snapshot=None,
        )
    )

    assert draft.name == "Berlin Marathon"
    assert draft.public_id == "berlin.42"
    assert draft.confidence == 0.05
    assert "Page fetch failed" in draft.evidence
