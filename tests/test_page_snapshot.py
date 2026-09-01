import asyncio
import gzip
import json
from datetime import UTC, datetime

import httpx
import pytest

from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageLink,
    PageSnapshot,
    fetch_enriched_page_snapshot,
    fetch_page_snapshot,
    merge_page_snapshots,
    store_page_snapshot,
)
from run4221.researcher.artifacts import ResearchArtifactStore

MAX_EXPECTED_SNAPSHOT_LINKS = 100


def run(coro):
    return asyncio.run(coro)


async def public_resolver(_hostname: str) -> tuple[str, ...]:
    return ("93.184.216.34",)


def test_fetch_page_snapshot_extracts_text_title_links_and_hash() -> None:
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        assert request.headers["user-agent"].startswith("run4221-bot/")
        html = """
        <html>
          <head><title>Zurich Marathon 2027 | Registration</title></head>
          <body>
            <h1>Zurich Marathon</h1>
            <p>Race day is April 18, 2027.</p>
            <a href="/register">Register now</a>
            <script>hidden()</script>
          </body>
        </html>
        """
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
            request=request,
        )

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/event",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())

    assert snapshot.source_url == "https://example.com/event"
    assert snapshot.final_url == "https://example.com/event"
    assert snapshot.status_code == 200
    assert snapshot.title == "Zurich Marathon 2027 | Registration"
    assert "Zurich Marathon Race day is April 18, 2027." in snapshot.normalized_text
    assert "hidden()" not in snapshot.normalized_text
    assert len(snapshot.text_hash) == 64
    assert snapshot.links[0].url == "https://example.com/register"
    assert snapshot.links[0].text == "Register now"
    assert len(requested_urls) == 1
    assert requested_urls[0].endswith("/event")


def test_fetch_page_snapshot_caps_links_for_bounded_audit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        links = "".join(
            f'<a href="/event-{index}">Event {index}</a>'
            for index in range(MAX_EXPECTED_SNAPSHOT_LINKS + 1)
        )
        return httpx.Response(200, text=links, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/events",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())

    assert len(snapshot.links) == MAX_EXPECTED_SNAPSHOT_LINKS
    assert snapshot.links[-1].url == "https://example.com/event-99"


def test_fetch_page_snapshot_keeps_late_registration_link_within_cap() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        links = "".join(
            f'<a href="/info-{index}">Info {index}</a>'
            for index in range(MAX_EXPECTED_SNAPSHOT_LINKS)
        )
        links += '<a href="/register">Register now</a>'
        return httpx.Response(200, text=links, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/events",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())

    assert len(snapshot.links) == MAX_EXPECTED_SNAPSHOT_LINKS
    assert snapshot.links[0].url == "https://example.com/register"
    assert all(link.url != "https://example.com/info-99" for link in snapshot.links)


def test_fetch_page_snapshot_deduplicates_before_link_cap() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        links = '<a href="/about">About</a>' * MAX_EXPECTED_SNAPSHOT_LINKS
        links += '<a href="/register">Register now</a>'
        return httpx.Response(200, text=links, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/events",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())

    assert [link.url for link in snapshot.links] == [
        "https://example.com/register",
        "https://example.com/about",
    ]


def test_fetched_capped_snapshot_is_accepted_by_audit_store(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        links = "".join(
            f'<a href="/event-{index}">Event {index}</a>'
            for index in range(MAX_EXPECTED_SNAPSHOT_LINKS + 1)
        )
        return httpx.Response(200, text=links, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/events",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())
    store = ResearchArtifactStore(tmp_path)
    run_id = store.create_run(job_type="refresh")

    reference = store.write_page_snapshot(run_id, snapshot)

    payload = store.read_artifact(reference)
    assert len(payload["content"]["links"]) == MAX_EXPECTED_SNAPSHOT_LINKS


def test_merge_page_snapshots_caps_combined_links_for_bounded_audit() -> None:
    fetched_at = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    root = PageSnapshot(
        source_url="https://example.com/events",
        final_url="https://example.com/events",
        fetched_at=fetched_at,
        status_code=200,
        content_type="text/html",
        title="Events",
        normalized_text="Events",
        text_hash="a" * 64,
        links=tuple(
            PageLink(url=f"https://example.com/root-{index}", text=f"Root {index}")
            for index in range(75)
        ),
    )
    linked = PageSnapshot(
        source_url="https://example.com/register",
        final_url="https://example.com/register",
        fetched_at=fetched_at,
        status_code=200,
        content_type="text/html",
        title="Registration",
        normalized_text="Registration",
        text_hash="b" * 64,
        links=tuple(
            PageLink(url=f"https://example.com/linked-{index}", text=f"Linked {index}")
            for index in range(75)
        ),
    )

    merged = merge_page_snapshots(root, (linked,))

    assert len(merged.links) == MAX_EXPECTED_SNAPSHOT_LINKS
    assert merged.links[-1].url == "https://example.com/linked-24"


def test_merge_page_snapshots_caps_oversized_root_without_linked_pages() -> None:
    fetched_at = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    root = PageSnapshot(
        source_url="https://example.com/events",
        final_url="https://example.com/events",
        fetched_at=fetched_at,
        status_code=200,
        content_type="text/html",
        title="Events",
        normalized_text="Events",
        text_hash="a" * 64,
        links=tuple(
            PageLink(url=f"https://example.com/root-{index}", text=f"Root {index}")
            for index in range(MAX_EXPECTED_SNAPSHOT_LINKS + 1)
        ),
    )

    merged = merge_page_snapshots(root, ())

    assert len(merged.links) == MAX_EXPECTED_SNAPSHOT_LINKS
    assert merged.links[-1].url == "https://example.com/root-99"


def test_store_page_snapshot_writes_json(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<title>Test</title><p>Hello</p>", request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())
    path = store_page_snapshot(snapshot, tmp_path)

    assert path.parent == tmp_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["source_url"] == "https://example.com"
    assert payload["title"] == "Test"
    assert payload["text_hash"] == snapshot.text_hash


def test_fetch_enriched_page_snapshot_fetches_same_domain_registration_page() -> None:
    seen_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(f"https://{request.headers['host']}{request.url.path}")
        assert request.url.host == "93.184.216.34"
        if request.url.path == "/":
            html = """
            <html>
              <head><title>Badenmarathon</title></head>
              <body>
                <h1>Badenmarathon</h1>
                <a href="/anmeldung">Anmeldung</a>
                <a href="https://other.example/register">External registration</a>
              </body>
            </html>
            """
            return httpx.Response(200, text=html, request=request)

        if request.url.path == "/anmeldung":
            html = """
            <html>
              <head><title>Anmeldung Badenmarathon</title></head>
              <body>
                <p>Registration opens on 2026-05-31.</p>
                <a href="/payment">Register now</a>
              </body>
            </html>
            """
            return httpx.Response(200, text=html, request=request)

        return httpx.Response(404, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_enriched_page_snapshot(
                "https://example.com/",
                client=client,
                resolve_host=public_resolver,
            )

    snapshot = run(fetch())

    assert seen_urls == ["https://example.com/", "https://example.com/anmeldung"]
    assert "Linked page: https://example.com/anmeldung." in snapshot.normalized_text
    assert "Registration opens on 2026-05-31." in snapshot.normalized_text
    assert any(link.url == "https://example.com/payment" for link in snapshot.links)


def test_fetch_page_snapshot_rejects_private_literal_address() -> None:
    async def fetch():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot("http://127.0.0.1/admin", client=client)

    with pytest.raises(PageFetchError, match="Private or non-public address"):
        run(fetch())


def test_fetch_page_snapshot_rejects_invalid_resolver_address() -> None:
    async def invalid_resolver(_hostname: str) -> tuple[str, ...]:
        return ("not-an-ip-address",)

    async def fetch():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/",
                client=client,
                resolve_host=invalid_resolver,
            )

    with pytest.raises(PageFetchError, match="invalid address"):
        run(fetch())


@pytest.mark.parametrize(
    "addresses",
    [
        ("127.0.0.1",),
        ("169.254.169.254",),
        ("93.184.216.34", "127.0.0.1"),
    ],
)
def test_fetch_page_snapshot_rejects_private_resolver_addresses(addresses) -> None:
    request_sent = False

    async def resolver(_hostname: str) -> tuple[str, ...]:
        return addresses

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_sent
        request_sent = True
        return httpx.Response(200, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/",
                client=client,
                resolve_host=resolver,
            )

    with pytest.raises(PageFetchError, match="Private or non-public address"):
        run(fetch())

    assert request_sent is False


def test_fetch_page_snapshot_times_out_stalled_resolver() -> None:
    async def stalled_resolver(_hostname: str) -> tuple[str, ...]:
        await asyncio.Event().wait()
        return ()

    async def fetch():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/",
                client=client,
                resolve_host=stalled_resolver,
                timeout=0.01,
            )

    with pytest.raises(PageFetchError, match="Timed out resolving host"):
        run(fetch())


def test_fetch_page_snapshot_revalidates_redirect_target() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data"},
            request=request,
        )

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/start",
                client=client,
                resolve_host=public_resolver,
            )

    with pytest.raises(PageFetchError, match="Private or non-public address"):
        run(fetch())


def test_fetch_page_snapshot_blocks_cross_origin_redirect_before_request() -> None:
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"location": "https://other.example/registration"},
            request=request,
        )

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/start",
                client=client,
                resolve_host=public_resolver,
                allowed_origin="https://example.com/event",
            )

    with pytest.raises(PageFetchError, match="leaves the approved origin"):
        run(fetch())

    assert len(requested_urls) == 1


@pytest.mark.parametrize(
    ("candidate_url", "error"),
    (
        ("http://example.com/registration", "leaves the approved origin"),
        ("https://example.com:8443/registration", "Non-default URL ports"),
    ),
)
def test_fetch_page_snapshot_rejects_origin_change_before_request(
    candidate_url: str,
    error: str,
) -> None:
    requested_urls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        return httpx.Response(200, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                candidate_url,
                client=client,
                resolve_host=public_resolver,
                allowed_origin="https://example.com/event",
            )

    with pytest.raises(PageFetchError, match=error):
        run(fetch())

    assert requested_urls == []


def test_fetch_page_snapshot_enforces_response_size_limit() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 11, request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/large",
                client=client,
                resolve_host=public_resolver,
                max_response_bytes=10,
            )

    with pytest.raises(PageFetchError, match="response limit"):
        run(fetch())


def test_fetch_page_snapshot_rejects_compressed_body() -> None:
    html = b"<html><title>Compressed</title><p>" + b"x" * 10_000 + b"</p></html>"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        compressed = gzip.compress(html)
        assert len(compressed) < 200
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html; charset=utf-8",
                "content-encoding": "gzip",
                "content-length": str(len(compressed)),
            },
            content=compressed,
            request=request,
        )

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot(
                "https://example.com/compressed",
                client=client,
                resolve_host=public_resolver,
                max_response_bytes=200,
            )

    with pytest.raises(PageFetchError, match="Compressed page responses"):
        run(fetch())
