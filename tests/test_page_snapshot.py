import asyncio
import json

import httpx

from run4221.ingestion.page_snapshot import (
    fetch_enriched_page_snapshot,
    fetch_page_snapshot,
    store_page_snapshot,
)


def run(coro):
    return asyncio.run(coro)


def test_fetch_page_snapshot_extracts_text_title_links_and_hash() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
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
            return await fetch_page_snapshot("https://example.com/event", client=client)

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


def test_store_page_snapshot_writes_json(tmp_path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<title>Test</title><p>Hello</p>", request=request)

    async def fetch():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await fetch_page_snapshot("https://example.com", client=client)

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
        seen_urls.append(str(request.url))
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
            return await fetch_enriched_page_snapshot("https://example.com/", client=client)

    snapshot = run(fetch())

    assert seen_urls == ["https://example.com/", "https://example.com/anmeldung"]
    assert "Linked page: https://example.com/anmeldung." in snapshot.normalized_text
    assert "Registration opens on 2026-05-31." in snapshot.normalized_text
    assert any(link.url == "https://example.com/payment" for link in snapshot.links)
