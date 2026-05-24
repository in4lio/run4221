from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

MAX_SNAPSHOT_TEXT_CHARS = 50_000
DEFAULT_TIMEOUT_SECONDS = 10.0
USER_AGENT = "run4221-bot/0.1 (+https://run4221.com)"
LINK_DISCOVERY_TERMS = (
    "register",
    "registration",
    "entry",
    "entries",
    "sign up",
    "signup",
    "anmeldung",
    "anmelden",
    "teilnahme",
)
BLOCKED_STATUS_CODES = {401, 403, 429}
BOT_PROTECTION_TERMS = (
    "just a moment",
    "checking your browser",
    "verify you are human",
    "enable javascript and cookies",
    "cloudflare",
    "cf-browser-verification",
    "ddos-guard",
    "access denied",
)


@dataclass(frozen=True)
class PageLink:
    url: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return {"url": self.url, "text": self.text}


@dataclass(frozen=True)
class PageSnapshot:
    source_url: str
    final_url: str
    fetched_at: datetime
    status_code: int
    content_type: str | None
    title: str | None
    normalized_text: str
    text_hash: str
    links: tuple[PageLink, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "source_url": self.source_url,
            "final_url": self.final_url,
            "fetched_at": self.fetched_at.isoformat(),
            "status_code": self.status_code,
            "content_type": self.content_type,
            "title": self.title,
            "normalized_text": self.normalized_text,
            "text_hash": self.text_hash,
            "links": [link.to_dict() for link in self.links],
        }


class PageFetchError(RuntimeError):
    pass


def blocked_page_reason(snapshot: PageSnapshot) -> str | None:
    searchable = f"{snapshot.title or ''} {snapshot.normalized_text[:2_000]}".casefold()
    has_protection_text = any(term in searchable for term in BOT_PROTECTION_TERMS)
    if snapshot.status_code in BLOCKED_STATUS_CODES and has_protection_text:
        return f"site protection challenge page (HTTP {snapshot.status_code})"
    if snapshot.status_code in BLOCKED_STATUS_CODES:
        return f"blocked HTTP status {snapshot.status_code}"
    if has_protection_text:
        return "site protection challenge page"

    return None


async def fetch_page_snapshot(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> PageSnapshot:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
    }
    try:
        if client is not None:
            response = await client.get(
                url,
                follow_redirects=True,
                headers=headers,
                timeout=timeout,
            )
        else:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers=headers,
                timeout=timeout,
            ) as default_client:
                response = await default_client.get(url)
    except httpx.HTTPError as error:
        raise PageFetchError(f"Could not fetch {url}: {error}") from error

    title, normalized_text, links = extract_response_content(response.text, str(response.url))
    stored_text = normalized_text[:MAX_SNAPSHOT_TEXT_CHARS]
    return PageSnapshot(
        source_url=url,
        final_url=str(response.url),
        fetched_at=datetime.now(UTC),
        status_code=response.status_code,
        content_type=response.headers.get("content-type"),
        title=title,
        normalized_text=stored_text,
        text_hash=hash_text(stored_text),
        links=links,
    )


async def fetch_enriched_page_snapshot(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_linked_pages: int = 3,
) -> PageSnapshot:
    if client is None:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        }
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers=headers,
            timeout=timeout,
        ) as default_client:
            return await fetch_enriched_page_snapshot(
                url,
                client=default_client,
                timeout=timeout,
                max_linked_pages=max_linked_pages,
            )

    root_snapshot = await fetch_page_snapshot(url, client=client, timeout=timeout)
    linked_snapshots = []
    for candidate_url in same_domain_candidate_urls(root_snapshot, max_linked_pages):
        try:
            linked_snapshots.append(
                await fetch_page_snapshot(candidate_url, client=client, timeout=timeout)
            )
        except PageFetchError:
            continue

    return merge_page_snapshots(root_snapshot, tuple(linked_snapshots))


def extract_response_content(
    content: str,
    base_url: str,
) -> tuple[str | None, str, tuple[PageLink, ...]]:
    parser = SnapshotHTMLParser(base_url)
    parser.feed(content)
    parser.close()

    title = normalize_text(" ".join(parser.title_parts)) or None
    body_text = normalize_text(" ".join(parser.text_parts))
    links = tuple(
        PageLink(url=url, text=text)
        for url, text in parser.links
        if url
    )
    if body_text:
        return title, body_text, links

    return title, normalize_text(content), links


def same_domain_candidate_urls(snapshot: PageSnapshot, limit: int) -> tuple[str, ...]:
    root_domain = urlparse(snapshot.final_url).netloc.casefold()
    root_url = normalize_fetch_url(snapshot.final_url)
    candidates: list[str] = []
    for link in snapshot.links:
        parsed = urlparse(link.url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if parsed.netloc.casefold() != root_domain:
            continue
        normalized_url = normalize_fetch_url(link.url)
        if normalized_url == root_url or normalized_url in candidates:
            continue

        searchable = f"{link.text} {link.url}".casefold()
        if any(term in searchable for term in LINK_DISCOVERY_TERMS):
            candidates.append(normalized_url)
        if len(candidates) >= limit:
            break

    return tuple(candidates)


def merge_page_snapshots(
    root_snapshot: PageSnapshot,
    linked_snapshots: tuple[PageSnapshot, ...],
) -> PageSnapshot:
    if not linked_snapshots:
        return root_snapshot

    text_parts = [root_snapshot.normalized_text]
    links = [*root_snapshot.links]
    seen_links = {(link.url, link.text) for link in links}
    for linked_snapshot in linked_snapshots:
        text_parts.append(
            " ".join(
                part
                for part in (
                    f"Linked page: {linked_snapshot.final_url}.",
                    f"Title: {linked_snapshot.title}." if linked_snapshot.title else "",
                    linked_snapshot.normalized_text,
                )
                if part
            )
        )
        for link in linked_snapshot.links:
            key = (link.url, link.text)
            if key not in seen_links:
                links.append(link)
                seen_links.add(key)

    normalized_text = normalize_text(" ".join(text_parts))[:MAX_SNAPSHOT_TEXT_CHARS]
    return PageSnapshot(
        source_url=root_snapshot.source_url,
        final_url=root_snapshot.final_url,
        fetched_at=root_snapshot.fetched_at,
        status_code=root_snapshot.status_code,
        content_type=root_snapshot.content_type,
        title=root_snapshot.title,
        normalized_text=normalized_text,
        text_hash=hash_text(normalized_text),
        links=tuple(links),
    )


def normalize_fetch_url(value: str) -> str:
    parsed = urlparse(value)
    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.casefold(),
            parsed.netloc.casefold(),
            path,
            "",
            parsed.query,
            "",
        )
    )


class SnapshotHTMLParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._title_depth = 0
        self._link_stack: list[tuple[str, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if normalized_tag == "title":
            self._title_depth += 1
            return
        if normalized_tag == "a":
            href = dict(attrs).get("href")
            if href:
                self._link_stack.append((urljoin(self.base_url, href), []))

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if normalized_tag == "title" and self._title_depth:
            self._title_depth -= 1
            return
        if normalized_tag == "a" and self._link_stack:
            url, text_parts = self._link_stack.pop()
            text = normalize_text(" ".join(text_parts))
            self.links.append((url, text))

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return

        if self._title_depth:
            self.title_parts.append(data)

        self.text_parts.append(data)
        if self._link_stack:
            self._link_stack[-1][1].append(data)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(value)).strip()


def hash_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def default_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "page_snapshots"


def store_page_snapshot(snapshot: PageSnapshot, root: str | Path | None = None) -> Path:
    directory = Path(root) if root is not None else default_snapshot_dir()
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{snapshot.fetched_at.strftime('%Y%m%dT%H%M%SZ')}-{snapshot.text_hash[:12]}.json"
    path = directory / filename
    path.write_text(json.dumps(snapshot.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path
