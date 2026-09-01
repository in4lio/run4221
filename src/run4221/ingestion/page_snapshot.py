from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx

MAX_SNAPSHOT_TEXT_CHARS = 50_000
MAX_SNAPSHOT_LINKS = 100
MAX_RESPONSE_BYTES = 2_000_000
MAX_REDIRECTS = 5
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_DNS_TIMEOUT_SECONDS = 5.0
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


type HostResolver = Callable[[str], Awaitable[tuple[str, ...]]]


async def resolve_host_addresses(hostname: str) -> tuple[str, ...]:
    try:
        records = await asyncio.to_thread(
            socket.getaddrinfo,
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except OSError as error:
        raise PageFetchError(f"Could not resolve host {hostname}: {error}") from error
    return tuple(dict.fromkeys(record[4][0] for record in records))


async def validate_public_http_url(
    url: str,
    *,
    resolve_host: HostResolver = resolve_host_addresses,
    resolver_timeout: float = DEFAULT_DNS_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    parsed = urlparse(url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        raise PageFetchError(f"Only absolute public HTTP(S) URLs are allowed: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise PageFetchError("URLs containing credentials are not allowed.")

    hostname = parsed.hostname.rstrip(".").casefold()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise PageFetchError(f"Private or local host is not allowed: {hostname}")

    try:
        literal_address = ipaddress.ip_address(hostname)
    except ValueError:
        try:
            addresses = await asyncio.wait_for(
                resolve_host(hostname),
                timeout=resolver_timeout,
            )
        except TimeoutError as error:
            raise PageFetchError(f"Timed out resolving host {hostname}.") from error
    else:
        addresses = (str(literal_address),)

    if not addresses:
        raise PageFetchError(f"Host did not resolve to an address: {hostname}")
    parsed_addresses = []
    for address in addresses:
        try:
            parsed_addresses.append(ipaddress.ip_address(address))
        except ValueError as error:
            raise PageFetchError(f"Host resolved to an invalid address: {address}") from error
    blocked = tuple(str(address) for address in parsed_addresses if not address.is_global)
    if blocked:
        raise PageFetchError(f"Private or non-public address is not allowed: {blocked[0]}")
    return tuple(
        str(address) for address in sorted(parsed_addresses, key=lambda address: address.version)
    )


def validate_allowed_origin(url: str, allowed_origin: str) -> None:
    if _normalized_public_origin(url) != _normalized_public_origin(allowed_origin):
        raise PageFetchError(f"URL leaves the approved origin: {url}")


def _normalized_public_origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(url)
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise PageFetchError(f"Only absolute public HTTP(S) URLs are allowed: {url}")
    try:
        port = parsed.port
    except ValueError as error:
        raise PageFetchError(f"Invalid URL port: {url}") from error
    default_port = 443 if scheme == "https" else 80
    if port not in {None, default_port}:
        raise PageFetchError(f"Non-default URL ports are not allowed: {url}")
    return scheme, parsed.hostname.rstrip(".").casefold(), default_port


async def fetch_public_response(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    resolve_host: HostResolver,
    max_response_bytes: int,
    allowed_origin: str | None = None,
) -> httpx.Response:
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        if allowed_origin is not None:
            validate_allowed_origin(current_url, allowed_origin)
        addresses = await validate_public_http_url(
            current_url,
            resolve_host=resolve_host,
            resolver_timeout=min(timeout, DEFAULT_DNS_TIMEOUT_SECONDS),
        )
        original_url = httpx.URL(current_url)
        request_headers = {**headers, "Host": original_url.netloc.decode("ascii")}
        redirect_url = None
        last_connect_error = None
        for address in addresses:
            pinned_url = original_url.copy_with(host=address)
            try:
                async with client.stream(
                    "GET",
                    pinned_url,
                    follow_redirects=False,
                    headers=request_headers,
                    timeout=timeout,
                    extensions={"sni_hostname": original_url.host},
                ) as streamed:
                    if streamed.is_redirect:
                        location = streamed.headers.get("location")
                        if not location:
                            raise PageFetchError(
                                f"Redirect response has no Location header: {current_url}"
                            )
                        if redirect_count == MAX_REDIRECTS:
                            raise PageFetchError(f"Too many redirects while fetching {url}")
                        redirect_url = urljoin(current_url, location)
                        break

                    content_type = streamed.headers.get("content-type")
                    if content_type:
                        media_type = content_type.split(";", maxsplit=1)[0].strip().casefold()
                        if not (
                            media_type.startswith("text/") or media_type == "application/xhtml+xml"
                        ):
                            raise PageFetchError(f"Unsupported page content type: {media_type}")

                    content_encoding = streamed.headers.get("content-encoding", "identity")
                    if content_encoding.strip().casefold() != "identity":
                        raise PageFetchError("Compressed page responses are not supported.")

                    declared_length = streamed.headers.get("content-length")
                    if declared_length and declared_length.isdigit():
                        if int(declared_length) > max_response_bytes:
                            raise PageFetchError(
                                f"Page exceeds the {max_response_bytes}-byte response limit."
                            )

                    content = (
                        bytearray(streamed.content) if streamed.is_stream_consumed else bytearray()
                    )
                    if len(content) > max_response_bytes:
                        raise PageFetchError(
                            f"Page exceeds the {max_response_bytes}-byte response limit."
                        )
                    if not streamed.is_stream_consumed:
                        async for chunk in streamed.aiter_raw():
                            content.extend(chunk)
                            if len(content) > max_response_bytes:
                                raise PageFetchError(
                                    f"Page exceeds the {max_response_bytes}-byte response limit."
                                )
                    response_headers = httpx.Headers(
                        (name, value)
                        for name, value in streamed.headers.raw
                        if name.lower() not in {b"content-encoding", b"content-length"}
                    )
                    return httpx.Response(
                        status_code=streamed.status_code,
                        headers=response_headers,
                        content=bytes(content),
                        request=httpx.Request("GET", original_url, headers=request_headers),
                    )
            except (httpx.ConnectError, httpx.ConnectTimeout) as error:
                last_connect_error = error
                continue

        if redirect_url is not None:
            current_url = redirect_url
            continue
        if last_connect_error is not None:
            raise last_connect_error
        raise PageFetchError(f"Could not fetch a public address for {current_url}")

    raise PageFetchError(f"Too many redirects while fetching {url}")


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
    resolve_host: HostResolver = resolve_host_addresses,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
    allowed_origin: str | None = None,
) -> PageSnapshot:
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "identity",
    }
    try:
        if client is not None:
            response = await fetch_public_response(
                client,
                url,
                headers=headers,
                timeout=timeout,
                resolve_host=resolve_host,
                max_response_bytes=max_response_bytes,
                allowed_origin=allowed_origin,
            )
        else:
            async with httpx.AsyncClient(
                follow_redirects=False,
                headers=headers,
                timeout=timeout,
                trust_env=False,
            ) as default_client:
                response = await fetch_public_response(
                    default_client,
                    url,
                    headers=headers,
                    timeout=timeout,
                    resolve_host=resolve_host,
                    max_response_bytes=max_response_bytes,
                    allowed_origin=allowed_origin,
                )
    except (httpx.HTTPError, ValueError) as error:
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
    resolve_host: HostResolver = resolve_host_addresses,
    max_response_bytes: int = MAX_RESPONSE_BYTES,
) -> PageSnapshot:
    if client is None:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
            "Accept-Encoding": "identity",
        }
        async with httpx.AsyncClient(
            follow_redirects=False,
            headers=headers,
            timeout=timeout,
            trust_env=False,
        ) as default_client:
            return await fetch_enriched_page_snapshot(
                url,
                client=default_client,
                timeout=timeout,
                max_linked_pages=max_linked_pages,
                resolve_host=resolve_host,
                max_response_bytes=max_response_bytes,
            )

    root_snapshot = await fetch_page_snapshot(
        url,
        client=client,
        timeout=timeout,
        resolve_host=resolve_host,
        max_response_bytes=max_response_bytes,
    )
    linked_snapshots = []
    for candidate_url in same_domain_candidate_urls(root_snapshot, max_linked_pages):
        try:
            linked_snapshots.append(
                await fetch_page_snapshot(
                    candidate_url,
                    client=client,
                    timeout=timeout,
                    resolve_host=resolve_host,
                    max_response_bytes=max_response_bytes,
                )
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
    links = select_snapshot_links(PageLink(url=url, text=text) for url, text in parser.links if url)
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
    root_links = select_snapshot_links(root_snapshot.links)
    if not linked_snapshots and root_links == root_snapshot.links:
        return root_snapshot

    text_parts = [root_snapshot.normalized_text]
    combined_links = [*root_snapshot.links]
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
        combined_links.extend(linked_snapshot.links)

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
        links=select_snapshot_links(combined_links),
    )


def select_snapshot_links(links: Iterable[PageLink]) -> tuple[PageLink, ...]:
    """Keep a bounded, useful, auditable set of unique public-page links."""

    unique: dict[str, PageLink] = {}
    relevant: set[str] = set()
    for link in links:
        parsed = urlparse(link.url)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            continue
        key = normalize_fetch_url(link.url)
        is_relevant = any(
            term in f"{link.text} {link.url}".casefold() for term in LINK_DISCOVERY_TERMS
        )
        if key not in unique or (is_relevant and key not in relevant):
            unique[key] = link
        if is_relevant:
            relevant.add(key)

    ordered_keys = [key for key in unique if key in relevant]
    ordered_keys.extend(key for key in unique if key not in relevant)
    return tuple(unique[key] for key in ordered_keys[:MAX_SNAPSHOT_LINKS])


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
