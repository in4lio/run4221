from __future__ import annotations

from urllib.parse import urlsplit

from run4221.researcher.schemas import validate_http_url


def source_domain(url: str) -> str:
    validate_http_url(url)
    hostname = urlsplit(url).hostname
    assert hostname is not None
    return hostname.rstrip(".").casefold()
