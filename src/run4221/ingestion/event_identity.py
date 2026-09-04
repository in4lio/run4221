"""Neutral event identity, timezone, and registration URL helpers.

These helpers are provider-independent and are shared by the researcher,
moderator flows, and the legacy extraction pipeline.
"""

from __future__ import annotations

import re
from urllib.parse import unquote, urlparse

NON_PRIMARY_EVENT_TERMS = (
    "mini",
    "kids",
    "kid",
    "youth",
    "junior",
    "children",
    "school",
    "schueler",
    "schüler",
)

HALF_MARATHON_TERMS = (
    "half",
    "halfmarathon",
    "halb",
    "halbmarathon",
    "21",
    "21k",
    "21km",
)

FULL_MARATHON_TERMS = (
    "full",
    "42",
    "42k",
    "42km",
)

REGISTRATION_LINK_TERMS = (
    "register",
    "registration",
    "entry",
    "entries",
    "sign up",
    "signup",
    "anmeldung",
    "anmelden",
    "teilnahme",
    "startplatz",
)

DATED_ARTICLE_PATH = re.compile(r"/20\d{2}/\d{1,2}/\d{1,2}/")
ARTICLE_PATH_TERMS = (
    "/blog/",
    "/news/",
    "/press/",
    "/story/",
    "/stories/",
    "/article/",
)

CITY_TIMEZONES = {
    "berlin": "Europe/Berlin",
    "karlsruhe": "Europe/Berlin",
    "lisbon": "Europe/Lisbon",
    "prague": "Europe/Prague",
    "copenhagen": "Europe/Copenhagen",
    "cardiff": "Europe/London",
    "valencia": "Europe/Madrid",
    "tokyo": "Asia/Tokyo",
    "boston": "America/New_York",
    "chicago": "America/Chicago",
    "new york": "America/New_York",
    "new york city": "America/New_York",
    "london": "Europe/London",
    "sydney": "Australia/Sydney",
}

COUNTRY_TIMEZONES = {
    "germany": "Europe/Berlin",
    "portugal": "Europe/Lisbon",
    "czech republic": "Europe/Prague",
    "czechia": "Europe/Prague",
    "denmark": "Europe/Copenhagen",
    "united kingdom": "Europe/London",
    "uk": "Europe/London",
    "spain": "Europe/Madrid",
    "japan": "Asia/Tokyo",
    "australia": "Australia/Sydney",
}

REGION_TIMEZONES = {
    "de": "Europe/Berlin",
    "pt": "Europe/Lisbon",
    "cz": "Europe/Prague",
    "dk": "Europe/Copenhagen",
    "gb": "Europe/London",
    "es": "Europe/Madrid",
    "jp": "Asia/Tokyo",
}


def select_registration_url_for_distances(
    candidates: tuple[tuple[str, str], ...],
    distances: tuple[str, ...],
    *,
    fallback: str | None = None,
) -> str | None:
    if len(distances) != 1:
        return fallback

    for url, text in candidates:
        if is_homepage_url(url):
            continue
        if is_likely_article_url(url):
            continue
        if conflicts_with_event_identity(url, text, distances):
            continue
        if matches_distance_url(url, text, distances):
            return url

    return fallback


def is_likely_article_url(value: str, event_date: str | None = None) -> bool:
    parsed = urlparse(value)
    path = parsed.path.casefold()
    searchable = f"{path} {parsed.query}".casefold()
    has_registration_term = any(term in searchable for term in REGISTRATION_LINK_TERMS)
    if has_registration_term:
        return False

    if any(term in path for term in ARTICLE_PATH_TERMS):
        return True

    if DATED_ARTICLE_PATH.search(path):
        return True

    if event_date:
        event_year = event_date[:4]
        years = re.findall(r"\b20\d{2}\b", path)
        return any(year < event_year for year in years)

    return False


def is_homepage_url(value: str) -> bool:
    parsed = urlparse(value)
    return (parsed.path.rstrip("/") or "/") == "/" and not parsed.query


def matches_distance_url(url: str, text: str, distances: tuple[str, ...]) -> bool:
    tokens = event_identity_tokens(url, text)
    if "half_marathon" in distances:
        return any(term in tokens for term in HALF_MARATHON_TERMS)
    if "marathon" in distances:
        return "marathon" in tokens and not any(
            term in tokens for term in HALF_MARATHON_TERMS
        )

    return False


def conflicts_with_event_identity(
    url: str,
    text: str,
    distances: tuple[str, ...],
) -> bool:
    if not distances:
        return False
    tokens = event_identity_tokens(url, text)
    if any(term in tokens for term in NON_PRIMARY_EVENT_TERMS):
        return True
    if "marathon" in distances and "half_marathon" not in distances:
        return any(term in tokens for term in HALF_MARATHON_TERMS)
    if "half_marathon" in distances and "marathon" not in distances:
        return any(term in tokens for term in FULL_MARATHON_TERMS)
    return False


def event_identity_tokens(url: str, text: str) -> frozenset[str]:
    parsed = urlparse(url)
    searchable = unquote(f"{parsed.path} {parsed.query} {text}").casefold()
    return frozenset(re.findall(r"[^\W_]+", searchable))


def resolve_timezone(
    timezone: str,
    country: str,
    regions: tuple[str, ...],
    city: str,
) -> str:
    cleaned_timezone = timezone.strip()
    if cleaned_timezone and cleaned_timezone != "Etc/UTC":
        return cleaned_timezone

    normalized_city = normalize_lookup_key(city)
    if normalized_city in CITY_TIMEZONES:
        return CITY_TIMEZONES[normalized_city]

    normalized_country = normalize_lookup_key(country)
    if normalized_country in COUNTRY_TIMEZONES:
        return COUNTRY_TIMEZONES[normalized_country]

    for region in regions:
        normalized_region = region.casefold().strip().removeprefix("#")
        if normalized_region in REGION_TIMEZONES:
            return REGION_TIMEZONES[normalized_region]

    return cleaned_timezone or "Etc/UTC"


def normalize_lookup_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())
