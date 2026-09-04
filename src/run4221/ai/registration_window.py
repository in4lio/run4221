from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from run4221.ai.event_extractor import (
    infer_event_date,
    infer_registration_url,
    snapshot_evidence,
)
from run4221.db.repository import (
    ProposedEventUpdateCreate,
    RegistrationWindowApply,
    apply_registration_window_update,
    create_proposed_event_update,
)
from run4221.events import TrackedEvent
from run4221.ingestion.event_identity import (
    conflicts_with_event_identity,
    is_likely_article_url,
    select_registration_url_for_distances,
)
from run4221.ingestion.page_snapshot import (
    PageFetchError,
    PageSnapshot,
    blocked_page_reason,
    fetch_enriched_page_snapshot,
    store_page_snapshot,
)

type SnapshotFetcher = Callable[[str], Awaitable[PageSnapshot]]
type SnapshotStore = Callable[[PageSnapshot], Path]

REGISTRATION_REVIEW_CONFIDENCE = 0.85
REGISTRATION_OPEN_TERMS = (
    "registration is open",
    "registration open",
    "register now",
    "enter now",
    "sign up now",
    "secure your place",
    "sichere dir jetzt",
    "anmeldung geöffnet",
    "jetzt anmelden",
)
REGISTRATION_CLOSED_TERMS = (
    "registration is closed",
    "registration closed",
    "entries are closed",
    "entry is closed",
    "anmeldung geschlossen",
)
SOLD_OUT_TERMS = ("sold out", "ausverkauft")
WAITLIST_TERMS = ("waitlist", "waiting list", "warteliste")
REGISTRATION_DATE_PATTERNS = (
    r"(?:registration|entries|entry|anmeldung|registrierung)\s+"
    r"(?:opens|starts|beginnt|öffnet)\s+(?:on\s+)?(?P<date>[^.]{0,80})",
    r"(?:opens|starts|beginnt|öffnet)\s+(?:on\s+)?(?P<date>[^.]{0,80})\s+"
    r"(?:for\s+)?(?:registration|entries|entry|anmeldung|registrierung)",
)


@dataclass(frozen=True)
class RegistrationWindowExtraction:
    registration_status: str
    registration_open_at: str | None
    registration_open_precision: str
    registration_close_at: str | None
    registration_url: str | None
    event_date: str | None
    confidence: float
    evidence_snippets: tuple[str, ...]
    provider_name: str = "unknown"


@dataclass(frozen=True)
class RegistrationWindowUpdateResult:
    event_id: str
    source_url: str
    registration_status: str
    registration_open_at: str | None
    registration_open_precision: str
    registration_close_at: str | None
    registration_url: str | None
    event_date: str | None
    confidence: float
    evidence: str
    needs_moderator_review: bool
    proposed_update_id: int | None = None
    applied: bool = False


class RegistrationWindowProvider(Protocol):
    provider_name: str

    async def extract(
        self,
        snapshot: PageSnapshot,
        event: TrackedEvent,
    ) -> RegistrationWindowExtraction:
        """Return current registration-window fields extracted from a page snapshot."""


async def update_registration_window(
    event: TrackedEvent,
    *,
    fetch_snapshot: SnapshotFetcher | None = fetch_enriched_page_snapshot,
    store_snapshot: SnapshotStore | None = store_page_snapshot,
    provider: RegistrationWindowProvider | None = None,
    database_url: str | None = None,
    auto_confirm: bool = False,
) -> RegistrationWindowUpdateResult:
    source_url, source_notes = registration_source_url(event)
    if fetch_snapshot is None:
        extraction = fallback_registration_extraction(event)
        evidence_parts = [
            *source_notes,
            "Registration check skipped because no fetcher was configured.",
        ]
    else:
        try:
            snapshot = await fetch_snapshot(source_url)
        except PageFetchError as error:
            extraction = fallback_registration_extraction(
                event,
                confidence=0.05,
                evidence=(f"Page fetch failed: {error}.",),
            )
            evidence_parts = [*source_notes, *extraction.evidence_snippets]
        else:
            snapshot_path = None
            storage_note = ""
            if store_snapshot is not None:
                try:
                    snapshot_path = store_snapshot(snapshot)
                except OSError as error:
                    storage_note = f" Snapshot storage failed: {error}."

            evidence_parts = [
                *source_notes,
                *snapshot_evidence(snapshot, snapshot_path, storage_note),
            ]
            if reason := blocked_page_reason(snapshot):
                extraction = fallback_registration_extraction(
                    event,
                    confidence=0.05,
                    evidence=(f"Page blocked: {reason}.",),
                )
                evidence_parts.extend(extraction.evidence_snippets)
                evidence_parts.append("Registration extractor provider: fallback.")
            else:
                extractor = provider or HeuristicRegistrationWindowProvider()
                extraction = await extractor.extract(snapshot, event)
                evidence_parts.extend(extraction.evidence_snippets)
                evidence_parts.append(
                    f"Registration extractor provider: {extractor.provider_name}."
                )

    confidence = min(max(extraction.confidence, 0.0), 1.0)
    proposed_fields = registration_proposed_fields(extraction)
    current_fields = registration_current_fields(event)
    has_update = has_meaningful_registration_update(current_fields, proposed_fields)
    needs_review = has_update and (
        not auto_confirm
        or confidence < REGISTRATION_REVIEW_CONFIDENCE
        or has_event_profile_change(event, proposed_fields)
    )

    proposed_update_id = None
    applied = False
    if has_update and needs_review:
        proposed = create_proposed_event_update(
            ProposedEventUpdateCreate(
                event_id=event.id,
                update_type="registration_window",
                current_fields=current_fields,
                proposed_fields=proposed_fields,
                evidence=tuple(evidence_parts),
                confidence=confidence,
                change_summary=registration_change_summary(current_fields, proposed_fields),
            ),
            database_url=database_url,
        )
        proposed_update_id = proposed.id
    elif has_update:
        applied_event = apply_registration_window_update(
            event.id,
            RegistrationWindowApply(
                registration_status=extraction.registration_status,
                registration_open_at=extraction.registration_open_at,
                registration_open_precision=extraction.registration_open_precision,
                registration_close_at=extraction.registration_close_at,
                registration_url=extraction.registration_url,
                event_date=extraction.event_date,
            ),
            database_url=database_url,
        )
        applied = applied_event is not None

    return RegistrationWindowUpdateResult(
        event_id=event.id,
        source_url=source_url,
        registration_status=extraction.registration_status,
        registration_open_at=extraction.registration_open_at,
        registration_open_precision=extraction.registration_open_precision,
        registration_close_at=extraction.registration_close_at,
        registration_url=extraction.registration_url,
        event_date=extraction.event_date,
        confidence=confidence,
        evidence=" ".join(evidence_parts),
        needs_moderator_review=needs_review,
        proposed_update_id=proposed_update_id,
        applied=applied,
    )


class HeuristicRegistrationWindowProvider:
    provider_name = "heuristic"

    async def extract(
        self,
        snapshot: PageSnapshot,
        event: TrackedEvent,
    ) -> RegistrationWindowExtraction:
        text = snapshot.normalized_text
        lowered = text.casefold()
        registration_status = infer_registration_status(lowered)
        registration_open_at = infer_registration_open_date(text)
        registration_url = infer_registration_url(snapshot, event.distances)
        registration_url = registration_url or select_registration_url_for_distances(
            tuple((link.url, link.text) for link in snapshot.links),
            event.distances,
            fallback=safe_registration_url(
                event.registration_url,
                event.event_date,
                event.distances,
            ),
        )
        raw_event_date = infer_event_date(text)
        stale_event_date = is_stale_event_date(raw_event_date, event.event_date)
        event_date = None if stale_event_date else raw_event_date
        if stale_event_date:
            registration_status = "unknown"

        confidence = 0.25
        if snapshot.title:
            confidence += 0.1
        if registration_status != "unknown":
            confidence += 0.2
        if registration_open_at:
            confidence += 0.2
        if registration_url:
            confidence += 0.1
        if event_date:
            confidence += 0.1

        evidence = []
        if registration_status != "unknown":
            evidence.append(f"Detected registration status: {registration_status}.")
        if registration_open_at:
            evidence.append(f"Detected registration opening date: {registration_open_at}.")
        if registration_url:
            evidence.append(f"Detected registration URL: {registration_url}.")
        if event_date and event_date != event.event_date:
            evidence.append(f"Detected event date: {event_date}.")
        if stale_event_date:
            evidence.append(
                f"Ignored stale page date {raw_event_date}; saved event date is "
                f"{event.event_date}."
            )

        return RegistrationWindowExtraction(
            registration_status=registration_status,
            registration_open_at=registration_open_at,
            registration_open_precision="date_only" if registration_open_at else "unknown",
            registration_close_at=None,
            registration_url=registration_url,
            event_date=event_date,
            confidence=min(confidence, 0.75),
            evidence_snippets=tuple(evidence),
            provider_name=self.provider_name,
        )


def fallback_registration_extraction(
    event: TrackedEvent,
    *,
    confidence: float = 0.0,
    evidence: tuple[str, ...] = (),
) -> RegistrationWindowExtraction:
    return RegistrationWindowExtraction(
        registration_status=event.registration_status,
        registration_open_at=event.registration_open_at,
        registration_open_precision=event.registration_open_precision,
        registration_close_at=event.registration_close_at,
        registration_url=safe_registration_url(
            event.registration_url,
            event.event_date,
            event.distances,
        ),
        event_date=event.event_date,
        confidence=confidence,
        evidence_snippets=evidence,
        provider_name="fallback",
    )


def registration_source_url(event: TrackedEvent) -> tuple[str, tuple[str, ...]]:
    safe_url = safe_registration_url(
        event.registration_url,
        event.event_date,
        event.distances,
    )
    if safe_url:
        return safe_url, ()

    if event.registration_url:
        return (
            event.official_url,
            (
                "Ignored stored registration URL because it looks like an old article "
                f"or stale page: {event.registration_url}.",
            ),
        )

    return event.official_url, ()


def safe_registration_url(
    value: str | None,
    event_date: str | None,
    distances: tuple[str, ...] = (),
) -> str | None:
    if not value:
        return None
    if is_likely_article_url(value, event_date):
        return None
    if conflicts_with_event_identity(value, "", distances):
        return None

    return value


def is_stale_event_date(candidate: str | None, saved_event_date: str | None) -> bool:
    if not candidate or not saved_event_date:
        return False

    return candidate < saved_event_date


def infer_registration_status(text: str) -> str:
    if any(term in text for term in SOLD_OUT_TERMS):
        return "sold_out"
    if any(term in text for term in WAITLIST_TERMS):
        return "waitlist"
    if any(term in text for term in REGISTRATION_CLOSED_TERMS):
        return "closed"
    if any(term in text for term in REGISTRATION_OPEN_TERMS):
        return "open"

    return "unknown"


def infer_registration_open_date(text: str) -> str | None:
    for pattern in REGISTRATION_DATE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            parsed = infer_event_date(match.group("date"))
            if parsed:
                return parsed

    return None


def registration_current_fields(event: TrackedEvent) -> dict[str, object]:
    return {
        "registration_status": event.registration_status,
        "registration_open_at": event.registration_open_at,
        "registration_open_precision": event.registration_open_precision,
        "registration_close_at": event.registration_close_at,
        "registration_url": event.registration_url,
        "event_date": event.event_date,
    }


def registration_proposed_fields(
    extraction: RegistrationWindowExtraction,
) -> dict[str, object]:
    fields = {
        "registration_status": extraction.registration_status,
        "registration_open_at": extraction.registration_open_at,
        "registration_open_precision": extraction.registration_open_precision,
        "registration_close_at": extraction.registration_close_at,
        "registration_url": extraction.registration_url,
        "event_date": extraction.event_date,
    }
    return {
        field: value
        for field, value in fields.items()
        if value is not None and value not in {"", "unknown"}
    }


def has_meaningful_registration_update(
    current_fields: dict[str, object],
    proposed_fields: dict[str, object],
) -> bool:
    for key, proposed_value in proposed_fields.items():
        if proposed_value in {None, "", "unknown"}:
            continue
        if current_fields.get(key) != proposed_value:
            return True

    return False


def has_event_profile_change(event: TrackedEvent, proposed_fields: dict[str, object]) -> bool:
    proposed_event_date = proposed_fields.get("event_date")
    return proposed_event_date not in {None, "", event.event_date}


def registration_change_summary(
    current_fields: dict[str, object],
    proposed_fields: dict[str, object],
) -> str:
    changed = [
        key
        for key, proposed_value in proposed_fields.items()
        if proposed_value not in {None, "", "unknown"} and current_fields.get(key) != proposed_value
    ]
    if not changed:
        return "No registration changes detected."

    return "Registration update proposed: " + ", ".join(changed) + "."
