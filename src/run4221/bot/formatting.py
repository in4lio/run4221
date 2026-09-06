import re
from dataclasses import dataclass
from html import escape
from urllib.parse import urlsplit

from run4221.events import TrackedEvent

RESEARCHER_WORKER_PREFIX = "Researcher worker:"
RESEARCHER_SOURCE_CHECK_PREFIX = "Source check:"
RESEARCHER_FIELD_SUPPORT_PREFIX = "Researcher field support:"
RESEARCHER_CONFLICT_PREFIX = "Researcher conflict:"
RESEARCHER_DECISION_PREFIX = "researcher-decision:v1 "
RESEARCHER_EVIDENCE_PREFIX = "researcher-evidence:v1 "
MAX_RESEARCHER_CAPTURED_SOURCES = 3
MAX_RESEARCHER_FIELD_SUPPORT = 6
MAX_RESEARCHER_CONFLICTS = 4


@dataclass(frozen=True)
class ResearcherCapturedSource:
    source_url: str
    captured_at: str
    artifact_name: str
    hash_prefix: str


@dataclass(frozen=True)
class ResearcherProvenance:
    summary: str
    trust_reason: str
    source_url: str
    captured_at: str
    run_id: str
    artifact_name: str
    hash_prefix: str
    additional_sources: tuple[ResearcherCapturedSource, ...] = ()
    field_support: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def captured_sources(self) -> tuple[ResearcherCapturedSource, ...]:
        return (
            ResearcherCapturedSource(
                source_url=self.source_url,
                captured_at=self.captured_at,
                artifact_name=self.artifact_name,
                hash_prefix=self.hash_prefix,
            ),
            *self.additional_sources,
        )


def format_major_title(title: str) -> str:
    return f"<b>✨ {escape(title)}</b>"


def format_field_line(label: str, value: object, *, kind: str = "text") -> str:
    return f"<b>{escape(label)}</b>: {format_field_value(value, kind=kind)}"


def format_field_value(value: object, *, kind: str = "text") -> str:
    text = str(value)
    if kind == "id":
        return f"<code>{escape(text)}</code>"
    if kind == "tag":
        return f"<u>{escape(text)}</u>"

    return escape(text)


def parse_researcher_provenance(
    evidence: str | tuple[str, ...] | list[str] | None,
) -> ResearcherProvenance | None:
    values = (evidence,) if isinstance(evidence, str) else evidence or ()
    summary = ""
    trust_reason = ""
    decision: dict[str, str] = {}
    captured: list[dict[str, str]] = []
    field_support: list[str] = []
    conflicts: list[str] = []
    for value in values:
        for raw_line in str(value).splitlines():
            line = raw_line.strip()
            if line.startswith(RESEARCHER_WORKER_PREFIX):
                summary = line.removeprefix(RESEARCHER_WORKER_PREFIX).strip()
            elif line.startswith(RESEARCHER_SOURCE_CHECK_PREFIX):
                trust_reason = line.removeprefix(RESEARCHER_SOURCE_CHECK_PREFIX).strip()
            elif (
                line.startswith(RESEARCHER_FIELD_SUPPORT_PREFIX)
                and len(field_support) < MAX_RESEARCHER_FIELD_SUPPORT
            ):
                field_support.append(
                    line.removeprefix(RESEARCHER_FIELD_SUPPORT_PREFIX).strip()
                )
            elif (
                line.startswith(RESEARCHER_CONFLICT_PREFIX)
                and len(conflicts) < MAX_RESEARCHER_CONFLICTS
            ):
                conflicts.append(line.removeprefix(RESEARCHER_CONFLICT_PREFIX).strip())
            elif line.startswith(RESEARCHER_DECISION_PREFIX):
                decision = _parse_researcher_marker(line, RESEARCHER_DECISION_PREFIX)
            elif (
                line.startswith(RESEARCHER_EVIDENCE_PREFIX)
                and len(captured) < MAX_RESEARCHER_CAPTURED_SOURCES
            ):
                captured.append(_parse_researcher_marker(line, RESEARCHER_EVIDENCE_PREFIX))

    if not (summary and decision and captured):
        return None

    captured_sources = tuple(_researcher_captured_source(marker, decision) for marker in captured)
    primary = captured_sources[0]
    return ResearcherProvenance(
        summary=summary or "Researcher-created queue finding.",
        trust_reason=trust_reason.removesuffix("."),
        source_url=primary.source_url,
        captured_at=primary.captured_at,
        run_id=_safe_token(captured[0].get("run", "") or decision.get("run", ""), 128),
        artifact_name=primary.artifact_name,
        hash_prefix=primary.hash_prefix,
        additional_sources=captured_sources[1:],
        field_support=tuple(item for item in field_support if item),
        conflicts=tuple(item for item in conflicts if item),
    )


def _parse_researcher_marker(line: str, prefix: str) -> dict[str, str]:
    body = line.removeprefix(prefix)
    body, separator, captured_at = body.rpartition(" captured_at=")
    if not separator:
        body, captured_at = line.removeprefix(prefix), ""
    body, separator, source = body.partition(" source=")
    if not separator:
        source = ""
    fields = dict(token.split("=", 1) for token in body.split() if "=" in token)
    if source:
        fields["source"] = source
    if captured_at:
        fields["captured_at"] = captured_at
    return fields


def _safe_source_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "unknown"
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return "unknown"
    return value


def _safe_token(value: str, max_length: int) -> str:
    return value if re.fullmatch(rf"[A-Za-z0-9_.:+-]{{1,{max_length}}}", value) else "unknown"


def _safe_artifact_basename(value: str) -> str:
    return _safe_token(re.split(r"[/\\]", value)[-1], 180)


def _safe_hash_prefix(value: str) -> str:
    return value[:12] if re.fullmatch(r"[a-f0-9]{12,64}", value) else "unknown"


def _researcher_captured_source(
    captured: dict[str, str],
    decision: dict[str, str],
) -> ResearcherCapturedSource:
    return ResearcherCapturedSource(
        source_url=_safe_source_url(captured.get("source", "")),
        captured_at=_safe_token(captured.get("captured_at", ""), 80),
        artifact_name=_safe_artifact_basename(
            captured.get("artifact", "") or decision.get("artifact", "")
        ),
        hash_prefix=_safe_hash_prefix(captured.get("sha256", "") or decision.get("sha256", "")),
    )


def format_researcher_source_check(
    provenance: ResearcherProvenance | None,
    *,
    max_html_chars: int | None = None,
) -> str:
    if provenance is None:
        return ""

    source_fields: list[tuple[str, object, int, str]] = []
    for index, source in enumerate(provenance.captured_sources, start=1):
        suffix = "" if index == 1 else f" {index}"
        source_fields.extend(
            [
                (f"Source{suffix}", source.source_url, 500, "text"),
                (f"Captured{suffix}", source.captured_at, 100, "text"),
                (f"Artifact{suffix}", source.artifact_name, 180, "text"),
                (f"Hash{suffix}", source.hash_prefix, 32, "id"),
            ]
        )
    optional_fields = [*source_fields, ("Evidence", provenance.summary, 400, "text")]
    if provenance.trust_reason:
        optional_fields.append(("Trust", provenance.trust_reason, 200, "text"))
    support_fields: list[tuple[str, object, int, str]] = []
    if provenance.field_support:
        # The stored evidence keeps the full "field <- artifact#hash" refs;
        # the moderator card shows only the field names.
        supported = ", ".join(
            entry.split(" <- ", 1)[0].strip() or entry.strip()
            for entry in provenance.field_support
        )
        support_fields.append(("Supported fields", supported, 300, "text"))
    critical_fields = [
        ("Run ID", provenance.run_id, 160, "id"),
        *support_fields,
        *(("Conflict", conflict, 350, "text") for conflict in provenance.conflicts),
    ]
    fields = [*optional_fields, *support_fields]
    fields.extend(
        ("Conflict", conflict, 350, "text") for conflict in provenance.conflicts
    )
    fields.append(("Run ID", provenance.run_id, 160, "id"))
    if max_html_chars is not None:
        return _format_bounded_source_check(
            critical_fields,
            optional_fields,
            max_html_chars=max_html_chars,
        )

    lines = ["<blockquote><b>Source check</b>"]
    lines.extend(
        format_bounded_field_line(label, value, max_html_chars=limit, kind=kind)
        for label, value, limit, kind in fields
    )
    lines[-1] += "</blockquote>"
    return "\n".join(lines)


def _format_bounded_source_check(
    critical_fields: list[tuple[str, object, int, str]],
    optional_fields: list[tuple[str, object, int, str]],
    *,
    max_html_chars: int,
) -> str:
    header = "<blockquote><b>Source check</b>"
    footer = "</blockquote>"
    if max_html_chars < len(header) + len(footer):
        return ""

    line_budget = (
        max_html_chars - len(header) - len(footer) - len(critical_fields)
    ) // max(1, len(critical_fields))
    lines = [header]
    for label, value, desired_limit, kind in critical_fields:
        structural = len(f"<b>{escape(label)}</b>: ") + (13 if kind == "id" else 0)
        lines.append(
            format_bounded_field_line(
                label,
                value,
                max_html_chars=max(1, min(desired_limit, line_budget - structural)),
                kind=kind,
            )
        )
    if len("\n".join(lines)) + len(footer) > max_html_chars:
        return ""

    omitted = False
    for label, value, desired_limit, kind in optional_fields:
        structural = len(f"<b>{escape(label)}</b>: ") + (13 if kind == "id" else 0)
        available = max_html_chars - len("\n".join(lines)) - len(footer) - 1 - structural
        if available < 16:
            omitted = True
            continue
        lines.append(
            format_bounded_field_line(
                label,
                value,
                max_html_chars=min(desired_limit, available),
                kind=kind,
            )
        )

    omission_line = "<i>Additional provenance is stored with the update.</i>"
    if omitted and len("\n".join([*lines, omission_line])) + len(footer) <= max_html_chars:
        lines.append(omission_line)
    lines[-1] += footer
    return "\n".join(lines)


def format_bounded_field_line(
    label: str,
    value: object,
    *,
    max_html_chars: int,
    kind: str = "text",
) -> str:
    rendered = bounded_html_escape(value, max_html_chars=max_html_chars)
    if kind == "id":
        rendered = f"<code>{rendered}</code>"
    elif kind == "tag":
        rendered = f"<u>{rendered}</u>"
    return f"<b>{escape(label)}</b>: {rendered}"


def bounded_html_escape(value: object, *, max_html_chars: int) -> str:
    compact = " ".join(str(value).split()) or "unknown"
    rendered = escape(compact, quote=False)
    if len(rendered) <= max_html_chars:
        return rendered

    truncated = rendered[: max_html_chars - 1]
    if truncated.rfind("&") > truncated.rfind(";"):
        truncated = truncated[: truncated.rfind("&")]
    return truncated.rstrip() + "…"


def format_event_card(event: TrackedEvent) -> str:
    date = event.event_date or "date TBA"
    return (
        f"<b>{escape(event.name)}</b>\n"
        f"{escape(event.location)} | {escape(event.distance_label)} | {escape(date)}\n"
        f"{format_field_line('Tags', event.tag_label, kind='tag')}\n"
        f"{format_field_line('ID', event.public_id, kind='id')}"
    )


def format_event_list(events: tuple[TrackedEvent, ...], *, title: str) -> str:
    if not events:
        return "No tracked events found."

    lines = [format_major_title(title), ""]
    lines.extend(format_event_card(event) for event in events)
    return "\n".join(lines)


# Typed provider-limit failure classification. A run status carries only the
# agent's safe detail sentence (or its bare error code when no detail exists),
# so match those exact values instead of substring-scanning free text.
_PROVIDER_LIMIT_ERROR_CODES = frozenset(
    {"authentication", "quota", "rate_limit", "timeout"}
)
_PROVIDER_LIMIT_DETAILS = frozenset(
    {
        "openai authentication failed.",
        "openai quota is unavailable.",
        "openai rate limit was reached.",
        "the bounded agent call timed out.",
    }
)
_QUEUE_FULL_TERMS = ("queue is full", "queue-full", "queue_full")


def research_outcome_headline(result) -> str:
    """One plain-text sentence for a typed researcher run result.

    Accepts any object with ``status`` (state/outcome/detail) plus the optional
    ``queue_reference``/``conflicting_update_id`` fields of a refresh result.
    Every failure is a named message; no evidence-string parsing happens here.
    """

    status = result.status
    state = str(status.status)
    outcome = str(status.outcome)
    detail = (status.detail or "").casefold()
    if state == "failed":
        if detail in _PROVIDER_LIMIT_DETAILS or detail.strip() in _PROVIDER_LIMIT_ERROR_CODES:
            return (
                "The researcher provider rejected the run: "
                "check its configuration or usage limits."
            )
        return "The check failed before reaching a decision."
    if state == "capped":
        if any(term in detail for term in _QUEUE_FULL_TERMS):
            return "The moderation queue is full. Review pending updates, then retry."
        return "The check stopped at its run budget."
    if state == "skipped":
        conflicting_update_id = getattr(result, "conflicting_update_id", None)
        if conflicting_update_id is not None:
            return f"Update #{conflicting_update_id} is already pending for this event."
        return "The captured evidence did not support a validated change."
    if outcome == "no_change":
        return "No material change detected."
    if outcome == "proposal_created":
        reference = str(getattr(result, "queue_reference", None) or "")
        update_id = reference.rsplit(":", 1)[-1]
        return f"Created pending update #{update_id} for moderator review."
    if outcome == "profile_completed":
        return "Profile draft completed."
    return "The check finished without a queueable decision."


def format_event_detail(event: TrackedEvent) -> str:
    date = event.event_date or "date TBA"
    registration_url = event.registration_url or event.official_url
    return "\n".join(
        [
            format_major_title(event.name),
            "",
            format_field_line("ID", event.public_id, kind="id"),
            format_field_line("Location", event.location),
            format_field_line("Distance", event.distance_label),
            format_field_line("Event date", date),
            format_field_line("Tags", event.tag_label, kind="tag"),
            format_field_line(
                "Registration status",
                event.registration_status.replace("_", " "),
            ),
            format_field_line("Registration opens", event.registration_open_at or "unknown"),
            format_field_line(
                "Registration open precision",
                event.registration_open_precision,
            ),
            format_field_line("Registration closes", event.registration_close_at or "unknown"),
            "",
            format_field_line("Official page", event.official_url),
            format_field_line("Registration page", registration_url),
        ]
    )
