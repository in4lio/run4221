from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Protocol


class ChannelEvent(Protocol):
    name: str
    distance_label: str
    event_date: str | None
    location: str
    official_url: str
    registration_url: str | None
    registration_open_at: str | None
    registration_close_at: str | None


MESSAGE_HEADINGS = {
    "event_announced": "New event",
    "registration_date_discovered": "Registration date announced",
    "opens_tomorrow": "Registration opens tomorrow",
    "opens_today": "Registration opens today",
    "closes_tomorrow": "Registration closes tomorrow",
    "registration_open": "Registration is open",
    "registration_updated": "Registration update",
    "registration_closed": "Registration is closed",
    "sold_out": "Sold out",
    "waitlist": "Waitlist available",
    "correction": "Event update",
}

MESSAGE_EMOJIS = {
    "event_announced": "✨",
    "registration_date_discovered": "📅",
    "opens_tomorrow": "🔔",
    "opens_today": "🔔",
    "closes_tomorrow": "🔔",
    "registration_open": "✅",
    "registration_updated": "⚠️",
    "registration_closed": "🔒",
    "sold_out": "⛔",
    "waitlist": "⏳",
    "correction": "⚠️",
}


def render_channel_message(
    message_type: str,
    event: ChannelEvent,
    *,
    update_lines: tuple[str, ...] = (),
) -> str:
    """Render deterministic public copy from approved event fields only."""

    try:
        heading = MESSAGE_HEADINGS[message_type]
        emoji = MESSAGE_EMOJIS[message_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported channel message type: {message_type}") from exc
    if message_type == "registration_updated" and not update_lines:
        raise ValueError("Registration update requires a concrete changed value")

    lines = [
        f"<b>{escape(event.name)}</b>",
        escape(event.distance_label),
    ]
    if event.event_date:
        lines.append(f"Event date: {escape(event.event_date)}")
    lines.append(escape(event.location))
    lines.extend(("", f"<b>{emoji} {escape(heading)}</b>"))

    if message_type in {"registration_date_discovered", "opens_tomorrow", "opens_today"}:
        if event.registration_open_at:
            lines.append(escape(format_public_value(event.registration_open_at)))
    if message_type in {"closes_tomorrow", "registration_closed"}:
        if event.registration_close_at:
            lines.append(escape(format_public_value(event.registration_close_at)))
    lines.extend(escape(line) for line in update_lines)

    lines.extend(
        (
            "",
            f'<a href="{escape(event.official_url, quote=True)}">Official event page</a>',
        )
    )
    if event.registration_url and event.registration_url != event.official_url:
        lines.append(
            f'<a href="{escape(event.registration_url, quote=True)}">Registration page</a>'
        )
    return "\n".join(lines)


def format_public_value(value: str) -> str:
    """Keep dates unchanged and make ISO datetimes readable in Telegram."""

    if len(value) == 10:
        return value
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value.replace("T", " ")
    rendered = parsed.strftime("%Y-%m-%d %H:%M:%S")
    offset = parsed.strftime("%z")
    if offset:
        rendered += f" {offset[:3]}:{offset[3:]}"
    return rendered
