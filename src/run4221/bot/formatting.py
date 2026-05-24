from html import escape

from run4221.events import TrackedEvent


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
