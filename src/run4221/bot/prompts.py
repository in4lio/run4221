from __future__ import annotations

from html import escape

PROMPT_ARTICLES = {
    "archived event ID": "an",
    "event ID": "an",
    "event name": "an",
    "field name": "a",
    "new value": "a",
    "official event or registration URL": "an",
    "official event URL": "an",
    "optional note": "an",
    "search term": "a",
    "suggestion number": "a",
    "update ID": "an",
}


def waiting_prompt(
    flow: str,
    prompt: str,
    *,
    example: str | None = None,
    extra: tuple[str, ...] = (),
) -> str:
    lines = [
        f"<b>💬 {escape(flow)}</b>",
        input_instruction(prompt),
    ]
    if example:
        lines.append(f"<b>Example</b>: <i>{escape(example)}</i>")
    lines.extend(extra)
    return "\n".join(lines)


def input_instruction(prompt: str) -> str:
    article = PROMPT_ARTICLES.get(prompt)
    escaped_prompt = escape(prompt)
    if article:
        return f"Send {article} {escaped_prompt}."

    return f"Send {escaped_prompt}."
