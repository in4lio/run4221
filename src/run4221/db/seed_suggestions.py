from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from run4221.db.bootstrap import ensure_database_schema
from run4221.db.repository import (
    EVENT_SUGGESTION_MAX_PENDING_TOTAL,
    EventSuggestionCreate,
    EventSuggestionRecord,
    EventWriteError,
    add_event_suggestion,
    count_event_suggestions,
)
from run4221.db.seed_prompts import load_prompt_seed_dir, seed_prompt_records
from run4221.db.session import get_engine, resolve_database_url
from run4221.events import DISTANCE_CODE_TO_KEY, DISTANCE_KEY_TO_CODE, normalize_tag

RESET_CONFIRMATION = "RESET_SUGGESTION_BOOTSTRAP"


@dataclass(frozen=True)
class SuggestionSeedResult:
    created: tuple[EventSuggestionRecord, ...]
    backup_path: Path | None = None


def seed_suggestions_from_file(
    input_path: str | Path,
    *,
    database_url: str | None = None,
    reset_sqlite: bool = False,
    confirm_reset: str | None = None,
) -> SuggestionSeedResult:
    suggestions = load_suggestion_seed_file(input_path)
    validate_seed_batch_size(suggestions)

    backup_path = None
    if reset_sqlite:
        backup_path = reset_sqlite_database(database_url, confirm_reset=confirm_reset)

    ensure_database_schema(database_url)
    validate_seed_queue_capacity(suggestions, database_url=database_url)
    created = tuple(
        add_event_suggestion(suggestion, database_url=database_url)
        for suggestion in suggestions
    )
    return SuggestionSeedResult(created=created, backup_path=backup_path)


def load_suggestion_seed_file(input_path: str | Path) -> tuple[EventSuggestionCreate, ...]:
    path = Path(input_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_suggestions = payload.get("suggestions") if isinstance(payload, dict) else payload
    if not isinstance(raw_suggestions, list):
        raise EventWriteError("Suggestion seed file must contain a JSON list or suggestions list.")

    return tuple(
        suggestion_from_seed_item(item, index=index)
        for index, item in enumerate(raw_suggestions, start=1)
    )


def validate_seed_batch_size(suggestions: tuple[EventSuggestionCreate, ...]) -> None:
    if len(suggestions) > EVENT_SUGGESTION_MAX_PENDING_TOTAL:
        raise EventWriteError(
            "Suggestion seed file contains too many pending suggestions. "
            f"Maximum: {EVENT_SUGGESTION_MAX_PENDING_TOTAL}."
        )


def validate_seed_queue_capacity(
    suggestions: tuple[EventSuggestionCreate, ...],
    *,
    database_url: str | None,
) -> None:
    pending_count = count_event_suggestions(database_url=database_url)
    if pending_count + len(suggestions) > EVENT_SUGGESTION_MAX_PENDING_TOTAL:
        raise EventWriteError(
            "Suggestion queue does not have enough space for this seed file. "
            f"Pending: {pending_count}. "
            f"Maximum: {EVENT_SUGGESTION_MAX_PENDING_TOTAL}."
        )


def suggestion_from_seed_item(item: object, *, index: int) -> EventSuggestionCreate:
    if not isinstance(item, dict):
        raise EventWriteError(f"Suggestion #{index} must be a JSON object.")

    event_name = seed_text(item, "event_name", "name")
    if event_name is None:
        raise EventWriteError(f"Suggestion #{index} requires event_name.")

    distances = parse_distance_seed_value(item.get("distances"))
    if not distances:
        raise EventWriteError(f"Suggestion #{index} requires at least one distance.")

    return EventSuggestionCreate(
        event_name=event_name,
        url=seed_text(item, "url", "official_url", "registration_url"),
        event_date=seed_text(item, "event_date", "date"),
        location=seed_text(item, "location"),
        region_tags=parse_region_seed_value(item.get("region_tags") or item.get("tags")),
        distances=distances,
        note=seed_text(item, "note"),
        submitter_user_id=seed_text(item, "submitter_user_id") or "seed:launch",
        submitter_username=seed_text(item, "submitter_username") or "run4221_seed",
        submitter_display_name=seed_text(item, "submitter_display_name") or "Run4221 seed",
        submitter_is_moderator=bool(item.get("submitter_is_moderator", True)),
    )


def reset_sqlite_database(
    database_url: str | None = None,
    *,
    confirm_reset: str | None,
) -> Path | None:
    if confirm_reset != RESET_CONFIRMATION:
        raise EventWriteError(f"Reset requires --confirm-reset {RESET_CONFIRMATION}.")

    database_path = sqlite_database_path(database_url)
    if database_path is None:
        raise EventWriteError("Reset is only supported for file-based SQLite databases.")
    if not database_path.exists():
        return None

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_path = database_path.with_name(f"{database_path.name}.{timestamp}.bak")
    shutil.copy2(database_path, backup_path)
    database_path.unlink()
    get_engine.cache_clear()
    return backup_path


def sqlite_database_path(database_url: str | None = None) -> Path | None:
    resolved_url = resolve_database_url(database_url)
    if not resolved_url.startswith("sqlite:///"):
        return None

    raw_path = resolved_url.removeprefix("sqlite:///")
    if raw_path in {"", ":memory:"}:
        return None
    return Path(raw_path)


def seed_text(item: dict[str, object], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def parse_distance_seed_value(value: object) -> tuple[str, ...]:
    terms = split_seed_terms(value)
    distances: list[str] = []
    for term in terms:
        normalized = term.casefold().strip()
        distance = parse_distance_term(normalized)
        if distance is None:
            raise EventWriteError(f"Unsupported distance in suggestion seed: {term}")
        distances.append(distance)
    return tuple(dict.fromkeys(distances))


def parse_distance_term(term: str) -> str | None:
    if term in DISTANCE_CODE_TO_KEY:
        return DISTANCE_CODE_TO_KEY[term]
    if term.endswith("k") and term[:-1] in DISTANCE_CODE_TO_KEY:
        return DISTANCE_CODE_TO_KEY[term[:-1]]
    if term in DISTANCE_KEY_TO_CODE:
        return term

    normalized_tag = normalize_tag(term)
    if normalized_tag in DISTANCE_KEY_TO_CODE:
        return normalized_tag
    return None


def parse_region_seed_value(value: object) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalize_tag(term) for term in split_seed_terms(value)))


def split_seed_terms(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(value, list | tuple):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return (str(value).strip(),) if str(value).strip() else ()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Seed pending event suggestions for the moderator review workflow."
    )
    parser.add_argument("--input", required=True, help="Path to suggestion seed JSON.")
    parser.add_argument(
        "--database-url",
        default=None,
        help="Database URL. Defaults to DATABASE_URL.",
    )
    parser.add_argument(
        "--reset-sqlite",
        action="store_true",
        help="Back up and remove a file-based SQLite database before seeding.",
    )
    parser.add_argument(
        "--confirm-reset",
        default=None,
        help=f"Required with --reset-sqlite: {RESET_CONFIRMATION}",
    )
    parser.add_argument(
        "--prompts-dir",
        default="private/prompts",
        help="Optional prompt seed directory. Defaults to private/prompts when it exists.",
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Do not seed prompt versions from --prompts-dir.",
    )
    args = parser.parse_args(argv)

    prompt_records = ()
    if not args.skip_prompts:
        prompt_records = load_prompt_seed_dir(args.prompts_dir)

    result = seed_suggestions_from_file(
        args.input,
        database_url=args.database_url,
        reset_sqlite=args.reset_sqlite,
        confirm_reset=args.confirm_reset,
    )
    prompt_versions = seed_prompt_records(
        prompt_records,
        database_url=args.database_url,
        created_by="seed_suggestions",
    )
    if result.backup_path is not None:
        print(f"Backed up previous database to {result.backup_path}")
    print(f"Seeded {len(result.created)} pending suggestions.")
    for suggestion in result.created:
        print(f"#{suggestion.id}: {suggestion.event_name}")
    if prompt_versions:
        print(f"Seeded {len(prompt_versions)} active prompt versions.")
        for prompt in prompt_versions:
            print(f"{prompt.prompt_key} v{prompt.version}: {prompt.status}")


if __name__ == "__main__":
    main()
