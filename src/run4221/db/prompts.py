from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select

from run4221.db import models
from run4221.db.bootstrap import ensure_database_schema
from run4221.db.models import utcnow
from run4221.db.session import session_scope

RESEARCH_AGENT_PROMPT = "research_agent"

PROMPT_STATUSES = {"draft", "active", "previous", "failed", "retired"}


@dataclass(frozen=True)
class PromptVersionRecord:
    id: int
    prompt_key: str
    version: int
    content: str
    status: str
    created_by: str | None
    failure_reason: str | None = None
    fallback_reason: str | None = None
    source: str = "db"
    file_path: str | None = None


class PromptConfigError(RuntimeError):
    pass


def upsert_active_prompt_version(
    prompt_key: str,
    content: str,
    *,
    database_url: str | None = None,
    created_by: str = "seed",
) -> PromptVersionRecord:
    normalized_key = normalize_prompt_key(prompt_key)
    normalized_content = validate_prompt_content(content)
    ensure_database_schema(database_url)

    with session_scope(database_url) as session:
        active = latest_prompt_model(session, normalized_key, status="active")
        if active is not None and active.content == normalized_content:
            return prompt_model_to_record(active)

        next_version = (
            session.scalar(
                select(func.max(models.PromptVersion.version)).where(
                    models.PromptVersion.prompt_key == normalized_key
                )
            )
            or 0
        ) + 1

        now = utcnow()
        for current_active in session.scalars(
            select(models.PromptVersion).where(
                models.PromptVersion.prompt_key == normalized_key,
                models.PromptVersion.status == "active",
            )
        ):
            current_active.status = "previous"
            current_active.retired_at = now

        model = models.PromptVersion(
            prompt_key=normalized_key,
            version=next_version,
            content=normalized_content,
            status="active",
            created_by=created_by,
            activated_at=now,
        )
        session.add(model)
        session.flush()
        return prompt_model_to_record(model)


def get_runtime_prompt(
    prompt_key: str,
    *,
    database_url: str | None = None,
) -> PromptVersionRecord:
    normalized_key = normalize_prompt_key(prompt_key)
    ensure_database_schema(database_url)

    with session_scope(database_url) as session:
        active = latest_prompt_model(session, normalized_key, status="active")
        if active is not None and active.content.strip():
            return prompt_model_to_record(active)

        fallback_reason = "Active prompt is missing."
        if active is not None:
            active.status = "failed"
            active.failure_reason = "Active prompt content is empty."
            active.retired_at = utcnow()
            fallback_reason = "Active prompt is invalid; using previous DB version."

        previous = latest_prompt_model(session, normalized_key, status="previous")
        if previous is not None and previous.content.strip():
            return prompt_model_to_record(previous, fallback_reason=fallback_reason)

    raise PromptConfigError(f"No usable DB prompt found for {normalized_key}.")


def get_file_prompt(
    prompt_key: str,
    *,
    prompts_dir: str | Path,
) -> PromptVersionRecord:
    normalized_key = normalize_prompt_key(prompt_key)
    directory = Path(prompts_dir)
    if not directory.exists():
        raise PromptConfigError(f"Prompt directory does not exist: {directory}")
    if not directory.is_dir():
        raise PromptConfigError(f"Prompt path is not a directory: {directory}")

    for suffix in (".instructions.txt", ".prompt.txt", ".txt"):
        path = directory / f"{normalized_key}{suffix}"
        if path.exists():
            content = validate_prompt_content(path.read_text(encoding="utf-8"))
            return PromptVersionRecord(
                id=0,
                prompt_key=normalized_key,
                version=0,
                content=content,
                status="file",
                created_by=None,
                source="file",
                file_path=str(path),
            )

    raise PromptConfigError(f"No usable prompt file found for {normalized_key} in {directory}.")


def mark_prompt_failed_and_restore_previous(
    prompt_key: str,
    version: int,
    reason: str,
    *,
    database_url: str | None = None,
) -> PromptVersionRecord | None:
    normalized_key = normalize_prompt_key(prompt_key)
    ensure_database_schema(database_url)

    with session_scope(database_url) as session:
        model = session.scalar(
            select(models.PromptVersion).where(
                models.PromptVersion.prompt_key == normalized_key,
                models.PromptVersion.version == version,
            )
        )
        if model is None:
            return None

        model.status = "failed"
        model.failure_reason = reason.strip() or "Prompt failed during execution."
        model.retired_at = utcnow()

        previous = latest_prompt_model(session, normalized_key, status="previous")
        if previous is None:
            session.flush()
            return prompt_model_to_record(model)

        previous.status = "active"
        previous.activated_at = utcnow()
        previous.retired_at = None
        session.flush()
        return prompt_model_to_record(previous, fallback_reason="Restored previous DB version.")


def latest_prompt_model(
    session,
    prompt_key: str,
    *,
    status: str,
) -> models.PromptVersion | None:
    return session.scalar(
        select(models.PromptVersion)
        .where(
            models.PromptVersion.prompt_key == prompt_key,
            models.PromptVersion.status == status,
        )
        .order_by(models.PromptVersion.version.desc(), models.PromptVersion.id.desc())
        .limit(1)
    )


def prompt_model_to_record(
    model: models.PromptVersion,
    *,
    fallback_reason: str | None = None,
) -> PromptVersionRecord:
    return PromptVersionRecord(
        id=model.id,
        prompt_key=model.prompt_key,
        version=model.version,
        content=model.content,
        status=model.status,
        created_by=model.created_by,
        failure_reason=model.failure_reason,
        fallback_reason=fallback_reason,
        source="db",
    )


def normalize_prompt_key(value: str) -> str:
    normalized = value.strip().casefold().replace("-", "_")
    if not re.fullmatch(r"[a-z0-9_]+", normalized):
        raise PromptConfigError(f"Invalid prompt key: {value}")
    return normalized


def validate_prompt_content(content: str) -> str:
    normalized_content = content.strip()
    if not normalized_content:
        raise PromptConfigError("Prompt content cannot be empty.")
    return normalized_content
