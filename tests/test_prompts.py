import pytest

from run4221.db import models
from run4221.db.prompts import (
    RESEARCH_AGENT_PROMPT,
    PromptConfigError,
    get_file_prompt,
    get_runtime_prompt,
    mark_prompt_failed_and_restore_previous,
    upsert_active_prompt_version,
)
from run4221.db.seed_prompts import load_prompt_seed_dir, seed_prompts_from_dir
from run4221.db.session import session_scope


def database_url(tmp_path) -> str:
    return f"sqlite:///{tmp_path / 'run4221-test.sqlite3'}"


def test_prompt_versions_promote_active_and_keep_previous(tmp_path) -> None:
    url = database_url(tmp_path)

    first = upsert_active_prompt_version(
        RESEARCH_AGENT_PROMPT,
        "First prompt.",
        database_url=url,
    )
    same = upsert_active_prompt_version(
        RESEARCH_AGENT_PROMPT,
        "First prompt.",
        database_url=url,
    )
    second = upsert_active_prompt_version(
        RESEARCH_AGENT_PROMPT,
        "Second prompt.",
        database_url=url,
    )
    runtime = get_runtime_prompt(RESEARCH_AGENT_PROMPT, database_url=url)

    assert first.version == 1
    assert same.version == 1
    assert second.version == 2
    assert runtime.content == "Second prompt."
    with session_scope(url) as session:
        previous = session.get(models.PromptVersion, first.id)
        assert previous is not None
        assert previous.status == "previous"


def test_runtime_prompt_falls_back_to_previous_when_active_is_invalid(tmp_path) -> None:
    url = database_url(tmp_path)
    first = upsert_active_prompt_version(
        "research_agent",
        "First prompt.",
        database_url=url,
    )
    second = upsert_active_prompt_version(
        "research_agent",
        "Second prompt.",
        database_url=url,
    )
    with session_scope(url) as session:
        active = session.get(models.PromptVersion, second.id)
        assert active is not None
        active.content = " "

    runtime = get_runtime_prompt("research_agent", database_url=url)

    assert runtime.id == first.id
    assert runtime.content == "First prompt."
    assert runtime.fallback_reason == "Active prompt is invalid; using previous DB version."


def test_runtime_prompt_requires_usable_db_prompt(tmp_path) -> None:
    with pytest.raises(PromptConfigError, match="No usable DB prompt"):
        get_runtime_prompt("research_agent", database_url=database_url(tmp_path))


def test_file_prompt_loads_private_style_text_file(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_path = prompts_dir / "research_agent.instructions.txt"
    prompt_path.write_text("File prompt.", encoding="utf-8")

    runtime = get_file_prompt("research_agent", prompts_dir=prompts_dir)

    assert runtime.source == "file"
    assert runtime.status == "file"
    assert runtime.version == 0
    assert runtime.content == "File prompt."
    assert runtime.file_path == str(prompt_path)


def test_file_prompt_requires_existing_prompt_file(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    with pytest.raises(PromptConfigError, match="No usable prompt file"):
        get_file_prompt("research_agent", prompts_dir=prompts_dir)


def test_mark_prompt_failed_restores_previous_version(tmp_path) -> None:
    url = database_url(tmp_path)
    first = upsert_active_prompt_version(
        "research_agent",
        "First prompt.",
        database_url=url,
    )
    second = upsert_active_prompt_version(
        "research_agent",
        "Second prompt.",
        database_url=url,
    )

    restored = mark_prompt_failed_and_restore_previous(
        "research_agent",
        second.version,
        "Bad prompt output.",
        database_url=url,
    )
    runtime = get_runtime_prompt("research_agent", database_url=url)

    assert restored is not None
    assert restored.id == first.id
    assert runtime.id == first.id
    with session_scope(url) as session:
        failed = session.get(models.PromptVersion, second.id)
        assert failed is not None
        assert failed.status == "failed"
        assert failed.failure_reason == "Bad prompt output."


def test_seed_prompts_from_private_style_text_files(tmp_path) -> None:
    url = database_url(tmp_path)
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "event_notes.instructions.txt").write_text(
        "Event notes prompt.",
        encoding="utf-8",
    )
    (prompts_dir / "research_agent.instructions.txt").write_text(
        "Research agent prompt.",
        encoding="utf-8",
    )
    (prompts_dir / "weekly_digest.instructions.txt").write_text(
        "Weekly digest prompt.",
        encoding="utf-8",
    )

    seed_records = load_prompt_seed_dir(prompts_dir)
    prompt_versions = seed_prompts_from_dir(prompts_dir, database_url=url)

    assert [record.prompt_key for record in seed_records] == [
        "event_notes",
        RESEARCH_AGENT_PROMPT,
        "weekly_digest",
    ]
    assert [record.prompt_key for record in prompt_versions] == [
        "event_notes",
        RESEARCH_AGENT_PROMPT,
        "weekly_digest",
    ]
    assert get_runtime_prompt(RESEARCH_AGENT_PROMPT, database_url=url).content == (
        "Research agent prompt."
    )
