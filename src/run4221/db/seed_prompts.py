from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from run4221.db.prompts import (
    PromptVersionRecord,
    normalize_prompt_key,
    upsert_active_prompt_version,
    validate_prompt_content,
)


@dataclass(frozen=True)
class PromptSeedRecord:
    prompt_key: str
    content: str
    path: Path


def load_prompt_seed_dir(prompts_dir: str | Path) -> tuple[PromptSeedRecord, ...]:
    directory = Path(prompts_dir)
    if not directory.exists():
        return ()
    if not directory.is_dir():
        raise ValueError(f"Prompt seed path is not a directory: {directory}")

    records = []
    for path in sorted(directory.glob("*.txt")):
        prompt_key = prompt_key_from_path(path)
        content = validate_prompt_content(path.read_text(encoding="utf-8"))
        records.append(
            PromptSeedRecord(
                prompt_key=prompt_key,
                content=content,
                path=path,
            )
        )

    return tuple(records)


def seed_prompt_records(
    records: tuple[PromptSeedRecord, ...],
    *,
    database_url: str | None = None,
    created_by: str = "seed",
) -> tuple[PromptVersionRecord, ...]:
    return tuple(
        upsert_active_prompt_version(
            record.prompt_key,
            record.content,
            database_url=database_url,
            created_by=created_by,
        )
        for record in records
    )


def seed_prompts_from_dir(
    prompts_dir: str | Path,
    *,
    database_url: str | None = None,
    created_by: str = "seed",
) -> tuple[PromptVersionRecord, ...]:
    return seed_prompt_records(
        load_prompt_seed_dir(prompts_dir),
        database_url=database_url,
        created_by=created_by,
    )


def prompt_key_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".instructions.txt", ".prompt.txt", ".txt"):
        if name.endswith(suffix):
            return normalize_prompt_key(name.removesuffix(suffix))
    return normalize_prompt_key(path.stem)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed active prompt versions from text files.")
    parser.add_argument("--prompts-dir", default="private/prompts")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)

    records = seed_prompts_from_dir(args.prompts_dir, database_url=args.database_url)
    print(f"Seeded {len(records)} active prompt versions.")
    for record in records:
        print(f"{record.prompt_key} v{record.version}: {record.status}")


if __name__ == "__main__":
    main()
