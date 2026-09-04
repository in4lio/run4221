"""One-engine invariant: the legacy extractor stack must not grow back.

The researcher engine is the only AI lever in the codebase. These checks pin
that by construction: no references to the deleted legacy "ai" package, no
provider imports outside the researcher package, no heuristic provider
classes, and no legacy prompt keys.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

# The escaped dot keeps this file itself from matching the banned plain text.
LEGACY_PACKAGE_PATTERN = re.compile(r"run4221\.ai(?![0-9A-Za-z_])")
PROVIDER_IMPORT_PATTERN = re.compile(r"^\s*(?:from|import)\s+(?:openai|agents)\b", re.MULTILINE)
HEURISTIC_CLASS_PATTERN = re.compile(r"\bclass\s+Heuristic")

RESEARCHER_PACKAGE = SRC_DIR / "run4221" / "researcher"
MODERATOR_TOOLS = SRC_DIR / "run4221" / "agent" / "moderator_tools.py"


def python_files(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    )


def test_legacy_ai_package_is_gone_from_src_and_tests() -> None:
    assert not (SRC_DIR / "run4221" / "ai").exists()

    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for root in (SRC_DIR, TESTS_DIR)
        for path in python_files(root)
        if LEGACY_PACKAGE_PATTERN.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_only_the_researcher_package_imports_the_provider_sdks() -> None:
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in python_files(SRC_DIR)
        if PROVIDER_IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
        and not path.is_relative_to(RESEARCHER_PACKAGE)
    ]

    assert offenders == []


def test_no_heuristic_provider_classes_remain_in_src() -> None:
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in python_files(SRC_DIR)
        if HEURISTIC_CLASS_PATTERN.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_legacy_prompt_keys_survive_only_as_the_moderator_tool_name() -> None:
    # "discover_event_profile" remains the public moderator agent tool NAME in
    # moderator_tools.py; as a prompt key it must be gone everywhere else.
    allowed_paths = {"discover_event_profile": {MODERATOR_TOOLS}}

    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}: {key}"
        for path in python_files(SRC_DIR)
        for key in ("discover_event_profile", "update_registration_window")
        if key in path.read_text(encoding="utf-8")
        and path not in allowed_paths.get(key, set())
    ]

    assert offenders == []
