from __future__ import annotations

from pydantic import ValidationError
from sqlalchemy import text

from run4221.config import Settings
from run4221.db.session import (
    get_engine,
    require_initialized_database,
    resolve_database_url,
)


def check_database(database_url: str | None = None) -> None:
    resolved_url = resolve_database_url(database_url)
    require_initialized_database(resolved_url)
    with get_engine(resolved_url).connect() as connection:
        connection.execute(text("SELECT COUNT(*) FROM events"))


def check_runtime() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        details = "; ".join(
            item["msg"]
            for item in error.errors(include_input=False, include_url=False)
        )
        raise RuntimeError(f"run4221 bot configuration is invalid: {details}") from None
    check_database(settings.database_url)


def main() -> None:
    check_runtime()
    print("run4221 health check passed")


if __name__ == "__main__":
    main()
