from __future__ import annotations

from sqlalchemy import text

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


def main() -> None:
    check_database()
    print("run4221 health check passed")


if __name__ == "__main__":
    main()
