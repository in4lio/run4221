from __future__ import annotations

from sqlalchemy import text

from run4221.db.session import get_engine


def check_database(database_url: str | None = None) -> None:
    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))


def main() -> None:
    check_database()
    print("run4221 health check passed")


if __name__ == "__main__":
    main()
