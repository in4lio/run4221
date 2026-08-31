from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_TRANSACTION_MAX_ATTEMPTS = 3
_SQLITE_RETRY_BASE_SECONDS = 0.05

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_database_path() -> Path:
    return project_root() / "data" / "run4221.sqlite3"


def default_database_url() -> str:
    return f"sqlite:///{default_database_path()}"


def resolve_database_url(database_url: str | None = None) -> str:
    if database_url:
        return database_url

    return (
        os.getenv("DATABASE_URL")
        or os.getenv("RUN4221_DATABASE_URL")
        or default_database_url()
    )


def sqlite_database_path(database_url: str | None = None) -> Path | None:
    resolved_url = make_url(resolve_database_url(database_url))
    if resolved_url.get_backend_name() != "sqlite":
        return None
    if resolved_url.database in {None, ":memory:"}:
        return None
    return Path(resolved_url.database)


def require_initialized_database(database_url: str | None = None) -> None:
    resolved_url = resolve_database_url(database_url)
    database_path = sqlite_database_path(resolved_url)
    if database_path is not None and (
        not database_path.is_file() or database_path.stat().st_size == 0
    ):
        raise RuntimeError(
            f"Configured database has not been initialized: {database_path}"
        )

    engine = get_engine(resolved_url)
    if not inspect(engine).has_table("events"):
        raise RuntimeError("Configured database is missing the events table.")


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    resolved_url = resolve_database_url(database_url)
    url = make_url(resolved_url)
    is_sqlite = url.get_backend_name() == "sqlite"
    if is_sqlite and url.database is not None:
        database_path = Path(url.database)
        if database_path != Path(":memory:"):
            database_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, object] = {}
    if is_sqlite:
        connect_args = {
            "check_same_thread": False,
            "timeout": SQLITE_BUSY_TIMEOUT_MS / 1_000,
        }

    engine = create_engine(resolved_url, connect_args=connect_args, future=True)
    if is_sqlite:
        event.listen(
            engine,
            "connect",
            _sqlite_connection_configurer(database=url.database),
        )
    return engine


def _sqlite_connection_configurer(
    *,
    database: str | None,
) -> Callable[[Any, Any], None]:
    use_wal = database not in {None, "", ":memory:"}

    def configure(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            if use_wal:
                cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()

    return configure


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


def run_serialized_transaction[ResultT](
    operation: Callable[[Session], ResultT],
    *,
    database_url: str | None = None,
    max_attempts: int = SQLITE_TRANSACTION_MAX_ATTEMPTS,
) -> ResultT:
    """Run a short write transaction, serializing SQLite admission decisions.

    SQLite's busy timeout handles brief contention inside each attempt. A small,
    bounded whole-transaction retry covers a competing writer that outlives that
    timeout without retrying arbitrary operational failures.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    engine = get_engine(database_url)
    for attempt in range(max_attempts):
        session = Session(bind=engine, expire_on_commit=False)
        try:
            if engine.dialect.name == "sqlite":
                session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            result = operation(session)
            session.commit()
            return result
        except OperationalError as exc:
            session.rollback()
            if (
                engine.dialect.name != "sqlite"
                or not _is_sqlite_busy_error(exc)
                or attempt + 1 >= max_attempts
            ):
                raise
            time.sleep(_SQLITE_RETRY_BASE_SECONDS * (2**attempt))
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    raise RuntimeError("Serialized transaction exhausted without returning or raising.")


def _is_sqlite_busy_error(error: OperationalError) -> bool:
    sqlite_error_code = getattr(error.orig, "sqlite_errorcode", None)
    if sqlite_error_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error.orig).casefold()
    return "database is locked" in message or "database table is locked" in message


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    session_factory = make_session_factory(database_url)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
