from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


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


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    resolved_url = resolve_database_url(database_url)
    if resolved_url.startswith("sqlite:///"):
        database_path = Path(resolved_url.removeprefix("sqlite:///"))
        if database_path != Path(":memory:"):
            database_path.parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False} if resolved_url.startswith("sqlite") else {}
    return create_engine(resolved_url, connect_args=connect_args, future=True)


def make_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(database_url), expire_on_commit=False)


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
