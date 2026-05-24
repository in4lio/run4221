from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from run4221.db.models import Base, Event, EventSuggestion
from run4221.db.seed import seed_initial_data
from run4221.db.session import get_engine, session_scope


def ensure_database_schema(database_url: str | None = None) -> None:
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)


def initialize_database(
    database_url: str | None = None,
    *,
    seed_initial_events: bool = True,
) -> None:
    ensure_database_schema(database_url)
    if not seed_initial_events:
        return

    with session_scope(database_url) as session:
        if database_is_empty(session):
            seed_initial_data(session)


def database_is_empty(session: Session) -> bool:
    return (
        session.scalar(select(Event.id).limit(1)) is None
        and session.scalar(select(EventSuggestion.id).limit(1)) is None
    )
