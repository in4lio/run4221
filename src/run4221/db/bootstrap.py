from __future__ import annotations

from weakref import WeakSet

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from run4221.db.models import Base, Event, EventSuggestion
from run4221.db.seed import seed_initial_data
from run4221.db.session import get_engine, session_scope

# Engines whose schema this process already created. create_all is idempotent
# but issues per-table reflection queries on every call; accessor functions
# call this on every operation, so repeat calls must be cheap. Keyed by the
# live Engine object: a reset (which clears the engine cache) or an evicted
# engine naturally drops out and gets create_all again.
_schema_ready_engines: WeakSet[Engine] = WeakSet()


def ensure_database_schema(database_url: str | None = None) -> None:
    engine = get_engine(database_url)
    if engine in _schema_ready_engines:
        return
    Base.metadata.create_all(engine)
    _schema_ready_engines.add(engine)


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
