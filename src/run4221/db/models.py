from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"

    id: Mapped[str] = mapped_column(String(140), primary_key=True)
    public_id: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    canonical_name: Mapped[str] = mapped_column(String(240))
    city: Mapped[str] = mapped_column(String(120))
    country: Mapped[str] = mapped_column(String(120))
    timezone: Mapped[str] = mapped_column(String(80))
    next_event_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    distances: Mapped[list[str]] = mapped_column(JSON, default=list)
    registration_status: Mapped[str] = mapped_column(String(40), default="unknown")
    status: Mapped[str] = mapped_column(String(40), default="monitoring")
    recurrence: Mapped[str] = mapped_column(String(40), default="annual")
    current_edition_id: Mapped[int | None] = mapped_column(
        ForeignKey("event_editions.id", use_alter=True),
        nullable=True,
    )
    official_url: Mapped[str] = mapped_column(Text)
    registration_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    creation_source: Mapped[str] = mapped_column(String(40), default="seed_collection")
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )

    legacy_ids: Mapped[list[EventLegacyId]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    search_keywords: Mapped[list[EventSearchKeyword]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    regions: Mapped[list[EventRegion]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    collections: Mapped[list[EventCollectionMember]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    sources: Mapped[list[EventSource]] = relationship(
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    editions: Mapped[list[EventEdition]] = relationship(
        foreign_keys="EventEdition.event_id",
        back_populates="event",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    registration_windows: Mapped[list[RegistrationWindow]] = relationship(lazy="selectin")
    current_edition: Mapped[EventEdition | None] = relationship(
        foreign_keys=[current_edition_id],
        post_update=True,
        lazy="selectin",
    )


class Region(Base):
    __tablename__ = "regions"

    tag: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    scope: Mapped[str] = mapped_column(String(40))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class EventRegion(Base):
    __tablename__ = "event_regions"

    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), primary_key=True)
    region_tag: Mapped[str] = mapped_column(ForeignKey("regions.tag"), primary_key=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)

    event: Mapped[Event] = relationship(back_populates="regions")
    region: Mapped[Region] = relationship(lazy="selectin")


class EventSearchKeyword(Base):
    __tablename__ = "event_search_keywords"
    __table_args__ = (UniqueConstraint("event_id", "keyword"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    keyword: Mapped[str] = mapped_column(String(240))
    keyword_type: Mapped[str] = mapped_column(String(40), default="common_name")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    event: Mapped[Event] = relationship(back_populates="search_keywords")


class EventLegacyId(Base):
    __tablename__ = "event_legacy_ids"
    __table_args__ = (UniqueConstraint("event_id", "legacy_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    legacy_id: Mapped[str] = mapped_column(String(120))
    reason: Mapped[str] = mapped_column(String(80), default="slug")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    event: Mapped[Event] = relationship(back_populates="legacy_ids")


class EventCollection(Base):
    __tablename__ = "event_collections"

    slug: Mapped[str] = mapped_column(String(80), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class EventCollectionMember(Base):
    __tablename__ = "event_collection_members"

    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), primary_key=True)
    collection_slug: Mapped[str] = mapped_column(
        ForeignKey("event_collections.slug"),
        primary_key=True,
    )
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    event: Mapped[Event] = relationship(back_populates="collections")
    collection: Mapped[EventCollection] = relationship(lazy="selectin")


class EventSource(Base):
    __tablename__ = "event_sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    url: Mapped[str] = mapped_column(Text)
    source_type: Mapped[str] = mapped_column(String(40))
    priority: Mapped[int] = mapped_column(Integer, default=100)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_checked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    event: Mapped[Event] = relationship(back_populates="sources")


class EventEdition(Base):
    __tablename__ = "event_editions"
    __table_args__ = (UniqueConstraint("event_id", "edition_label"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    edition_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edition_label: Mapped[str] = mapped_column(String(40))
    event_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="unknown")
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )

    event: Mapped[Event] = relationship(
        foreign_keys=[event_id],
        back_populates="editions",
    )
    registration_windows: Mapped[list[RegistrationWindow]] = relationship(
        back_populates="event_edition",
        cascade="all, delete-orphan",
    )


class RegistrationWindow(Base):
    __tablename__ = "registration_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    event_edition_id: Mapped[int | None] = mapped_column(ForeignKey("event_editions.id"))
    registration_open_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    registration_open_precision: Mapped[str] = mapped_column(String(40), default="unknown")
    registration_close_at: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="unknown")
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    event_edition: Mapped[EventEdition | None] = relationship(
        back_populates="registration_windows",
    )


class ProposedEventUpdate(Base):
    __tablename__ = "proposed_event_updates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"), index=True)
    update_type: Mapped[str] = mapped_column(String(40))
    current_fields: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    proposed_fields: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    evidence: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    change_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    reviewed_by_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PromptVersion(Base):
    __tablename__ = "prompt_versions"
    __table_args__ = (UniqueConstraint("prompt_key", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_key: Mapped[str] = mapped_column(String(120), index=True)
    version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), default="draft", index=True)
    created_by: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class EventSuggestion(Base):
    __tablename__ = "event_suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    event_name: Mapped[str] = mapped_column(String(240))
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    location: Mapped[str | None] = mapped_column(String(240), nullable=True)
    region_tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    distances: Mapped[list[str]] = mapped_column(JSON, default=list)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitter_user_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    submitter_username: Mapped[str | None] = mapped_column(String(80), nullable=True)
    submitter_display_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
    )
