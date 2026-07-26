import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlmodel import Field, SQLModel

SCHEMA = "mediaarchivist"
EMBEDDING_DIMENSIONS = 1024


def utc_now() -> datetime:
    return datetime.now(UTC)


class ChatType(StrEnum):
    GROUP = "group"
    SUPERGROUP = "supergroup"


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    ANIMATION = "animation"


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class Chat(SQLModel, table=True):
    __tablename__ = "chats"
    __table_args__ = {"schema": SCHEMA}

    chat_id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    type: str = Field(sa_column=Column(String(32), nullable=False))
    title: str | None = Field(default=None, sa_column=Column(Text))
    active: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, server_default=text("false")),
    )
    bot_status: str | None = Field(default=None, sa_column=Column(String(32)))
    join_date: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Message(SQLModel, table=True):
    __tablename__ = "messages"
    __table_args__ = (
        UniqueConstraint("chat_id", "message_id", name="uq_messages_chat_message"),
        Index("ix_messages_chat_id", "chat_id"),
        {"schema": SCHEMA},
    )

    message_uuid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(UUID(as_uuid=True), primary_key=True),
    )
    chat_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey(f"{SCHEMA}.chats.chat_id", ondelete="CASCADE"),
            nullable=False,
        )
    )
    sender_id: int | None = Field(default=None, sa_column=Column(BigInteger))
    message_id: int = Field(sa_column=Column(BigInteger, nullable=False))
    caption: str | None = Field(default=None, sa_column=Column(Text))
    media_group_id: str | None = Field(default=None, sa_column=Column(String(255)))
    add_date: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )


class Media(SQLModel, table=True):
    __tablename__ = "media"
    __table_args__ = (
        UniqueConstraint("message_uuid", name="uq_media_message"),
        Index("ix_media_status", "status"),
        Index("ix_media_file_unique_id", "file_unique_id"),
        Index("ix_media_embedding_model", "embedding_model"),
        Index(
            "ix_media_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
        Index(
            "ix_media_search_text_trgm",
            "search_text",
            postgresql_using="gin",
            postgresql_ops={"search_text": "gin_trgm_ops"},
        ),
        Index(
            "ix_media_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_where=text("embedding IS NOT NULL"),
        ),
        {"schema": SCHEMA},
    )

    media_uuid: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        sa_column=Column(
            UUID(as_uuid=True),
            primary_key=True,
            server_default=text("gen_random_uuid()"),
        ),
    )
    message_uuid: uuid.UUID = Field(
        sa_column=Column(
            UUID(as_uuid=True),
            ForeignKey(f"{SCHEMA}.messages.message_uuid", ondelete="CASCADE"),
            nullable=False,
        )
    )
    file_id: str = Field(sa_column=Column(Text, nullable=False))
    file_unique_id: str | None = Field(default=None, sa_column=Column(Text))
    media_type: str = Field(sa_column=Column(String(16), nullable=False))
    mime_type: str | None = Field(default=None, sa_column=Column(String(255)))
    file_size: int | None = Field(default=None, sa_column=Column(BigInteger))
    duration: int | None = Field(default=None)
    description: dict[str, Any] | None = Field(
        default=None,
        sa_column=Column(JSONB),
    )
    transcript: str | None = Field(default=None, sa_column=Column(Text))
    search_text: str | None = Field(default=None, sa_column=Column(Text))
    search_vector: Any | None = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('simple', coalesce(search_text, ''))", persisted=True),
        ),
    )
    embedding: Any | None = Field(
        default=None,
        sa_column=Column(Vector(EMBEDDING_DIMENSIONS)),
    )
    status: str = Field(
        default=ProcessingStatus.PENDING,
        sa_column=Column(
            String(32),
            nullable=False,
            server_default=text("'pending'"),
        ),
    )
    error: str | None = Field(default=None, sa_column=Column(Text))
    model_version: str | None = Field(default=None, sa_column=Column(String(255)))
    embedding_model: str | None = Field(default=None, sa_column=Column(String(255)))
    prompt_version: str | None = Field(default=None, sa_column=Column(String(64)))
    add_date: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )
