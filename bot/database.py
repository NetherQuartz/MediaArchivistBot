import uuid
import os

from enum import Enum
from datetime import datetime
from typing import Iterator
from contextlib import contextmanager

from sqlmodel import SQLModel, Field, Column, Session, create_engine, text
from sqlalchemy.types import BigInteger, Text
from pgvector.sqlalchemy import Vector


class MediaType(Enum):
    image = "image"
    video = "video"


class ChatType(Enum):
    private = "private"
    group = "group"
    supergroup = "supergroup"
    channel = "channel"


class Chat(SQLModel, table=True):
    __tablename__: str = "chats"

    chat_id: int = Field(primary_key=True, sa_type=BigInteger)
    type: ChatType
    join_date: datetime = Field(default_factory=datetime.now)


class User(SQLModel, table=True):
    __tablename__: str = "users"

    user_id: int = Field(primary_key=True, default=None, sa_type=BigInteger)
    chat_id: int = Field(foreign_key="chats.chat_id", sa_type=BigInteger)
    join_date: datetime = Field(default_factory=datetime.now)


class Message(SQLModel, table=True):
    __tablename__: str = "messages"

    message_uuid: uuid.UUID = Field(primary_key=True, default_factory=uuid.uuid4)
    chat_id: int = Field(foreign_key="chats.chat_id", sa_type=BigInteger)
    sender_id: int = Field(sa_type=BigInteger)
    message_id: int = Field(sa_type=BigInteger)


class File(SQLModel, table=True):
    __tablename__: str = "files"

    file_id: str = Field(primary_key=True)
    message_uuid: uuid.UUID = Field(foreign_key="messages.message_uuid")
    media_type: MediaType
    description: str | None = Field(sa_type=Text)
    embedding: Vector | None = Field(sa_column=Column(Vector(1024), nullable=True))

    class Config:
        arbitrary_types_allowed = True


@contextmanager
def get_db() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


database = os.getenv("POSTGRES_DB")
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")

engine = create_engine(f"postgresql+psycopg2://{user}:{password}@pgvector-database:5432/{database}")

with get_db() as session:
    session.exec(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.commit()

SQLModel.metadata.create_all(engine)
