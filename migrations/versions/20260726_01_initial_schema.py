"""Create the managed media archive schema.

This initial revision also upgrades the pre-Alembic chats/messages/files layout
in place when those tables already exist.

Revision ID: 20260726_01
Revises:
Create Date: 2026-07-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20260726_01"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "mediaarchivist"


def _table_exists(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names(schema=SCHEMA)


def _columns(name: str) -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(name, schema=SCHEMA)
    }


def _unique_constraints(name: str) -> set[str]:
    return {
        constraint["name"]
        for constraint in sa.inspect(op.get_bind()).get_unique_constraints(
            name,
            schema=SCHEMA,
        )
        if constraint["name"]
    }


def _create_chats() -> None:
    op.create_table(
        "chats",
        sa.Column("chat_id", sa.BigInteger(), primary_key=True),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("title", sa.Text()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("bot_status", sa.String(32)),
        sa.Column(
            "join_date",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        schema=SCHEMA,
    )


def _upgrade_chats() -> None:
    columns = _columns("chats")
    if "title" not in columns:
        op.add_column("chats", sa.Column("title", sa.Text()), schema=SCHEMA)
    if "active" not in columns:
        op.add_column(
            "chats",
            sa.Column(
                "active",
                sa.Boolean(),
                nullable=False,
                server_default=sa.text("false"),
            ),
            schema=SCHEMA,
        )
    if "bot_status" not in columns:
        op.add_column(
            "chats",
            sa.Column("bot_status", sa.String(32)),
            schema=SCHEMA,
        )
    if "updated_at" not in columns:
        op.add_column(
            "chats",
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            schema=SCHEMA,
        )
    op.execute(
        f"""
        ALTER TABLE {SCHEMA}.chats
        ALTER COLUMN type TYPE varchar(32) USING type::text,
        ALTER COLUMN join_date TYPE timestamptz
            USING join_date AT TIME ZONE 'UTC'
        """
    )


def _create_messages() -> None:
    op.create_table(
        "messages",
        sa.Column(
            "message_uuid",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "chat_id",
            sa.BigInteger(),
            sa.ForeignKey(f"{SCHEMA}.chats.chat_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sender_id", sa.BigInteger()),
        sa.Column("message_id", sa.BigInteger(), nullable=False),
        sa.Column("caption", sa.Text()),
        sa.Column("media_group_id", sa.String(255)),
        sa.Column(
            "add_date",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "chat_id",
            "message_id",
            name="uq_messages_chat_message",
        ),
        schema=SCHEMA,
    )


def _upgrade_messages() -> None:
    columns = _columns("messages")
    if "caption" not in columns:
        op.add_column("messages", sa.Column("caption", sa.Text()), schema=SCHEMA)
    if "media_group_id" not in columns:
        op.add_column(
            "messages",
            sa.Column("media_group_id", sa.String(255)),
            schema=SCHEMA,
        )
    op.execute(
        f"""
        ALTER TABLE {SCHEMA}.messages
        ALTER COLUMN sender_id DROP NOT NULL,
        ALTER COLUMN add_date TYPE timestamptz
            USING add_date AT TIME ZONE 'UTC'
        """
    )
    if _table_exists("files"):
        op.execute(
            f"""
            WITH ranked AS (
                SELECT
                    message_uuid,
                    first_value(message_uuid) OVER (
                        PARTITION BY chat_id, message_id
                        ORDER BY add_date, message_uuid
                    ) AS keep_uuid
                FROM {SCHEMA}.messages
            )
            UPDATE {SCHEMA}.files AS files
            SET message_uuid = ranked.keep_uuid
            FROM ranked
            WHERE files.message_uuid = ranked.message_uuid
              AND ranked.message_uuid <> ranked.keep_uuid
            """
        )
    op.execute(
        f"""
        WITH ranked AS (
            SELECT
                message_uuid,
                row_number() OVER (
                    PARTITION BY chat_id, message_id
                    ORDER BY add_date, message_uuid
                ) AS row_number
            FROM {SCHEMA}.messages
        )
        DELETE FROM {SCHEMA}.messages AS messages
        USING ranked
        WHERE messages.message_uuid = ranked.message_uuid
          AND ranked.row_number > 1
        """
    )
    if "uq_messages_chat_message" not in _unique_constraints("messages"):
        op.create_unique_constraint(
            "uq_messages_chat_message",
            "messages",
            ["chat_id", "message_id"],
            schema=SCHEMA,
        )


def _create_media() -> None:
    op.create_table(
        "media",
        sa.Column(
            "media_uuid",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "message_uuid",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                f"{SCHEMA}.messages.message_uuid",
                ondelete="CASCADE",
            ),
            nullable=False,
        ),
        sa.Column("file_id", sa.Text(), nullable=False),
        sa.Column("file_unique_id", sa.Text()),
        sa.Column("media_type", sa.String(16), nullable=False),
        sa.Column("mime_type", sa.String(255)),
        sa.Column("file_size", sa.BigInteger()),
        sa.Column("duration", sa.Integer()),
        sa.Column("description", postgresql.JSONB()),
        sa.Column("transcript", sa.Text()),
        sa.Column("search_text", sa.Text()),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "to_tsvector('simple', coalesce(search_text, ''))",
                persisted=True,
            ),
        ),
        sa.Column("embedding", Vector(1024)),
        sa.Column(
            "status",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("error", sa.Text()),
        sa.Column("model_version", sa.String(255)),
        sa.Column("prompt_version", sa.String(64)),
        sa.Column(
            "add_date",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("message_uuid", name="uq_media_message"),
        schema=SCHEMA,
    )


def _copy_legacy_files() -> None:
    if not _table_exists("files"):
        return
    op.execute(
        f"""
        INSERT INTO {SCHEMA}.media (
            message_uuid,
            file_id,
            media_type,
            description,
            search_text,
            embedding,
            status,
            model_version,
            prompt_version,
            add_date,
            updated_at
        )
        SELECT
            message_uuid,
            file_id,
            media_type::text,
            CASE
                WHEN description IS NULL THEN NULL
                ELSE jsonb_build_object('summary', description)
            END,
            description,
            embedding,
            CASE WHEN embedding IS NULL THEN 'pending' ELSE 'ready' END,
            'pixtral-12b-2409',
            'legacy-v1',
            add_date AT TIME ZONE 'UTC',
            now()
        FROM {SCHEMA}.files
        ON CONFLICT (message_uuid) DO NOTHING
        """
    )
    op.drop_table("files", schema=SCHEMA)


def _create_indexes() -> None:
    op.create_index(
        "ix_messages_chat_id",
        "messages",
        ["chat_id"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_media_status",
        "media",
        ["status"],
        schema=SCHEMA,
    )
    op.create_index(
        "ix_media_file_unique_id",
        "media",
        ["file_unique_id"],
        schema=SCHEMA,
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_media_search_vector
        ON {SCHEMA}.media USING gin (search_vector)
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_media_search_text_trgm
        ON {SCHEMA}.media USING gin (search_text gin_trgm_ops)
        """
    )
    op.execute(
        f"""
        CREATE INDEX IF NOT EXISTS ix_media_embedding_hnsw
        ON {SCHEMA}.media USING hnsw (embedding vector_cosine_ops)
        WHERE embedding IS NOT NULL
        """
    )


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")

    if _table_exists("chats"):
        _upgrade_chats()
    else:
        _create_chats()

    if _table_exists("messages"):
        _upgrade_messages()
    else:
        _create_messages()

    if not _table_exists("media"):
        _create_media()
    _copy_legacy_files()

    if _table_exists("users"):
        op.drop_table("users", schema=SCHEMA)

    _create_indexes()


def downgrade() -> None:
    op.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
