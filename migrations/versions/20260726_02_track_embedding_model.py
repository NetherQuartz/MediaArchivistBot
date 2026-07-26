"""Track the embedding model used for each vector.

Revision ID: 20260726_02
Revises: 20260726_01
Create Date: 2026-07-26
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260726_02"
down_revision: str | Sequence[str] | None = "20260726_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "mediaarchivist"


def upgrade() -> None:
    op.add_column(
        "media",
        sa.Column("embedding_model", sa.String(255)),
        schema=SCHEMA,
    )
    op.execute(
        f"""
        UPDATE {SCHEMA}.media
        SET embedding_model = 'snowflake-arctic-embed2'
        WHERE embedding IS NOT NULL
          AND prompt_version = 'legacy-v1'
        """
    )
    op.create_index(
        "ix_media_embedding_model",
        "media",
        ["embedding_model"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_media_embedding_model",
        table_name="media",
        schema=SCHEMA,
    )
    op.drop_column("media", "embedding_model", schema=SCHEMA)
