"""Add content_hash to document_chunks

Revision ID: 002
Revises: 001
Create Date: 2026-08-10

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add content_hash column (nullable to allow backfill)
    op.add_column(
        "document_chunks",
        sa.Column("content_hash", sa.VARCHAR(64), nullable=True),
    )

    # Create index for content_hash lookups
    op.create_index("idx_chunks_content_hash", "document_chunks", ["content_hash"])

    # Backfill existing rows with SHA256 hash of content
    op.execute(
        """
        UPDATE document_chunks
        SET content_hash = encode(sha256(content::bytea), 'hex')
        WHERE content_hash IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index("idx_chunks_content_hash", table_name="document_chunks")
    op.drop_column("document_chunks", "content_hash")
