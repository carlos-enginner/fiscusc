"""Initial schema

Revision ID: 001
Revises: 
Create Date: 2026-08-09

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # --- documents ---
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("filename", sa.VARCHAR(255), nullable=False),
        sa.Column("document_type", sa.VARCHAR(50), nullable=False),
        sa.Column("version", sa.VARCHAR(50), nullable=True),
        sa.Column("sha256", sa.VARCHAR(64), nullable=False, unique=True),
        sa.Column("page_count", sa.Integer, nullable=True),
        sa.Column("file_size_bytes", sa.Integer, nullable=True),
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_documents_type", "documents", ["document_type"])
    op.create_index("idx_documents_sha256", "documents", ["sha256"])
    op.create_index("idx_documents_created", "documents", ["created_at"])

    # --- document_chunks ---
    op.create_table(
        "document_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("page", sa.Integer, nullable=False),
        sa.Column("section", sa.VARCHAR(255), nullable=True),
        sa.Column("chapter", sa.VARCHAR(255), nullable=True),
        sa.Column("article", sa.VARCHAR(100), nullable=True),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("content", sa.TEXT, nullable=False),
        sa.Column("content_length", sa.Integer, nullable=False),
        sa.Column("embedding", sa.Text, nullable=True),  # placeholder; raw SQL below
        sa.Column("metadata", postgresql.JSONB, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    # Alter column to real vector type after table creation
    op.execute("ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector(1024) USING NULL")
    op.create_index("idx_chunks_document", "document_chunks", ["document_id"])
    op.create_index("idx_chunks_page", "document_chunks", ["page"])
    op.create_index("idx_chunks_section", "document_chunks", ["section"])
    # HNSW index for vector similarity search
    op.execute(
        """
        CREATE INDEX idx_chunks_embedding ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    # --- faturas ---
    op.create_table(
        "faturas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("condominio_nome", sa.VARCHAR(255), nullable=True),
        sa.Column("condominio_cnpj", sa.VARCHAR(20), nullable=True),
        sa.Column("unidade_bloco", sa.VARCHAR(10), nullable=True),
        sa.Column("unidade_apartamento", sa.VARCHAR(10), nullable=True),
        sa.Column("proprietario", sa.VARCHAR(255), nullable=True),
        sa.Column("mes_referencia", sa.VARCHAR(20), nullable=True),
        sa.Column("data_vencimento", sa.Date, nullable=True),
        sa.Column("data_emissao", sa.Date, nullable=True),
        sa.Column("total_cobranca", sa.DECIMAL(12, 2), nullable=True),
        sa.Column("codigo_barras", sa.VARCHAR(100), nullable=True),
        sa.Column("modelo_provider", sa.VARCHAR(50), nullable=True),
        sa.Column("modelo_nome", sa.VARCHAR(100), nullable=True),
        sa.Column("tempo_extracao_seg", sa.DECIMAL(8, 2), nullable=True),
        sa.Column("health_check_score", sa.Integer, nullable=True),
        sa.Column("raw_data", postgresql.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_faturas_unidade", "faturas", ["unidade_bloco", "unidade_apartamento"])
    op.create_index("idx_faturas_vencimento", "faturas", ["data_vencimento"])
    op.create_index("idx_faturas_referencia", "faturas", ["mes_referencia"])

    # --- fatura_itens ---
    op.create_table(
        "fatura_itens",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("fatura_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("faturas.id", ondelete="CASCADE"), nullable=False),
        sa.Column("secao", sa.VARCHAR(50), nullable=False),
        sa.Column("grupo", sa.VARCHAR(100), nullable=True),
        sa.Column("descricao", sa.VARCHAR(255), nullable=False),
        sa.Column("valor", sa.DECIMAL(12, 2), nullable=False),
        sa.Column("categoria", sa.VARCHAR(50), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_fatura_itens_fatura", "fatura_itens", ["fatura_id"])
    op.create_index("idx_fatura_itens_secao", "fatura_itens", ["secao"])
    op.create_index("idx_fatura_itens_categoria", "fatura_itens", ["categoria"])

    # --- query_logs ---
    op.create_table(
        "query_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("uuid_generate_v4()")),
        sa.Column("query", sa.TEXT, nullable=False),
        sa.Column("agents_used", postgresql.ARRAY(sa.VARCHAR(50)), nullable=True),
        sa.Column("response", sa.TEXT, nullable=True),
        sa.Column("sources", postgresql.JSONB, nullable=True),
        sa.Column("latency_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_query_logs_created", "query_logs", ["created_at"])

    # --- Views ---
    op.execute("""
        CREATE VIEW v_chunks_with_document AS
        SELECT
            c.id,
            c.content,
            c.page,
            c.section,
            c.chapter,
            c.article,
            c.embedding,
            d.filename,
            d.document_type,
            d.version
        FROM document_chunks c
        JOIN documents d ON c.document_id = d.id
    """)

    op.execute("""
        CREATE VIEW v_despesas_por_categoria AS
        SELECT
            f.mes_referencia,
            fi.categoria,
            SUM(fi.valor) as total,
            COUNT(*) as qtd_itens
        FROM fatura_itens fi
        JOIN faturas f ON fi.fatura_id = f.id
        WHERE fi.secao = 'despesas'
        GROUP BY f.mes_referencia, fi.categoria
        ORDER BY f.mes_referencia, total DESC
    """)

    # --- Function: search_similar_chunks ---
    op.execute("""
        CREATE OR REPLACE FUNCTION search_similar_chunks(
            query_embedding vector(1024),
            match_count INT DEFAULT 5,
            filter_document_type VARCHAR DEFAULT NULL
        )
        RETURNS TABLE (
            id UUID,
            content TEXT,
            page INTEGER,
            section VARCHAR,
            document_type VARCHAR,
            filename VARCHAR,
            similarity FLOAT
        )
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RETURN QUERY
            SELECT
                c.id,
                c.content,
                c.page,
                c.section,
                d.document_type,
                d.filename,
                1 - (c.embedding <=> query_embedding) as similarity
            FROM document_chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE (filter_document_type IS NULL OR d.document_type = filter_document_type)
            ORDER BY c.embedding <=> query_embedding
            LIMIT match_count;
        END;
        $$
    """)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS search_similar_chunks")
    op.execute("DROP VIEW IF EXISTS v_despesas_por_categoria")
    op.execute("DROP VIEW IF EXISTS v_chunks_with_document")
    op.drop_table("query_logs")
    op.drop_table("fatura_itens")
    op.drop_table("faturas")
    op.drop_table("document_chunks")
    op.drop_table("documents")
