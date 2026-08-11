"""Update embedding dimension

Revision ID: 003
Revises: 002
Create Date: 2026-08-10

IMPORTANTE: Após executar esta migration, é OBRIGATÓRIO re-ingerir todos os
documentos para recalcular os embeddings com a nova dimensão. Os embeddings
existentes serão invalidados (coluna será NULLificada implicitamente pela
mudança de tipo).

Uso:
    # Para alterar para 768 dimensões:
    TARGET_EMBEDDING_DIMS=768 alembic upgrade head
    
    # Para reverter para 1024:
    alembic downgrade -1

"""
import os
from typing import Sequence, Union

from alembic import op

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Dimensão padrão caso não especificada
DEFAULT_DIMENSION = 1024
# Dimensão anterior (para downgrade)
PREVIOUS_DIMENSION = 1024


def get_target_dimension() -> int:
    """Obtém a dimensão alvo da variável de ambiente TARGET_EMBEDDING_DIMS."""
    dim_str = os.environ.get("TARGET_EMBEDDING_DIMS", str(DEFAULT_DIMENSION))
    try:
        dim = int(dim_str)
        if dim <= 0:
            raise ValueError(f"Dimensão deve ser positiva, recebido: {dim}")
        return dim
    except ValueError as e:
        raise ValueError(
            f"TARGET_EMBEDDING_DIMS deve ser um inteiro positivo, recebido: '{dim_str}'"
        ) from e


def upgrade() -> None:
    target_dim = get_target_dimension()
    
    # 1. Dropar índice HNSW (depende da coluna embedding)
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    
    # 2. Dropar view que depende da coluna embedding
    op.execute("DROP VIEW IF EXISTS v_chunks_with_document")
    
    # 3. Dropar function que depende do tipo vector com dimensão específica
    op.execute("DROP FUNCTION IF EXISTS search_similar_chunks")
    
    # 4. Alterar coluna embedding para nova dimensão
    # NULL é usado para converter os dados existentes (embeddings antigos são inválidos)
    op.execute(
        f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({target_dim}) USING NULL"
    )
    
    # 5. Recriar índice HNSW com nova dimensão
    op.execute(
        f"""
        CREATE INDEX idx_chunks_embedding ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    
    # 6. Recriar view v_chunks_with_document
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
    
    # 7. Recriar function search_similar_chunks com nova dimensão
    op.execute(f"""
        CREATE OR REPLACE FUNCTION search_similar_chunks(
            query_embedding vector({target_dim}),
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
    # Reverter para dimensão anterior (1024)
    
    # 1. Dropar índice HNSW
    op.execute("DROP INDEX IF EXISTS idx_chunks_embedding")
    
    # 2. Dropar view
    op.execute("DROP VIEW IF EXISTS v_chunks_with_document")
    
    # 3. Dropar function
    op.execute("DROP FUNCTION IF EXISTS search_similar_chunks")
    
    # 4. Reverter coluna embedding para dimensão anterior
    op.execute(
        f"ALTER TABLE document_chunks ALTER COLUMN embedding TYPE vector({PREVIOUS_DIMENSION}) USING NULL"
    )
    
    # 5. Recriar índice HNSW
    op.execute(
        f"""
        CREATE INDEX idx_chunks_embedding ON document_chunks
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )
    
    # 6. Recriar view
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
    
    # 7. Recriar function com dimensão anterior
    op.execute(f"""
        CREATE OR REPLACE FUNCTION search_similar_chunks(
            query_embedding vector({PREVIOUS_DIMENSION}),
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
