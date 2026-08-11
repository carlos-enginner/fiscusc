"""Validação de dimensões de embeddings entre provider e banco de dados."""
from sqlalchemy import Engine, text

from app.embeddings.service import EmbeddingsProvider


class EmbeddingDimensionMismatchError(Exception):
    """Erro quando dimensão do provider difere da dimensão no banco."""

    def __init__(self, provider_dim: int, db_dim: int):
        self.provider_dim = provider_dim
        self.db_dim = db_dim
        message = (
            f"Mismatch de dimensão de embeddings!\n"
            f"  - Provider: {provider_dim} dimensões\n"
            f"  - Banco de dados: {db_dim} dimensões\n\n"
            f"Para corrigir, execute as migrations para recriar a coluna:\n"
            f"  alembic downgrade base\n"
            f"  alembic upgrade head\n\n"
            f"ATENÇÃO: Isso apagará todos os embeddings existentes. "
            f"Será necessário re-ingerir os documentos."
        )
        super().__init__(message)


def get_database_embedding_dimension(engine: Engine) -> int | None:
    """
    Consulta a dimensão da coluna de embeddings no banco de dados.

    Args:
        engine: SQLAlchemy engine conectado ao banco.

    Returns:
        Dimensão da coluna vector, ou None se tabela/coluna não existe.
    """
    # Query para extrair dimensão do tipo vector(N) do PostgreSQL
    query = text("""
        SELECT 
            CASE 
                WHEN udt_name = 'vector' THEN 
                    CAST(
                        SUBSTRING(
                            format_type(a.atttypid, a.atttypmod) 
                            FROM '\\(([0-9]+)\\)'
                        ) AS INTEGER
                    )
                ELSE NULL
            END as dimension
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        JOIN information_schema.columns ic 
            ON ic.table_name = c.relname 
            AND ic.column_name = a.attname
            AND ic.table_schema = n.nspname
        WHERE c.relname = 'document_chunks'
        AND a.attname = 'embedding'
        AND n.nspname = 'public'
        AND a.attnum > 0
        AND NOT a.attisdropped
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        row = result.fetchone()
        if row and row[0] is not None:
            return int(row[0])
        return None


def validate_embedding_dimensions(
    provider: EmbeddingsProvider, engine: Engine
) -> None:
    """
    Valida que a dimensão do provider corresponde à do banco.

    Args:
        provider: Provider de embeddings a validar.
        engine: SQLAlchemy engine conectado ao banco.

    Raises:
        EmbeddingDimensionMismatchError: Se as dimensões não correspondem.

    Note:
        Se a tabela não existe ou não tem a coluna, não levanta erro
        (assume que a migration criará com a dimensão correta).
    """
    db_dim = get_database_embedding_dimension(engine)

    # Se banco não tem a coluna ainda, nada a validar
    if db_dim is None:
        return

    provider_dim = provider.dimensions

    if provider_dim != db_dim:
        raise EmbeddingDimensionMismatchError(provider_dim, db_dim)
