"""Testes da Task 2: Database Schema e Migrações.

Requerem o banco PostgreSQL rodando (docker compose up -d).
Marcados com @pytest.mark.integration para execução separada.
"""
import os

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://fiscusc:fiscusc@localhost:5432/fiscusc")


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(DATABASE_URL)
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def conn(engine):
    with engine.connect() as c:
        yield c


@pytest.mark.integration
def test_documents_table_exists(conn):
    """Tabela documents deve existir."""
    result = conn.execute(text("SELECT 1 FROM documents LIMIT 1"))
    assert result is not None


@pytest.mark.integration
def test_document_chunks_table_exists(conn):
    """Tabela document_chunks deve existir."""
    result = conn.execute(text("SELECT 1 FROM document_chunks LIMIT 1"))
    assert result is not None


@pytest.mark.integration
def test_faturas_table_exists(conn):
    """Tabela faturas deve existir."""
    result = conn.execute(text("SELECT 1 FROM faturas LIMIT 1"))
    assert result is not None


@pytest.mark.integration
def test_fatura_itens_table_exists(conn):
    """Tabela fatura_itens deve existir."""
    result = conn.execute(text("SELECT 1 FROM fatura_itens LIMIT 1"))
    assert result is not None


@pytest.mark.integration
def test_chunks_have_vector_column(conn):
    """Coluna embedding deve ser do tipo vector(1024)."""
    result = conn.execute(
        text("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_name = 'document_chunks'
            AND column_name = 'embedding'
        """)
    )
    row = result.fetchone()
    assert row is not None
    assert "vector" in str(row[0]).lower() or "user-defined" in str(row[0]).lower()


@pytest.mark.integration
def test_hnsw_index_exists(conn):
    """Índice HNSW deve existir para busca vetorial."""
    result = conn.execute(
        text("""
            SELECT indexname FROM pg_indexes
            WHERE indexname = 'idx_chunks_embedding'
        """)
    )
    row = result.fetchone()
    assert row is not None, "Índice HNSW idx_chunks_embedding não encontrado"


@pytest.mark.integration
def test_search_similar_chunks_function_exists(conn):
    """Função search_similar_chunks deve existir."""
    result = conn.execute(
        text("""
            SELECT routine_name FROM information_schema.routines
            WHERE routine_name = 'search_similar_chunks'
            AND routine_type = 'FUNCTION'
        """)
    )
    row = result.fetchone()
    assert row is not None, "Função search_similar_chunks não encontrada"


@pytest.mark.integration
def test_views_exist(conn):
    """Views v_chunks_with_document e v_despesas_por_categoria devem existir."""
    for view in ["v_chunks_with_document", "v_despesas_por_categoria"]:
        result = conn.execute(
            text(f"SELECT table_name FROM information_schema.views WHERE table_name = '{view}'")
        )
        row = result.fetchone()
        assert row is not None, f"View {view} não encontrada"


@pytest.mark.integration
def test_sha256_unique_constraint(conn):
    """Campo sha256 deve ter constraint unique."""
    result = conn.execute(
        text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = 'documents'
            AND constraint_type = 'UNIQUE'
        """)
    )
    rows = result.fetchall()
    assert len(rows) > 0, "Nenhuma constraint UNIQUE em documents"


@pytest.mark.integration
def test_alembic_version_table_exists(conn):
    """Tabela alembic_version deve existir após migration."""
    result = conn.execute(text("SELECT version_num FROM alembic_version"))
    row = result.fetchone()
    assert row is not None
    assert row[0] == "001"
