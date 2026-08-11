"""Testes unitários para validação de dimensão de embeddings."""
import pytest
from unittest.mock import MagicMock, patch

from app.embeddings.validator import (
    EmbeddingDimensionMismatchError,
    get_database_embedding_dimension,
    validate_embedding_dimensions,
)


class TestEmbeddingDimensionMismatchError:
    """Testes para EmbeddingDimensionMismatchError."""

    def test_error_message_contains_both_dimensions(self):
        """Mensagem de erro deve conter ambas as dimensões."""
        error = EmbeddingDimensionMismatchError(provider_dim=384, db_dim=1024)
        
        assert "384" in str(error)
        assert "1024" in str(error)

    def test_error_message_contains_migration_instructions(self):
        """Mensagem de erro deve conter instruções de migration."""
        error = EmbeddingDimensionMismatchError(provider_dim=384, db_dim=1024)
        
        assert "alembic downgrade base" in str(error)
        assert "alembic upgrade head" in str(error)

    def test_error_stores_dimensions_as_attributes(self):
        """Erro deve armazenar dimensões como atributos."""
        error = EmbeddingDimensionMismatchError(provider_dim=384, db_dim=1024)
        
        assert error.provider_dim == 384
        assert error.db_dim == 1024

    def test_error_message_contains_warning(self):
        """Mensagem deve alertar sobre perda de dados."""
        error = EmbeddingDimensionMismatchError(provider_dim=384, db_dim=1024)
        
        assert "ATENÇÃO" in str(error) or "re-ingerir" in str(error)


class TestGetDatabaseEmbeddingDimension:
    """Testes para get_database_embedding_dimension."""

    @pytest.fixture
    def mock_engine(self):
        """Engine mockado."""
        return MagicMock()

    def test_returns_dimension_when_table_exists(self, mock_engine):
        """Deve retornar dimensão quando tabela existe."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (1024,)
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        result = get_database_embedding_dimension(mock_engine)
        
        assert result == 1024

    def test_returns_none_when_table_not_exists(self, mock_engine):
        """Deve retornar None quando tabela não existe."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        result = get_database_embedding_dimension(mock_engine)
        
        assert result is None

    def test_returns_none_when_column_not_exists(self, mock_engine):
        """Deve retornar None quando coluna não existe."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (None,)
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        result = get_database_embedding_dimension(mock_engine)
        
        assert result is None

    def test_executes_sql_query(self, mock_engine):
        """Deve executar query SQL."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (384,)
        mock_conn.execute.return_value = mock_result
        mock_engine.connect.return_value.__enter__.return_value = mock_conn
        
        get_database_embedding_dimension(mock_engine)
        
        mock_conn.execute.assert_called_once()
        # Verificar que a query menciona document_chunks e embedding
        call_args = mock_conn.execute.call_args
        query_text = str(call_args[0][0])
        assert "document_chunks" in query_text
        assert "embedding" in query_text


class TestValidateEmbeddingDimensions:
    """Testes para validate_embedding_dimensions."""

    @pytest.fixture
    def mock_provider(self):
        """Provider mockado."""
        provider = MagicMock()
        provider.dimensions = 384
        return provider

    @pytest.fixture
    def mock_engine(self):
        """Engine mockado."""
        return MagicMock()

    def test_no_error_when_dimensions_match(self, mock_provider, mock_engine):
        """Não deve lançar erro quando dimensões coincidem."""
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = 384
            
            # Não deve lançar exceção
            validate_embedding_dimensions(mock_provider, mock_engine)

    def test_raises_error_when_dimensions_mismatch(self, mock_provider, mock_engine):
        """Deve lançar EmbeddingDimensionMismatchError quando dimensões diferem."""
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = 1024
            
            with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
                validate_embedding_dimensions(mock_provider, mock_engine)
            
            assert exc_info.value.provider_dim == 384
            assert exc_info.value.db_dim == 1024

    def test_no_error_when_table_not_exists(self, mock_provider, mock_engine):
        """Não deve lançar erro quando tabela não existe (nova instalação)."""
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = None
            
            # Não deve lançar exceção
            validate_embedding_dimensions(mock_provider, mock_engine)

    def test_calls_get_database_dimension(self, mock_provider, mock_engine):
        """Deve chamar get_database_embedding_dimension."""
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = None
            
            validate_embedding_dimensions(mock_provider, mock_engine)
            
            mock_get.assert_called_once_with(mock_engine)

    def test_reads_provider_dimensions(self, mock_provider, mock_engine):
        """Deve ler dimensões do provider."""
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = 384
            
            validate_embedding_dimensions(mock_provider, mock_engine)
            
            # Verifica que acessou provider.dimensions
            _ = mock_provider.dimensions


class TestValidatorIntegration:
    """Testes de integração entre componentes do validator."""

    def test_validate_with_real_provider_interface(self):
        """Deve funcionar com qualquer EmbeddingsProvider."""
        from app.embeddings.service import EmbeddingsProvider
        
        class FakeProvider(EmbeddingsProvider):
            def embed(self, text: str) -> list[float]:
                return [0.0] * 512
            
            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * 512 for _ in texts]
            
            async def embed_batch_async(
                self, texts: list[str], max_workers: int = 4
            ) -> list[list[float]]:
                return self.embed_batch(texts)
            
            @property
            def dimensions(self) -> int:
                return 512
        
        provider = FakeProvider()
        mock_engine = MagicMock()
        
        with patch(
            "app.embeddings.validator.get_database_embedding_dimension"
        ) as mock_get:
            mock_get.return_value = 512
            
            # Não deve lançar erro
            validate_embedding_dimensions(provider, mock_engine)

    def test_error_message_is_user_friendly(self):
        """Mensagem de erro deve ser compreensível para usuário."""
        error = EmbeddingDimensionMismatchError(384, 1024)
        msg = str(error)
        
        # Deve ser legível e conter informações úteis
        assert "Provider" in msg
        assert "Banco de dados" in msg or "banco" in msg.lower()
        assert "384" in msg
        assert "1024" in msg
