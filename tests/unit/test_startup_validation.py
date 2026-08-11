"""Testes para validação de embeddings no startup da aplicação."""
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient

from app.cli import cli
from app.embeddings.validator import EmbeddingDimensionMismatchError


class TestAPIStartupValidation:
    """Testes para validação de embeddings no startup da API."""

    def test_startup_passes_with_matching_dimensions(self):
        """API deve iniciar quando dimensões do provider correspondem ao banco."""
        with patch("app.api.main.get_settings") as mock_settings, \
             patch("app.api.main.create_embedding_provider") as mock_create, \
             patch("app.api.main.get_engine") as mock_engine, \
             patch("app.api.main.validate_embedding_dimensions") as mock_validate:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            mock_validate.return_value = None  # Sem erro
            
            # Importar app após patches
            from app.api.main import app
            client = TestClient(app)
            
            # API deve estar funcionando
            response = client.get("/")
            assert response.status_code == 200

    def test_startup_fails_with_dimension_mismatch(self):
        """API não deve iniciar quando há mismatch de dimensões."""
        with patch("app.api.main.get_settings") as mock_settings, \
             patch("app.api.main.create_embedding_provider") as mock_create, \
             patch("app.api.main.get_engine") as mock_engine, \
             patch("app.api.main.validate_embedding_dimensions") as mock_validate:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            
            # Simular mismatch
            mock_validate.side_effect = EmbeddingDimensionMismatchError(
                provider_dim=384, db_dim=1024
            )
            
            # Recarregar o módulo para aplicar patches
            import importlib
            import app.api.main as main_module
            
            # Aplicar os patches no contexto certo
            with pytest.raises(RuntimeError, match="mismatch de dimensões"):
                # O TestClient vai executar o lifespan e deve falhar
                with TestClient(main_module.app):
                    pass

    def test_startup_logs_provider_info(self):
        """API deve logar informações do provider no startup."""
        with patch("app.api.main.get_settings") as mock_settings, \
             patch("app.api.main.create_embedding_provider") as mock_create, \
             patch("app.api.main.get_engine") as mock_engine, \
             patch("app.api.main.validate_embedding_dimensions") as mock_validate, \
             patch("app.api.main.logger") as mock_logger:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            mock_validate.return_value = None
            
            from app.api.main import app
            with TestClient(app):
                pass
            
            # Verificar que logger.info foi chamado com info do provider
            calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any("fastembed" in call.lower() for call in calls)

    def test_startup_warns_when_using_ollama(self):
        """API deve emitir warning quando usar Ollama para embeddings."""
        with patch("app.api.main.get_settings") as mock_settings, \
             patch("app.api.main.create_embedding_provider") as mock_create, \
             patch("app.api.main.get_engine") as mock_engine, \
             patch("app.api.main.validate_embedding_dimensions") as mock_validate, \
             patch("app.api.main.logger") as mock_logger:
            
            # Setup mocks para Ollama
            mock_settings.return_value = MagicMock(
                embedding_provider="ollama",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 1024
            mock_create.return_value = mock_provider
            mock_validate.return_value = None
            
            from app.api.main import app
            with TestClient(app):
                pass
            
            # Verificar que logger.warning foi chamado sugerindo FastEmbed
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
            assert any("fastembed" in call.lower() for call in warning_calls)


class TestCLIIngestValidation:
    """Testes para validação de embeddings no comando ingest do CLI."""

    def test_ingest_validates_dimensions_before_processing(self):
        """ingest deve validar dimensões antes de processar documento."""
        runner = CliRunner()
        
        # Criar arquivo temporário
        from pathlib import Path
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"
        pdf_path = fixtures_dir / "reg_interno.pdf"
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            
            # Simular mismatch
            mock_validate.side_effect = EmbeddingDimensionMismatchError(
                provider_dim=384, db_dim=1024
            )
            
            result = runner.invoke(cli, ["ingest", str(pdf_path), "--type", "regimento"])
            
            # Deve falhar com exit code 1
            assert result.exit_code == 1
            # Deve mostrar mensagem de erro
            assert "erro" in result.output.lower() or "mismatch" in result.output.lower()

    def test_ingest_continues_when_dimensions_match(self):
        """ingest deve continuar normalmente quando dimensões correspondem."""
        runner = CliRunner()
        
        from pathlib import Path
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"
        pdf_path = fixtures_dir / "reg_interno.pdf"
        
        mock_result = MagicMock()
        mock_result.document_id = "test-uuid"
        mock_result.pages = 10
        mock_result.chunks_created = 45
        mock_result.sha256 = "abc123" * 11
        mock_result.already_existed = False
        mock_result.filename = "reg_interno.pdf"
        mock_result.document_type = "regimento"
        mock_result.status = "success"
        mock_result.metrics = None
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings, \
             patch("app.cli.get_session_factory") as mock_factory, \
             patch("app.cli.DocumentIngestionService") as mock_svc_cls, \
             patch("app.cli.EmbeddingsService"):
            
            # Setup mocks para validação passar
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            mock_validate.return_value = None  # Sem erro
            
            # Setup mock do serviço de ingestão
            mock_svc = MagicMock()
            mock_svc.ingest.return_value = mock_result
            mock_svc_cls.return_value = mock_svc
            
            result = runner.invoke(cli, ["ingest", str(pdf_path), "--type", "regimento"])
            
            # Deve ter sucesso
            assert result.exit_code == 0


class TestCLIQueryValidation:
    """Testes para validação de embeddings no comando query do CLI."""

    def test_query_validates_dimensions_before_processing(self):
        """query deve validar dimensões antes de processar pergunta."""
        runner = CliRunner()
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                embedding_provider="ollama",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 1024
            mock_create.return_value = mock_provider
            
            # Simular mismatch
            mock_validate.side_effect = EmbeddingDimensionMismatchError(
                provider_dim=1024, db_dim=384
            )
            
            result = runner.invoke(cli, ["query", "Qual o horário para obras?"])
            
            # Deve falhar com exit code 1
            assert result.exit_code == 1
            # Deve mostrar mensagem de erro
            assert "erro" in result.output.lower() or "mismatch" in result.output.lower()

    def test_query_shows_friendly_error_on_mismatch(self):
        """query deve mostrar mensagem amigável ao usuário em caso de mismatch."""
        runner = CliRunner()
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings:
            
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            
            mock_validate.side_effect = EmbeddingDimensionMismatchError(
                provider_dim=384, db_dim=1024
            )
            
            result = runner.invoke(cli, ["query", "Qual o horário para obras?"])
            
            # Deve conter instruções sobre como corrigir
            assert "alembic" in result.output.lower() or "migration" in result.output.lower()


class TestCLIValidationWarnings:
    """Testes para warnings de validação no CLI."""

    def test_ingest_warns_when_using_ollama(self):
        """ingest deve emitir warning quando usar Ollama."""
        runner = CliRunner()
        
        from pathlib import Path
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures"
        pdf_path = fixtures_dir / "reg_interno.pdf"
        
        mock_result = MagicMock()
        mock_result.document_id = "test-uuid"
        mock_result.pages = 10
        mock_result.chunks_created = 45
        mock_result.sha256 = "abc123" * 11
        mock_result.already_existed = False
        mock_result.metrics = None
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings, \
             patch("app.cli.get_session_factory") as mock_factory, \
             patch("app.cli.DocumentIngestionService") as mock_svc_cls, \
             patch("app.cli.EmbeddingsService"):
            
            # Setup para usar Ollama
            mock_settings.return_value = MagicMock(
                embedding_provider="ollama",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 1024
            mock_create.return_value = mock_provider
            mock_validate.return_value = None
            
            mock_svc = MagicMock()
            mock_svc.ingest.return_value = mock_result
            mock_svc_cls.return_value = mock_svc
            
            result = runner.invoke(cli, ["ingest", str(pdf_path), "--type", "regimento"])
            
            # Deve conter warning sobre FastEmbed
            assert "fastembed" in result.output.lower()

    def test_query_logs_provider_info(self):
        """query deve mostrar informações do provider sendo usado."""
        runner = CliRunner()
        
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "final_answer": "Resposta teste",
            "results": [{"source": "docs", "result": "...", "evidence": []}],
            "classifications": [],
        }
        
        with patch("app.embeddings.factory.create_embedding_provider") as mock_create, \
             patch("app.cli.get_engine") as mock_engine, \
             patch("app.embeddings.validator.validate_embedding_dimensions") as mock_validate, \
             patch("app.core.config.get_settings") as mock_settings, \
             patch("app.cli.get_session_factory") as mock_factory, \
             patch("app.cli.FiscusWorkflow") as mock_wf_cls, \
             patch("app.cli.DocsAgent"), \
             patch("app.cli.FinanceAgent"), \
             patch("app.cli.QueryClassifier"), \
             patch("app.cli.EmbeddingsService"), \
             patch("app.cli.DocumentRetriever"), \
             patch("app.llm.factory.create_llm_client"):
            
            mock_settings.return_value = MagicMock(
                embedding_provider="fastembed",
                fastembed_model="intfloat/multilingual-e5-small",
                embedding_model="qwen3-embedding:0.6b",
                llm_provider="ollama",
                llm_model="qwen3:8b",
            )
            mock_provider = MagicMock()
            mock_provider.dimensions = 384
            mock_create.return_value = mock_provider
            mock_validate.return_value = None
            
            mock_wf_cls.return_value = mock_workflow
            
            result = runner.invoke(cli, ["query", "Qual o horário para obras?"])
            
            # Deve mostrar informações do provider
            assert "fastembed" in result.output.lower() or "384" in result.output
