"""Testes da Task 12: CLI Commands."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from app.cli import cli

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "reg_interno.pdf"

runner = CliRunner()


class TestCLIStatus:
    def test_cli_status_runs(self):
        """CLI status deve executar sem crash."""
        with patch("app.cli.check_db_connection", return_value=True), \
             patch("app.cli.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"models": [{"name": "qwen3:8b"}]}
            mock_httpx.get.return_value = mock_resp

            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0

    def test_cli_status_shows_database(self):
        """CLI status deve mostrar status do banco."""
        with patch("app.cli.check_db_connection", return_value=True), \
             patch("app.cli.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"models": []}
            mock_httpx.get.return_value = mock_resp

            result = runner.invoke(cli, ["status"])

        assert "PostgreSQL" in result.output or "database" in result.output.lower()

    def test_cli_status_shows_ollama(self):
        """CLI status deve mostrar status do Ollama."""
        with patch("app.cli.check_db_connection", return_value=True), \
             patch("app.cli.httpx") as mock_httpx:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"models": []}
            mock_httpx.get.return_value = mock_resp

            result = runner.invoke(cli, ["status"])

        assert "Ollama" in result.output


class TestCLIIngest:
    def test_cli_ingest_processes_pdf(self):
        """CLI ingest deve processar PDF."""
        mock_result = MagicMock()
        mock_result.document_id = "test-uuid"
        mock_result.pages = 10
        mock_result.chunks_created = 45
        mock_result.sha256 = "abc123" * 11
        mock_result.already_existed = False
        mock_result.filename = "reg_interno.pdf"
        mock_result.document_type = "regimento"
        mock_result.status = "success"

        with patch("app.cli.DocumentIngestionService") as mock_svc_cls, \
             patch("app.cli.EmbeddingsService"), \
             patch("app.cli.get_engine"), \
             patch("app.cli.get_session_factory"):
            mock_svc = MagicMock()
            mock_svc.ingest.return_value = mock_result
            mock_svc_cls.return_value = mock_svc

            result = runner.invoke(cli, ["ingest", str(PDF_PATH), "--type", "regimento"])

        assert result.exit_code == 0
        assert "chunks" in result.output.lower() or "ingest" in result.output.lower()

    def test_cli_ingest_requires_valid_file(self):
        """CLI ingest com arquivo inexistente deve falhar."""
        result = runner.invoke(cli, ["ingest", "/nonexistent/file.pdf"])
        assert result.exit_code != 0

    def test_cli_ingest_invalid_type(self):
        """CLI ingest com tipo inválido deve falhar."""
        result = runner.invoke(cli, ["ingest", str(PDF_PATH), "--type", "invalid_type"])
        assert result.exit_code != 0

    def test_cli_ingest_already_exists(self):
        """CLI ingest de documento já existente deve notificar."""
        mock_result = MagicMock()
        mock_result.document_id = "existing-uuid"
        mock_result.already_existed = True

        with patch("app.cli.DocumentIngestionService") as mock_svc_cls, \
             patch("app.cli.EmbeddingsService"), \
             patch("app.cli.get_engine"), \
             patch("app.cli.get_session_factory"):
            mock_svc = MagicMock()
            mock_svc.ingest.return_value = mock_result
            mock_svc_cls.return_value = mock_svc

            result = runner.invoke(cli, ["ingest", str(PDF_PATH)])

        assert result.exit_code == 0
        assert "existe" in result.output.lower() or "existing" in result.output.lower()


class TestCLIQuery:
    def test_cli_query_returns_answer(self):
        """CLI query deve retornar resposta."""
        mock_workflow = MagicMock()
        mock_workflow.invoke.return_value = {
            "final_answer": "Obras são permitidas de segunda a sábado.",
            "results": [{"source": "docs", "result": "...", "evidence": []}],
            "classifications": [],
        }

        with patch("app.cli.FiscusWorkflow") as mock_wf_cls, \
             patch("app.cli.DocsAgent"), \
             patch("app.cli.FinanceAgent"), \
             patch("app.cli.QueryClassifier"), \
             patch("app.cli.EmbeddingsService"), \
             patch("app.cli.DocumentRetriever"), \
             patch("langchain_ollama.ChatOllama"), \
             patch("app.cli.get_engine"), \
             patch("app.cli.get_session_factory"):
            mock_wf_cls.return_value = mock_workflow

            result = runner.invoke(cli, ["query", "Horário para obras?"])

        assert result.exit_code == 0
        assert len(result.output) > 0

    def test_cli_query_requires_question(self):
        """CLI query sem pergunta deve falhar."""
        result = runner.invoke(cli, ["query"])
        assert result.exit_code != 0

    def test_cli_help_available(self):
        """CLI deve mostrar ajuda."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "query" in result.output
        assert "status" in result.output


# --- Testes de integração ---

@pytest.mark.integration
class TestCLIIntegration:
    def test_cli_status_real(self):
        """CLI status com banco real deve mostrar PostgreSQL OK."""
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "PostgreSQL" in result.output

    def test_cli_ingest_real(self):
        """CLI ingest com PDF real deve criar chunks."""
        result = runner.invoke(cli, [
            "ingest", str(PDF_PATH), "--type", "regimento"
        ])
        assert result.exit_code == 0
        assert "chunks" in result.output.lower() or "existe" in result.output.lower()


class TestProgressCallbacks:
    """Testes para o sistema de callbacks de progresso por fase."""

    def test_progress_callbacks_dataclass_defaults(self):
        """ProgressCallbacks deve ter todos callbacks como None por padrão."""
        from app.rag.ingestion import ProgressCallbacks
        
        cb = ProgressCallbacks()
        
        assert cb.on_phase_start is None
        assert cb.on_phase_end is None
        assert cb.on_extraction_progress is None
        assert cb.on_chunking_progress is None
        assert cb.on_embedding_progress is None
        assert cb.on_saving_progress is None

    def test_progress_callbacks_accepts_callables(self):
        """ProgressCallbacks deve aceitar funções como callbacks."""
        from app.rag.ingestion import ProgressCallbacks
        
        phase_starts = []
        phase_ends = []
        extraction_progress = []
        
        cb = ProgressCallbacks(
            on_phase_start=lambda p: phase_starts.append(p),
            on_phase_end=lambda p: phase_ends.append(p),
            on_extraction_progress=lambda c, t: extraction_progress.append((c, t)),
        )
        
        cb.on_phase_start("extraction")
        cb.on_phase_end("extraction")
        cb.on_extraction_progress(10, 100)
        
        assert phase_starts == ["extraction"]
        assert phase_ends == ["extraction"]
        assert extraction_progress == [(10, 100)]

    def test_ingest_calls_phase_callbacks(self):
        """ingest() deve chamar callbacks de início/fim de fase."""
        from app.rag.ingestion import DocumentIngestionService, ProgressCallbacks
        
        phases_started = []
        phases_ended = []
        
        mock_result = MagicMock()
        mock_result.document_id = "test-uuid"
        mock_result.pages = 5
        mock_result.chunks_created = 10
        mock_result.sha256 = "abc123" * 11
        mock_result.already_existed = False
        mock_result.filename = "test.pdf"
        mock_result.document_type = "regimento"
        mock_result.status = "success"
        mock_result.metrics = MagicMock()

        with patch("app.rag.ingestion.extract_pdf") as mock_extract, \
             patch("app.rag.ingestion.chunk_pages") as mock_chunk, \
             patch("app.rag.ingestion.calculate_sha256", return_value="abc123" * 11), \
             patch("app.rag.ingestion.get_file_size", return_value=1000), \
             patch("app.rag.ingestion.get_settings") as mock_settings:
            
            # Setup mocks
            mock_settings.return_value = MagicMock(
                enable_embedding_cache=False,
                enable_incremental_ingest=False,
                embedding_batch_size=16,
            )
            mock_extract.return_value = [MagicMock(page=1, text="test")]
            
            mock_chunk_obj = MagicMock()
            mock_chunk_obj.content = "test content"
            mock_chunk_obj.page = 1
            mock_chunk_obj.section = None
            mock_chunk_obj.chapter = None
            mock_chunk_obj.article = None
            mock_chunk_obj.chunk_index = 0
            mock_chunk_obj.content_length = 12
            mock_chunk.return_value = [mock_chunk_obj]
            
            mock_embeddings = MagicMock()
            mock_embeddings.embed_batch.return_value = [[0.1] * 1024]
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            svc = DocumentIngestionService(
                embeddings_service=mock_embeddings,
                db_session=mock_db,
            )
            # Override cache
            svc._cache = None
            
            callbacks = ProgressCallbacks(
                on_phase_start=lambda p: phases_started.append(p),
                on_phase_end=lambda p: phases_ended.append(p),
            )
            
            svc.ingest(
                path="test.pdf",
                document_type="regimento",
                progress_callbacks=callbacks,
            )
        
        # Deve ter chamado todas as 4 fases
        assert "extraction" in phases_started
        assert "chunking" in phases_started
        assert "embedding" in phases_started
        assert "saving" in phases_started
        
        assert "extraction" in phases_ended
        assert "chunking" in phases_ended
        assert "embedding" in phases_ended
        assert "saving" in phases_ended

    def test_ingest_calls_embedding_progress(self):
        """ingest() deve reportar progresso de embeddings."""
        from app.rag.ingestion import DocumentIngestionService, ProgressCallbacks
        
        embedding_progress = []

        with patch("app.rag.ingestion.extract_pdf") as mock_extract, \
             patch("app.rag.ingestion.chunk_pages") as mock_chunk, \
             patch("app.rag.ingestion.calculate_sha256", return_value="abc123" * 11), \
             patch("app.rag.ingestion.get_file_size", return_value=1000), \
             patch("app.rag.ingestion.get_settings") as mock_settings:
            
            mock_settings.return_value = MagicMock(
                enable_embedding_cache=False,
                enable_incremental_ingest=False,
                embedding_batch_size=2,  # Batch pequeno para testar múltiplas chamadas
            )
            mock_extract.return_value = [MagicMock(page=1, text="test")]
            
            # Criar 4 chunks para testar 2 batches
            chunks = []
            for i in range(4):
                c = MagicMock()
                c.content = f"content {i}"
                c.page = 1
                c.section = None
                c.chapter = None
                c.article = None
                c.chunk_index = i
                c.content_length = 10
                chunks.append(c)
            mock_chunk.return_value = chunks
            
            mock_embeddings = MagicMock()
            mock_embeddings.embed_batch.return_value = [[0.1] * 1024, [0.2] * 1024]
            
            mock_db = MagicMock()
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            svc = DocumentIngestionService(
                embeddings_service=mock_embeddings,
                db_session=mock_db,
            )
            svc._cache = None
            
            callbacks = ProgressCallbacks(
                on_embedding_progress=lambda c, t: embedding_progress.append((c, t)),
            )
            
            svc.ingest(
                path="test.pdf",
                document_type="regimento",
                progress_callbacks=callbacks,
            )
        
        # Com 4 chunks e batch_size=2, deve ter 2 updates de progresso
        assert len(embedding_progress) == 2
        assert embedding_progress[0] == (2, 4)  # Após primeiro batch
        assert embedding_progress[1] == (4, 4)  # Após segundo batch

    def test_cli_ingest_uses_progress_callbacks(self):
        """CLI ingest deve usar o novo sistema de callbacks."""
        mock_result = MagicMock()
        mock_result.document_id = "test-uuid"
        mock_result.pages = 10
        mock_result.chunks_created = 45
        mock_result.sha256 = "abc123" * 11
        mock_result.already_existed = False
        mock_result.filename = "reg_interno.pdf"
        mock_result.document_type = "regimento"
        mock_result.status = "success"
        mock_result.metrics = MagicMock()
        mock_result.metrics.total_ms = 5000
        mock_result.metrics.extraction_ms = 1000
        mock_result.metrics.chunking_ms = 500
        mock_result.metrics.embedding_ms = 3000
        mock_result.metrics.db_ms = 500
        mock_result.metrics.chunks_count = 45
        mock_result.metrics.cache_hits = 10
        mock_result.metrics.cache_misses = 35
        mock_result.metrics.incremental_reused = 0
        mock_result.metrics.chunks_per_sec = 9.0

        with patch("app.cli.DocumentIngestionService") as mock_svc_cls, \
             patch("app.cli.EmbeddingsService"), \
             patch("app.cli.get_engine"), \
             patch("app.cli.get_session_factory"):
            mock_svc = MagicMock()
            mock_svc.ingest.return_value = mock_result
            mock_svc_cls.return_value = mock_svc

            result = runner.invoke(cli, ["ingest", str(PDF_PATH), "--type", "regimento"])
            
            # Verificar que ingest foi chamado com progress_callbacks
            call_kwargs = mock_svc.ingest.call_args.kwargs
            assert "progress_callbacks" in call_kwargs
            assert call_kwargs["progress_callbacks"] is not None

        assert result.exit_code == 0
