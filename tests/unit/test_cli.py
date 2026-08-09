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
