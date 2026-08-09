"""Testes da Task 11: FastAPI Endpoints.

Testes unitários usam mock do workflow.
Testes de integração requerem a stack completa.
"""
import io
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "reg_interno.pdf"

client = TestClient(app)


# --- Testes do health endpoint ---

class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health deve retornar 200."""
        with patch("app.api.routes.health._check_dependencies") as mock_deps:
            mock_deps.return_value = MagicMock(
                database="healthy",
                ollama="healthy",
                embeddings_model=None,
                llm_model=None,
                error=None,
            )
            # Fazer request async funcionar com TestClient
            import asyncio

            async def fake_check():
                from app.api.schemas import DependencyStatus
                return DependencyStatus(database="healthy", ollama="healthy")

            mock_deps.side_effect = fake_check

            response = client.get("/api/v1/health")

        assert response.status_code == 200

    def test_health_has_status_field(self):
        """Response de health deve ter campo status."""
        with patch("app.api.routes.health._check_dependencies") as mock_deps:
            from app.api.schemas import DependencyStatus

            async def fake_check():
                return DependencyStatus(database="healthy", ollama="healthy")

            mock_deps.side_effect = fake_check
            response = client.get("/api/v1/health")

        data = response.json()
        assert "status" in data

    def test_health_has_version(self):
        """Response de health deve ter campo version."""
        with patch("app.api.routes.health._check_dependencies") as mock_deps:
            from app.api.schemas import DependencyStatus

            async def fake_check():
                return DependencyStatus(database="healthy", ollama="healthy")

            mock_deps.side_effect = fake_check
            response = client.get("/api/v1/health")

        data = response.json()
        assert "version" in data
        assert data["version"] == "1.0.0"


# --- Testes do query endpoint ---

class TestQueryEndpoint:
    def _mock_workflow(self, answer="Resposta teste", agents=None, results=None):
        """Cria mock do workflow."""
        workflow_mock = MagicMock()
        workflow_mock.invoke.return_value = {
            "query": "pergunta",
            "final_answer": answer,
            "classifications": [],
            "results": results or [
                {"source": agents[0] if agents else "docs", "result": answer, "evidence": []}
            ],
        }
        return workflow_mock

    def test_query_returns_200(self):
        """POST /query deve retornar 200."""
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow()
            response = client.post("/api/v1/query", json={"question": "Horário para obras?"})

        assert response.status_code == 200

    def test_query_has_answer_field(self):
        """Response deve ter campo answer."""
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow(answer="Segundo o Regimento...")
            response = client.post("/api/v1/query", json={"question": "Horário para obras?"})

        data = response.json()
        assert "answer" in data
        assert "Regimento" in data["answer"]

    def test_query_has_sources_field(self):
        """Response deve ter campo sources."""
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow()
            response = client.post("/api/v1/query", json={"question": "Horário?"})

        data = response.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_has_agents_used(self):
        """Response deve ter campo agents_used."""
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow()
            response = client.post("/api/v1/query", json={"question": "Horário?"})

        data = response.json()
        assert "agents_used" in data
        assert isinstance(data["agents_used"], list)

    def test_query_has_metadata(self):
        """Response deve ter campo metadata."""
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow()
            response = client.post("/api/v1/query", json={"question": "Horário?"})

        data = response.json()
        assert "metadata" in data
        assert "query_id" in data["metadata"]
        assert "latency_ms" in data["metadata"]

    def test_query_requires_question(self):
        """POST /query sem question deve retornar 422."""
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_requires_min_length(self):
        """POST /query com question muito curta deve retornar 422."""
        response = client.post("/api/v1/query", json={"question": "ab"})
        assert response.status_code == 422


# --- Testes do endpoint de documentos ---

class TestDocumentsEndpoint:
    def test_list_documents_endpoint_exists(self):
        """GET /documents deve existir como rota."""
        # Verificar no schema OpenAPI que a rota existe
        response = client.get("/openapi.json")
        data = response.json()
        assert "/api/v1/documents" in data["paths"]

    def test_ingest_requires_file(self):
        """POST /documents/ingest sem arquivo deve retornar 422."""
        response = client.post(
            "/api/v1/documents/ingest",
            data={"document_type": "regimento"},
        )
        assert response.status_code == 422

    def test_schemas_are_valid(self):
        """Schemas Pydantic devem ser instanciáveis."""
        from app.api.schemas import (
            QueryRequest,
            QueryFilters,
            QueryResponse,
            Source,
            QueryMetadata,
            IngestResponse,
            DocumentResponse,
            HealthResponse,
            DependencyStatus,
        )

        # QueryRequest
        req = QueryRequest(question="Horário para obras?")
        assert req.question == "Horário para obras?"

        # Source
        src = Source(type="document", page=5, score=0.9)
        assert src.type == "document"

        # DependencyStatus
        deps = DependencyStatus(database="healthy", ollama="healthy")
        assert deps.database == "healthy"

    def test_openapi_schema_valid(self):
        """OpenAPI schema deve ser acessível."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/api/v1/query" in data["paths"]
        assert "/api/v1/health" in data["paths"]


# --- Testes de integração ---

@pytest.mark.integration
class TestAPIIntegration:
    def test_health_endpoint_with_real_deps(self):
        """GET /health deve verificar dependências reais."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["dependencies"]["database"] == "healthy"

    def test_ingest_and_query_flow(self):
        """Fluxo completo: ingerir PDF e fazer query."""
        # Ingerir
        with open(PDF_PATH, "rb") as f:
            response = client.post(
                "/api/v1/documents/ingest",
                files={"file": ("reg_interno.pdf", f, "application/pdf")},
                data={"document_type": "regimento"},
            )
        assert response.status_code in (201, 409)

        # Query
        response = client.post("/api/v1/query", json={
            "question": "Qual o horário para obras?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
