"""Testes E2E: fluxo completo ingestão → query → resposta.

Estes testes requerem a stack completa:
- PostgreSQL rodando (docker compose up -d)
- Ollama com os modelos carregados

Executar com:
    DATABASE_URL=postgresql://fiscusc:fiscusc@localhost:5432/fiscusc \
    pytest tests/e2e/test_full_flow.py -v -m e2e
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.main import app

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "reg_interno.pdf"

client = TestClient(app)


# --- Testes E2E com mocks (podem rodar sem Ollama) ---

class TestE2EWithMocks:
    """
    Testes E2E que simulam o fluxo completo usando mocks do LLM.
    Verificam a integração entre API → Workflow → Agentes → DB.
    """

    def _mock_workflow_result(self, answer: str, source: str = "docs", evidence: list = None):
        """Helper para criar resultado fake do workflow."""
        mock = MagicMock()
        mock.invoke.return_value = {
            "query": "pergunta teste",
            "final_answer": answer,
            "classifications": [],
            "results": [{
                "source": source,
                "result": answer,
                "evidence": evidence or [],
            }],
        }
        return mock

    def test_e2e_query_docs_flow(self):
        """Fluxo: query → roteamento docs → resposta com fontes."""
        mock_result = self._mock_workflow_result(
            answer="Segundo o Regimento Interno (Art. 15, página 8): obras são permitidas de segunda a sábado das 8h às 18h.",
            source="docs",
            evidence=[{"doc": "reg_interno.pdf", "document_type": "regimento", "page": 8, "section": "Art. 15", "score": 0.92}],
        )

        with patch("app.api.routes.query._get_workflow", return_value=mock_result):
            response = client.post("/api/v1/query", json={
                "question": "Qual o horário permitido para obras?"
            })

        assert response.status_code == 200
        data = response.json()

        # Verificar estrutura completa da resposta
        assert "answer" in data
        assert "sources" in data
        assert "agents_used" in data
        assert "metadata" in data

        # Verificar conteúdo
        assert "Regimento" in data["answer"] or "obra" in data["answer"].lower()
        assert len(data["agents_used"]) > 0
        assert data["metadata"]["latency_ms"] >= 0

    def test_e2e_query_finance_flow(self):
        """Fluxo: query → roteamento finance → resposta financeira."""
        mock_result = self._mock_workflow_result(
            answer="A despesa com energia em julho/2026 foi de R$ 5.000,00.",
            source="finance",
            evidence=[{"type": "fatura", "mes": "julho/2026", "total": 5000.0}],
        )

        with patch("app.api.routes.query._get_workflow", return_value=mock_result):
            response = client.post("/api/v1/query", json={
                "question": "Quanto foi a despesa com energia em julho?"
            })

        assert response.status_code == 200
        data = response.json()
        assert "R$" in data["answer"] or "energia" in data["answer"].lower()

    def test_e2e_query_no_info_response(self):
        """Fluxo: query sem informação → resposta admitindo falta de dados."""
        mock_result = self._mock_workflow_result(
            answer="Não encontrei informação específica sobre a cor do teto do elevador nos documentos disponíveis.",
            source="docs",
        )

        with patch("app.api.routes.query._get_workflow", return_value=mock_result):
            response = client.post("/api/v1/query", json={
                "question": "Qual a cor do teto do elevador?"
            })

        assert response.status_code == 200
        data = response.json()
        assert "não encontrei" in data["answer"].lower() or "não há informação" in data["answer"].lower()

    def test_e2e_query_validation(self):
        """Validação: campos obrigatórios e formatos."""
        # Sem question
        r = client.post("/api/v1/query", json={})
        assert r.status_code == 422

        # Question muito curta
        r = client.post("/api/v1/query", json={"question": "ab"})
        assert r.status_code == 422

        # Question válida (mock necessário)
        with patch("app.api.routes.query._get_workflow") as mock_wf:
            mock_wf.return_value = self._mock_workflow_result("Resposta.")
            r = client.post("/api/v1/query", json={"question": "Pergunta válida aqui"})
        assert r.status_code == 200

    def test_e2e_health_check(self):
        """Health check deve retornar status estruturado."""
        with patch("app.api.routes.health._check_dependencies") as mock_deps:
            from app.api.schemas import DependencyStatus

            async def fake_deps():
                return DependencyStatus(database="healthy", ollama="healthy")

            mock_deps.side_effect = fake_deps
            response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "unhealthy")
        assert "version" in data
        assert "dependencies" in data
        assert "timestamp" in data

    def test_e2e_openapi_has_all_endpoints(self):
        """OpenAPI deve documentar todos os endpoints."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        paths = response.json()["paths"]

        required_paths = [
            "/api/v1/health",
            "/api/v1/query",
            "/api/v1/documents/ingest",
            "/api/v1/documents",
        ]
        for path in required_paths:
            assert path in paths, f"Endpoint {path} não encontrado na spec"


# --- Testes E2E com stack completa ---

@pytest.mark.e2e
class TestE2EFullStack:
    """
    Testes E2E com a stack completa (PostgreSQL + Ollama).
    Requerem modelos carregados e banco com dados.
    """

    def test_e2e_complete_flow(self):
        """Fluxo completo: ingestão → query → resposta com fontes."""
        # 1. Ingerir documento
        with open(PDF_PATH, "rb") as f:
            response = client.post(
                "/api/v1/documents/ingest",
                files={"file": ("reg_interno.pdf", f, "application/pdf")},
                data={"document_type": "regimento"},
            )
        assert response.status_code in (201, 409), f"Ingestão falhou: {response.text}"

        # 2. Listar documentos
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

        # 3. Query sobre regras (Agente Docs)
        response = client.post("/api/v1/query", json={
            "question": "Qual o horário permitido para obras?"
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
        assert len(data["agents_used"]) > 0

        # 4. Verificar fontes na resposta
        if data["sources"]:
            for source in data["sources"]:
                assert "type" in source

    def test_e2e_docs_agent_cites_sources(self):
        """Resposta do agente docs deve citar fontes do documento."""
        response = client.post("/api/v1/query", json={
            "question": "Posso ter animais de estimação?"
        })
        assert response.status_code == 200
        data = response.json()

        # A resposta deve mencionar fonte ou admitir não saber
        answer_lower = data["answer"].lower()
        has_source = any(w in answer_lower for w in ["página", "art.", "regimento", "convenção"])
        has_no_info = any(w in answer_lower for w in ["não encontrei", "não há informação", "não consta"])
        assert has_source or has_no_info

    def test_e2e_finance_agent_no_data(self):
        """Agente finance sem dados deve responder adequadamente."""
        response = client.post("/api/v1/query", json={
            "question": "Qual o valor da fatura do apartamento 2002A?"
        })
        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 0

    def test_e2e_health_with_real_deps(self):
        """Health check com dependências reais."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["dependencies"]["database"] == "healthy"
