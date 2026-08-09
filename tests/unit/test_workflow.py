"""Testes da Task 10: LangGraph Workflow.

Testes unitários usam mocks dos agentes e classificador.
Testes de integração requerem a stack completa.
"""
import time
from unittest.mock import MagicMock

import pytest

from app.orchestrator.classifier import Classification, ClassificationResult, QueryClassifier
from app.orchestrator.workflow import (
    AgentInput,
    AgentOutput,
    FiscusState,
    FiscusWorkflow,
    create_fiscus_workflow,
    _synthesize_results,
)


# --- Mocks ---

class FakeClassifier(QueryClassifier):
    """Classifier que retorna resultado predefinido."""

    def __init__(self, classifications: list[dict]):
        super().__init__(llm=None)
        self._fixed = [Classification(**c) for c in classifications]

    def classify(self, query: str) -> list[Classification]:
        return self._fixed


class FakeDocsAgent:
    def invoke(self, state: dict) -> dict:
        return {
            "results": [
                {
                    "source": "docs",
                    "result": f"Segundo o Regimento Interno (página 8): {state['query']}",
                    "evidence": [{"doc": "regimento.pdf", "page": 8, "section": "Art. 15", "score": 0.9}],
                }
            ]
        }


class FakeFinanceAgent:
    def invoke(self, state: dict) -> dict:
        return {
            "results": [
                {
                    "source": "finance",
                    "result": f"Valor da fatura: R$ 795,96 (julho/2026)",
                    "evidence": [{"type": "fatura", "mes": "julho/2026", "total": 795.96}],
                }
            ]
        }


# --- Testes unitários ---

class TestFiscusWorkflow:
    @pytest.fixture
    def workflow_docs_only(self):
        classifier = FakeClassifier([{"source": "docs", "query": "horário obras"}])
        return FiscusWorkflow(
            docs_agent=FakeDocsAgent(),
            finance_agent=FakeFinanceAgent(),
            classifier=classifier,
        )

    @pytest.fixture
    def workflow_finance_only(self):
        classifier = FakeClassifier([{"source": "finance", "query": "valor fatura"}])
        return FiscusWorkflow(
            docs_agent=FakeDocsAgent(),
            finance_agent=FakeFinanceAgent(),
            classifier=classifier,
        )

    @pytest.fixture
    def workflow_both(self):
        classifier = FakeClassifier([
            {"source": "docs", "query": "taxa mudança regras"},
            {"source": "finance", "query": "taxa mudança valor"},
        ])
        return FiscusWorkflow(
            docs_agent=FakeDocsAgent(),
            finance_agent=FakeFinanceAgent(),
            classifier=classifier,
        )

    def test_workflow_single_agent_docs(self, workflow_docs_only):
        """Workflow com roteamento docs deve retornar resposta."""
        result = workflow_docs_only.invoke("Horário para obras?")

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_workflow_single_agent_finance(self, workflow_finance_only):
        """Workflow com roteamento finance deve retornar resposta."""
        result = workflow_finance_only.invoke("Quanto paguei de condomínio?")

        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_workflow_has_results_list(self, workflow_docs_only):
        """Workflow deve retornar lista de results."""
        result = workflow_docs_only.invoke("Horário para obras?")

        assert "results" in result
        assert isinstance(result["results"], list)

    def test_workflow_single_agent_uses_result_directly(self, workflow_docs_only):
        """Com um agente, final_answer deve ser o result do agente."""
        result = workflow_docs_only.invoke("Horário para obras?")

        assert result["final_answer"] == result["results"][0]["result"]

    def test_workflow_multiple_agents_combines_results(self, workflow_both):
        """Com múltiplos agentes, final_answer deve combinar resultados."""
        result = workflow_both.invoke("Taxa de mudança está correta?")

        assert "final_answer" in result
        # Deve ter resultados de ambos os agentes
        sources = [r["source"] for r in result["results"]]
        assert "docs" in sources
        assert "finance" in sources

    def test_workflow_returns_sources(self, workflow_docs_only):
        """Workflow deve retornar evidências nos results."""
        result = workflow_docs_only.invoke("Horário para obras?")

        assert len(result["results"]) > 0
        assert "evidence" in result["results"][0]

    def test_create_fiscus_workflow_factory(self):
        """create_fiscus_workflow deve retornar instância."""
        classifier = FakeClassifier([{"source": "docs", "query": "teste"}])
        workflow = create_fiscus_workflow(
            docs_agent=FakeDocsAgent(),
            classifier=classifier,
        )
        assert isinstance(workflow, FiscusWorkflow)


class TestSynthesizeResults:
    def test_synthesize_empty_results(self):
        """Sem resultados, deve retornar mensagem padrão."""
        state = {
            "query": "pergunta",
            "classifications": [],
            "results": [],
            "final_answer": "",
        }
        result = _synthesize_results(state)
        assert "não foi possível" in result["final_answer"].lower()

    def test_synthesize_single_result(self):
        """Com um resultado, deve usar diretamente."""
        state = {
            "query": "pergunta",
            "classifications": [],
            "results": [{"source": "docs", "result": "Resposta direta", "evidence": []}],
            "final_answer": "",
        }
        result = _synthesize_results(state)
        assert result["final_answer"] == "Resposta direta"

    def test_synthesize_multiple_results_no_llm(self):
        """Com múltiplos resultados sem LLM, deve concatenar."""
        state = {
            "query": "pergunta",
            "classifications": [],
            "results": [
                {"source": "docs", "result": "Info do regimento", "evidence": []},
                {"source": "finance", "result": "Info financeira", "evidence": []},
            ],
            "final_answer": "",
        }
        result = _synthesize_results(state, llm=None)
        assert "DOCS" in result["final_answer"] or "docs" in result["final_answer"].lower()
        assert "FINANCE" in result["final_answer"] or "finance" in result["final_answer"].lower()

    def test_synthesize_multiple_results_with_llm(self):
        """Com LLM disponível, deve invocar síntese."""
        fake_llm = MagicMock()
        fake_response = MagicMock()
        fake_response.content = "Síntese: regimento permite, fatura cobra corretamente."
        fake_llm.invoke.return_value = fake_response

        state = {
            "query": "taxa correta?",
            "classifications": [],
            "results": [
                {"source": "docs", "result": "Regimento permite R$ 150", "evidence": []},
                {"source": "finance", "result": "Cobrado R$ 150", "evidence": []},
            ],
            "final_answer": "",
        }

        result = _synthesize_results(state, llm=fake_llm)
        assert "síntese" in result["final_answer"].lower()
        fake_llm.invoke.assert_called_once()


# --- Testes de integração ---

@pytest.mark.integration
class TestWorkflowIntegration:
    @pytest.fixture
    def full_workflow(self):
        """Workflow com agentes reais (requer Ollama + banco)."""
        from app.agents.docs.agent import DocsAgent
        from app.agents.finance.agent import FinanceAgent
        from app.core.database import get_session_factory, get_engine
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider
        from app.orchestrator.classifier import QueryClassifier
        from app.rag.retriever import DocumentRetriever
        from langchain_ollama import ChatOllama
        from app.core.config import get_settings

        settings = get_settings()
        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()

        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)
        llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)

        return FiscusWorkflow(
            docs_agent=DocsAgent(retriever=retriever, llm=llm),
            finance_agent=FinanceAgent(db_session=db, llm=llm),
            classifier=QueryClassifier(llm=llm),
            llm=llm,
        )

    def test_workflow_docs_query(self, full_workflow):
        """Query sobre documentos deve retornar resposta."""
        result = full_workflow.invoke("Qual o horário permitido para obras?")

        assert "final_answer" in result
        assert len(result["final_answer"]) > 10

    def test_workflow_returns_sources(self, full_workflow):
        """Workflow deve retornar fontes."""
        result = full_workflow.invoke("Posso ter animais de estimação?")

        assert len(result["results"]) > 0
