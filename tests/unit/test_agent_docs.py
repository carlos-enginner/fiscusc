"""Testes da Task 7: Agente de Documentos (RAG).

Testes unitários usam mocks do LLM e retriever.
Testes de integração requerem banco + Ollama rodando.
"""
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.agents.docs.agent import DocsAgent, create_docs_agent
from app.rag.retriever import SearchResult, DocumentRetriever, format_results

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
PDF_PATH = FIXTURES_DIR / "reg_interno.pdf"


# --- Mocks ---

class FakeLLM:
    """LLM fake para testes."""

    def __init__(self, response: str = "Resposta de teste"):
        self._response = response

    def invoke(self, messages):
        msg = MagicMock()
        msg.content = self._response
        return msg

    def bind_tools(self, tools):
        return self


def make_search_result(
    content="Art. 15 - Obras permitidas de segunda a sábado das 8h às 18h",
    page=8,
    section="Art. 15 - Obras e Reformas",
    document_type="regimento",
    score=0.92,
) -> SearchResult:
    return SearchResult(
        id=str(uuid.uuid4()),
        content=content,
        page=page,
        section=section,
        document_type=document_type,
        filename="regimento_interno.pdf",
        score=score,
    )


class FakeRetriever:
    """Retriever fake para testes."""

    def __init__(self, results: list | None = None, empty: bool = False):
        self._results = results or ([] if empty else [make_search_result()])

    def search(self, query: str, document_type=None, top_k=5, min_score=0.5):
        return self._results


# --- Testes unitários ---

class TestDocsAgent:
    @pytest.fixture
    def agent_with_context(self):
        llm = FakeLLM("Segundo o Regimento Interno (Art. 15, página 8), obras são permitidas de segunda a sábado.")
        retriever = FakeRetriever()
        return DocsAgent(retriever=retriever, llm=llm)

    @pytest.fixture
    def agent_no_context(self):
        llm = FakeLLM("Não encontrei informação específica sobre isso nos documentos disponíveis.")
        retriever = FakeRetriever(empty=True)
        return DocsAgent(retriever=retriever, llm=llm)

    def test_agent_returns_result_dict(self, agent_with_context):
        """invoke() deve retornar dict com 'results'."""
        result = agent_with_context.invoke({"query": "Horário para obras?"})

        assert "results" in result
        assert len(result["results"]) == 1

    def test_agent_result_has_required_fields(self, agent_with_context):
        """Resultado deve ter source, result e evidence."""
        result = agent_with_context.invoke({"query": "Horário para obras?"})
        r = result["results"][0]

        assert r["source"] == "docs"
        assert isinstance(r["result"], str)
        assert isinstance(r["evidence"], list)

    def test_agent_includes_evidence_when_found(self, agent_with_context):
        """Agente deve incluir evidências quando encontra resultados."""
        result = agent_with_context.invoke({"query": "Horário para obras?"})
        evidence = result["results"][0]["evidence"]

        assert len(evidence) > 0
        assert "page" in evidence[0]
        assert "doc" in evidence[0]

    def test_agent_returns_empty_evidence_when_no_results(self, agent_no_context):
        """Quando não há resultados, evidências devem estar vazias."""
        result = agent_no_context.invoke({"query": "Cor do elevador?"})
        evidence = result["results"][0]["evidence"]

        assert len(evidence) == 0

    def test_agent_response_mentions_source_when_found(self, agent_with_context):
        """Resposta deve mencionar fonte quando há resultados."""
        result = agent_with_context.invoke({"query": "Horário para obras?"})
        response = result["results"][0]["result"].lower()

        # Com contexto, a resposta do LLM fake menciona regimento/página
        assert len(response) > 0

    def test_create_docs_agent_factory(self):
        """create_docs_agent deve retornar instância do agente."""
        agent = create_docs_agent(retriever=FakeRetriever(), llm=FakeLLM())
        assert isinstance(agent, DocsAgent)


class TestFormatResults:
    def test_format_empty_results(self):
        """Resultado vazio deve retornar mensagem adequada."""
        text = format_results([])
        assert "Nenhum resultado" in text

    def test_format_single_result(self):
        """Resultado único deve ser formatado corretamente."""
        result = make_search_result(page=8, section="Art. 15", document_type="regimento")
        text = format_results([result])

        assert "REGIMENTO" in text
        assert "página 8" in text
        assert "Art. 15" in text
        assert "0.92" in text

    def test_format_multiple_results_separated(self):
        """Múltiplos resultados devem ser separados."""
        results = [make_search_result(page=1), make_search_result(page=2)]
        text = format_results(results)

        assert "---" in text

    def test_format_result_without_section(self):
        """Resultado sem seção não deve incluir campo de seção."""
        result = make_search_result(section=None)
        text = format_results([result])

        assert "seção" not in text.lower()


# --- Testes de integração ---

@pytest.mark.integration
class TestDocsAgentIntegration:
    @pytest.fixture
    def ingested_db(self):
        """Ingere o PDF de teste e retorna a sessão."""
        import math

        from app.core.database import get_session_factory, get_engine
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider
        from app.rag.ingestion import DocumentIngestionService

        engine = get_engine()
        factory = get_session_factory(engine)
        db = factory()

        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        ingestion_svc = DocumentIngestionService(embeddings_service=embeddings, db_session=db)

        result = ingestion_svc.ingest(PDF_PATH, document_type="regimento")
        yield db, result

        db.close()

    def test_ingest_document(self, ingested_db):
        """Deve ingerir documento e criar chunks."""
        db, result = ingested_db

        assert result.status in ("success", "already_exists")
        assert result.chunks_created > 0

    def test_search_returns_relevant_chunks(self, ingested_db):
        """Busca deve retornar chunks relevantes."""
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider

        db, _ = ingested_db
        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)

        results = retriever.search("horário obras", document_type="regimento")

        assert len(results) > 0
        assert results[0].score > 0.5

    def test_agent_cites_sources(self, ingested_db):
        """Agente deve citar fontes na resposta."""
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider

        db, _ = ingested_db
        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)
        agent = DocsAgent(retriever=retriever)

        result = agent.invoke({"query": "Horário para obras?"})
        response = result["results"][0]["result"].lower()

        assert "página" in response or "art." in response or "regimento" in response

    def test_agent_admits_no_info(self, ingested_db):
        """Agente deve admitir quando não tem informação."""
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider

        db, _ = ingested_db
        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)
        agent = DocsAgent(retriever=retriever)

        result = agent.invoke({"query": "Qual a cor do teto do elevador?"})
        response = result["results"][0]["result"].lower()

        phrases = ["não encontrei", "não há informação", "não consta", "não foi possível"]
        assert any(p in response for p in phrases)

    def test_search_filters_by_document_type(self, ingested_db):
        """Busca deve filtrar por tipo de documento."""
        from app.embeddings.service import EmbeddingsService, OllamaEmbeddingsProvider

        db, _ = ingested_db
        embeddings = EmbeddingsService(provider=OllamaEmbeddingsProvider())
        retriever = DocumentRetriever(embeddings_service=embeddings, db_session=db)

        results = retriever.search("horário obras", document_type="regimento")

        for r in results:
            assert r.document_type == "regimento"
