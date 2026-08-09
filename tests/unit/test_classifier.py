"""Testes da Task 9: Router/Classifier.

Testes unitários usam LLM fake.
Testes de integração requerem Ollama rodando.
"""
import pytest

from app.orchestrator.classifier import (
    Classification,
    ClassificationResult,
    QueryClassifier,
    classify_query,
)


# --- LLM fake para testes ---

class FakeStructuredLLM:
    """LLM fake que retorna classificação predefinida."""

    def __init__(self, classifications: list[dict]):
        self._classifications = classifications

    def invoke(self, messages):
        return ClassificationResult(
            classifications=[Classification(**c) for c in self._classifications]
        )


class FakeLLM:
    """ChatOllama fake."""

    def __init__(self, classifications: list[dict]):
        self._classifications = classifications

    def with_structured_output(self, schema):
        return FakeStructuredLLM(self._classifications)


# --- Testes unitários ---

class TestQueryClassifier:
    def _make_classifier(self, classifications: list[dict]) -> QueryClassifier:
        llm = FakeLLM(classifications)
        return QueryClassifier(llm=llm)

    def test_classify_routes_to_docs(self):
        """Perguntas sobre regras devem ir para docs."""
        classifier = self._make_classifier([
            {"source": "docs", "query": "horário permitido obras"}
        ])

        result = classifier.classify("Posso fazer obra sábado?")

        sources = [c.source for c in result]
        assert "docs" in sources
        assert "finance" not in sources

    def test_classify_routes_to_finance(self):
        """Perguntas sobre valores devem ir para finance."""
        classifier = self._make_classifier([
            {"source": "finance", "query": "valor fatura condomínio"}
        ])

        result = classifier.classify("Quanto paguei de condomínio?")

        sources = [c.source for c in result]
        assert "finance" in sources
        assert "docs" not in sources

    def test_classify_routes_to_both(self):
        """Perguntas mistas devem ir para ambos."""
        classifier = self._make_classifier([
            {"source": "docs", "query": "taxa mudança regras"},
            {"source": "finance", "query": "taxa mudança valor cobrado"},
        ])

        result = classifier.classify(
            "A taxa de mudança cobrada está de acordo com o regimento?"
        )

        sources = [c.source for c in result]
        assert "docs" in sources
        assert "finance" in sources

    def test_classification_has_optimized_query(self):
        """Cada classificação deve ter sub-pergunta."""
        classifier = self._make_classifier([
            {"source": "docs", "query": "horário permitido obras reforma"}
        ])

        result = classifier.classify("Posso fazer obra sábado?")

        assert len(result) > 0
        assert len(result[0].query) > 0

    def test_classification_source_is_valid(self):
        """source deve ser 'docs' ou 'finance'."""
        classifier = self._make_classifier([
            {"source": "docs", "query": "regras pets animais"}
        ])

        result = classifier.classify("Posso ter cachorro?")

        for c in result:
            assert c.source in ("docs", "finance")

    def test_classify_query_node_returns_dict(self):
        """classify_query deve retornar dict com 'classifications'."""
        classifier = self._make_classifier([
            {"source": "docs", "query": "horário obras"}
        ])

        result = classify_query({"query": "Horário para obras?"}, classifier=classifier)

        assert "classifications" in result
        assert len(result["classifications"]) > 0

    def test_classify_query_node_result_is_list(self):
        """'classifications' deve ser uma lista."""
        classifier = self._make_classifier([
            {"source": "finance", "query": "despesas junho"}
        ])

        result = classify_query({"query": "Despesas de junho?"}, classifier=classifier)

        assert isinstance(result["classifications"], list)


class TestClassification:
    def test_classification_model_validates_source(self):
        """source inválido deve lançar erro."""
        with pytest.raises(Exception):
            Classification(source="invalid", query="teste")

    def test_classification_accepts_docs(self):
        """'docs' deve ser válido."""
        c = Classification(source="docs", query="teste")
        assert c.source == "docs"

    def test_classification_accepts_finance(self):
        """'finance' deve ser válido."""
        c = Classification(source="finance", query="teste")
        assert c.source == "finance"


# --- Testes de integração ---

@pytest.mark.integration
class TestClassifierIntegration:
    @pytest.fixture
    def classifier(self):
        from langchain_ollama import ChatOllama
        from app.core.config import get_settings

        settings = get_settings()
        llm = ChatOllama(model=settings.llm_model, base_url=settings.ollama_base_url)
        return QueryClassifier(llm=llm)

    def test_classify_docs_query(self, classifier):
        """Query sobre regras deve ir para docs."""
        result = classifier.classify("Posso ter cachorro no apartamento?")
        sources = [c.source for c in result]
        assert "docs" in sources

    def test_classify_finance_query(self, classifier):
        """Query sobre valores deve ir para finance."""
        result = classifier.classify("Quanto é a taxa de condomínio?")
        sources = [c.source for c in result]
        assert "finance" in sources

    def test_classify_mixed_query(self, classifier):
        """Query mista deve ir para ambos."""
        result = classifier.classify(
            "O valor da taxa de mudança cobrado está correto segundo o regimento?"
        )
        sources = [c.source for c in result]
        assert "docs" in sources
        assert "finance" in sources
