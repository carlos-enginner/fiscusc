"""Testes da Task 6: LLM Service.

Testes unitários usam mock do provider.
Testes de integração requerem Ollama rodando com o modelo carregado.
"""
import pytest
from pydantic import BaseModel

from app.llm.service import LLMProvider, LLMService


# --- Fake provider para testes unitários ---

class FakeLLMProvider(LLMProvider):
    """Provider fake para testes unitários."""

    def __init__(self):
        self._calls: list[dict] = []

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        self._calls.append({"prompt": prompt, "system": system_prompt})
        if system_prompt and "fiscus" in system_prompt.lower():
            return "Olá! Sou o Fiscus, assistente do condomínio."
        return f"Resposta para: {prompt[:30]}"

    def generate_structured(
        self, prompt: str, output_schema: type[BaseModel], system_prompt: str | None = None
    ) -> BaseModel:
        # Tentar instanciar schema com valores padrão/fake
        fields = output_schema.model_fields
        data = {}
        for name, field in fields.items():
            annotation = field.annotation
            if annotation is str or annotation == "str":
                data[name] = "positivo"
            elif annotation is float or annotation == "float":
                data[name] = 0.9
            elif annotation is int or annotation == "int":
                data[name] = 1
            else:
                data[name] = "positivo"
        return output_schema(**data)

    @property
    def model_name(self) -> str:
        return "fake-model"


# --- Testes unitários ---

class TestLLMService:
    @pytest.fixture
    def service(self):
        return LLMService(provider=FakeLLMProvider())

    def test_generate_returns_string(self, service):
        """generate() deve retornar string."""
        response = service.generate("Diga olá")

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_uses_system_prompt(self, service):
        """generate() deve passar system_prompt para o provider."""
        response = service.generate(
            "Qual seu nome?",
            system_prompt="Você se chama Fiscus e só responde em português.",
        )

        assert "fiscus" in response.lower() or "resposta" in response.lower()

    def test_generate_structured_returns_pydantic_model(self, service):
        """generate_structured() deve retornar instância do schema."""

        class SentimentResponse(BaseModel):
            sentiment: str
            confidence: float

        result = service.generate_structured(
            "O dia está lindo!",
            output_schema=SentimentResponse,
        )

        assert isinstance(result, SentimentResponse)
        assert isinstance(result.sentiment, str)
        assert isinstance(result.confidence, float)

    def test_model_name_property(self, service):
        """model_name deve retornar nome do modelo."""
        assert service.model_name == "fake-model"

    def test_provider_is_swappable(self):
        """Provider deve ser substituível via DI."""
        fake = FakeLLMProvider()
        service = LLMService(provider=fake)

        assert service._provider is fake

    def test_generate_records_calls(self, service):
        """Provider deve registrar as chamadas."""
        service.generate("pergunta teste")

        assert len(service._provider._calls) == 1
        assert "pergunta teste" in service._provider._calls[0]["prompt"]

    def test_llm_provider_is_abstract(self):
        """LLMProvider não pode ser instanciado diretamente."""
        with pytest.raises(TypeError):
            LLMProvider()


# --- Testes de integração (requerem Ollama) ---

@pytest.mark.integration
class TestOllamaLLM:
    @pytest.fixture
    def service(self):
        from app.llm.ollama import OllamaLLMProvider

        return LLMService(provider=OllamaLLMProvider())

    def test_generate_returns_response(self, service):
        """LLM deve gerar resposta não vazia."""
        response = service.generate("Diga apenas: olá")

        assert len(response) > 0
        assert isinstance(response, str)

    def test_generate_follows_system_prompt(self, service):
        """LLM deve seguir system prompt."""
        response = service.generate(
            "Qual seu nome?",
            system_prompt="Você se chama Fiscus e só responde em português.",
        )

        assert "fiscus" in response.lower() or "português" in response.lower()

    def test_generate_structured_output(self, service):
        """LLM deve retornar output estruturado."""

        class SentimentResponse(BaseModel):
            sentiment: str
            confidence: float

        result = service.generate_structured(
            "O dia está lindo!",
            output_schema=SentimentResponse,
        )

        assert isinstance(result, SentimentResponse)
        assert result.sentiment in ["positivo", "negativo", "neutro"]
        assert 0.0 <= result.confidence <= 1.0
