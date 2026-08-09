"""Serviço de LLM com interface abstrata e implementação Ollama."""
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class LLMProvider(ABC):
    """Interface abstrata para providers de LLM."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Gera resposta de texto para um prompt."""
        ...

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Gera resposta estruturada (Pydantic model)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nome do modelo em uso."""
        ...


class LLMService:
    """
    Serviço de LLM.

    Usa OllamaLLMProvider por padrão.
    Provider pode ser trocado via DI para testes ou outros backends.
    """

    def __init__(self, provider: LLMProvider | None = None):
        if provider is None:
            # Import lazy para evitar circular imports
            from app.llm.ollama import OllamaLLMProvider

            provider = OllamaLLMProvider()
        self._provider = provider

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Gera resposta de texto."""
        return self._provider.generate(prompt, system_prompt=system_prompt)

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Gera resposta estruturada."""
        return self._provider.generate_structured(
            prompt, output_schema=output_schema, system_prompt=system_prompt
        )

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    def get_langchain_client(self):
        """Retorna cliente LangChain (para uso com LangGraph/Agents)."""
        from app.llm.ollama import OllamaLLMProvider

        if isinstance(self._provider, OllamaLLMProvider):
            return self._provider.get_client()
        raise NotImplementedError("get_langchain_client não disponível para este provider")
