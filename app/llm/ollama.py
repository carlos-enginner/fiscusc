"""Provider de LLM usando Ollama."""
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.core.config import get_settings
from app.llm.service import LLMProvider


class OllamaLLMProvider(LLMProvider):
    """Provider de LLM usando Ollama via langchain-ollama."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        settings = get_settings()
        self._model = model or settings.llm_model
        self._base_url = base_url or settings.ollama_base_url
        self._llm = ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=0,
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Gera resposta para um prompt."""
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self._llm.invoke(messages)
        return response.content

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Gera resposta estruturada usando Pydantic schema."""
        structured_llm = self._llm.with_structured_output(output_schema)

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        return structured_llm.invoke(messages)

    def get_client(self) -> ChatOllama:
        """Retorna o cliente LangChain para uso avançado."""
        return self._llm

    @property
    def model_name(self) -> str:
        return self._model
