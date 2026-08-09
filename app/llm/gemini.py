"""Provider de LLM usando Google Gemini (google-genai SDK)."""
import os

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.llm.service import LLMProvider


class GeminiLLMProvider(LLMProvider):
    """
    Provider de LLM usando Google Gemini via google-genai SDK.

    Compatível com a interface LLMProvider sem depender de langchain-google-genai.

    Configuração necessária no .env:
        GOOGLE_API_KEY=sua_chave_aqui
        LLM_PROVIDER=gemini
        LLM_MODEL=gemini-2.0-flash   # ou gemini-1.5-pro, gemini-2.5-flash etc.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            raise ImportError(
                "google-genai não instalado. Instale com:\n"
                "  pip install google-genai"
            )

        from app.core.config import get_settings
        settings = get_settings()

        self._model = model or settings.llm_model
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

        if not self._api_key:
            raise ValueError(
                "GOOGLE_API_KEY não configurada. Adicione ao .env:\n"
                "  GOOGLE_API_KEY=sua_chave_aqui"
            )

        self._client = genai.Client(api_key=self._api_key)
        self._types = genai_types

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Gera resposta de texto via Gemini."""
        contents = []

        if system_prompt:
            # Gemini usa system_instruction separado
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                ),
            )
        else:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )

        return response.text

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[BaseModel],
        system_prompt: str | None = None,
    ) -> BaseModel:
        """Gera resposta estruturada usando Pydantic schema via JSON mode."""
        import json

        schema_str = output_schema.model_json_schema()
        schema_prompt = (
            f"\nRetorne APENAS um JSON válido seguindo este schema:\n"
            f"{json.dumps(schema_str, ensure_ascii=False, indent=2)}\n"
            f"Sem explicações, apenas o JSON."
        )

        full_prompt = prompt + schema_prompt

        config = self._types.GenerateContentConfig(
            response_mime_type="application/json",
        )
        if system_prompt:
            config = self._types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            )

        response = self._client.models.generate_content(
            model=self._model,
            contents=full_prompt,
            config=config,
        )

        data = json.loads(response.text)
        return output_schema(**data)

    def get_langchain_client(self):
        """
        Retorna um wrapper compatível com LangChain para uso no LangGraph.
        Necessário para o workflow funcionar com Gemini.
        """
        return _GeminiLangChainAdapter(self)

    @property
    def model_name(self) -> str:
        return self._model


class _GeminiLangChainAdapter:
    """
    Adapter mínimo que faz o GeminiLLMProvider parecer um ChatOllama
    para o LangGraph/agentes.
    """

    def __init__(self, provider: GeminiLLMProvider):
        self._provider = provider

    def bind_tools(self, tools):
        """Compatibilidade com LangChain — Gemini ignora tools, usa busca direta."""
        return self

    def invoke(self, messages) -> object:
        """Converte messages LangChain para chamada Gemini."""
        system_prompt = None
        user_parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_parts.append(msg.content)
            elif hasattr(msg, "role"):
                if msg.get("role") == "system":
                    system_prompt = msg["content"]
                else:
                    user_parts.append(msg.get("content", ""))

        prompt = "\n\n".join(user_parts)
        text = self._provider.generate(prompt, system_prompt=system_prompt)

        # Retornar objeto com .content igual ao ChatOllama
        class _Msg:
            content = text

        return _Msg()

    def with_structured_output(self, schema: type[BaseModel]):
        """Retorna adapter para structured output."""
        return _GeminiStructuredAdapter(self._provider, schema)


class _GeminiStructuredAdapter:
    """Adapter para structured output com Gemini."""

    def __init__(self, provider: GeminiLLMProvider, schema: type[BaseModel]):
        self._provider = provider
        self._schema = schema

    def invoke(self, messages) -> BaseModel:
        system_prompt = None
        user_parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_parts.append(msg.content)
            elif hasattr(msg, "get"):
                if msg.get("role") == "system":
                    system_prompt = msg["content"]
                else:
                    user_parts.append(msg.get("content", ""))

        prompt = "\n\n".join(user_parts)

        try:
            return self._provider.generate_structured(
                prompt, output_schema=self._schema, system_prompt=system_prompt
            )
        except Exception:
            return None
