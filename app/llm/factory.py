"""Factory para criação do cliente LLM baseado no provider configurado."""


def create_llm_client():
    """
    Retorna o cliente LLM correto baseado em LLM_PROVIDER no .env.

    - LLM_PROVIDER=ollama → ChatOllama (padrão, local)
    - LLM_PROVIDER=gemini → GeminiLangChainAdapter (Google AI Studio)

    Uso:
        from app.llm.factory import create_llm_client
        llm = create_llm_client()
    """
    from app.core.config import get_settings

    settings = get_settings()
    provider = settings.llm_provider.lower()

    if provider == "gemini":
        from app.llm.gemini import GeminiLLMProvider
        gemini = GeminiLLMProvider(
            model=settings.llm_model,
            api_key=settings.google_api_key or None,
        )
        return gemini.get_langchain_client()

    # Padrão: Ollama
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
    )
