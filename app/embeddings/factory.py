"""Factory para criar providers de embeddings."""
from app.core.config import Settings, get_settings
from app.embeddings.service import EmbeddingsProvider


def create_embedding_provider(settings: Settings | None = None) -> EmbeddingsProvider:
    """
    Cria um provider de embeddings baseado na configuração.

    Args:
        settings: Configurações. Se None, usa get_settings().

    Returns:
        EmbeddingsProvider: FastEmbedProvider ou OllamaEmbeddingsProvider.

    Raises:
        ValueError: Se embedding_provider não for "fastembed" ou "ollama".
    """
    if settings is None:
        settings = get_settings()

    if settings.embedding_provider == "fastembed":
        from app.embeddings.fastembed_provider import FastEmbedProvider

        return FastEmbedProvider(settings.fastembed_model)

    if settings.embedding_provider == "ollama":
        from app.embeddings.service import OllamaEmbeddingsProvider

        return OllamaEmbeddingsProvider(settings.embedding_model)

    raise ValueError(
        f"embedding_provider inválido: {settings.embedding_provider}. "
        f"Deve ser 'fastembed' ou 'ollama'."
    )
