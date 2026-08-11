"""Provider de embeddings usando FastEmbed (local, sem HTTP)."""
from app.core.config import get_settings
from app.embeddings.service import EmbeddingsProvider


class FastEmbedProvider(EmbeddingsProvider):
    """
    Provider de embeddings usando FastEmbed.

    Executa localmente sem necessidade de servidor HTTP.
    Modelo padrão: intfloat/multilingual-e5-small (384 dims).
    """

    def __init__(self, model: str | None = None):
        """
        Inicializa o provider FastEmbed.

        Args:
            model: Nome do modelo FastEmbed. Se None, usa config.
        """
        from fastembed import TextEmbedding

        settings = get_settings()
        self._model_name = model or settings.fastembed_model
        self._model = TextEmbedding(model_name=self._model_name)
        # Auto-detectar dimensão via embedding de teste
        self._dimensions = self._detect_dimensions()

    def _detect_dimensions(self) -> int:
        """Detecta dimensão do modelo via embedding de teste."""
        test_embedding = list(self._model.embed(["test"]))[0]
        return len(test_embedding)

    def embed(self, text: str) -> list[float]:
        """Gera embedding para um texto."""
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Gera embeddings para múltiplos textos.

        Usa batch nativo do FastEmbed (sem HTTP).
        """
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    async def embed_batch_async(
        self, texts: list[str], max_workers: int = 4
    ) -> list[list[float]]:
        """
        Gera embeddings de forma assíncrona.

        FastEmbed já é eficiente com batch, então apenas
        delega para o método síncrono em thread separada.
        """
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)

    @property
    def dimensions(self) -> int:
        """Dimensão dos vetores gerados (auto-detectada)."""
        return self._dimensions
