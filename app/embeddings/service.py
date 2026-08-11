"""Serviço de embeddings com interface abstrata e implementação via Ollama."""
from abc import ABC, abstractmethod

import numpy as np
from ollama import Client

from app.core.config import get_settings


class EmbeddingsProvider(ABC):
    """Interface abstrata para providers de embeddings."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Gera embedding para um texto."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para múltiplos textos."""
        ...

    @abstractmethod
    async def embed_batch_async(
        self, texts: list[str], max_workers: int = 4
    ) -> list[list[float]]:
        """Gera embeddings para múltiplos textos de forma assíncrona."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensão dos vetores gerados."""
        ...


class OllamaEmbeddingsProvider(EmbeddingsProvider):
    """Provider de embeddings usando Ollama."""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        settings = get_settings()
        self._model = model or settings.embedding_model
        self._client = Client(host=base_url or settings.ollama_base_url)
        self._base_url = base_url or settings.ollama_base_url

    def embed(self, text: str) -> list[float]:
        """Gera embedding via Ollama."""
        response = self._client.embeddings(model=self._model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para múltiplos textos (agora usando async)."""
        import asyncio
        return asyncio.run(self.embed_batch_async(texts))

    async def embed_batch_async(
        self, texts: list[str], max_workers: int = 4
    ) -> list[list[float]]:
        """
        Gera embeddings em paralelo com controle de concorrência.
        
        Args:
            texts: Lista de textos
            max_workers: Máximo de requisições simultâneas
            
        Returns:
            Lista de embeddings na mesma ordem dos textos
        """
        import asyncio
        import httpx
        
        semaphore = asyncio.Semaphore(max_workers)
        
        async def embed_one(text: str) -> list[float]:
            async with semaphore:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"{self._base_url}/api/embeddings",
                        json={"model": self._model, "prompt": text}
                    )
                    response.raise_for_status()
                    return response.json()["embedding"]
        
        # Processar todos em paralelo mantendo ordem
        return await asyncio.gather(*[embed_one(t) for t in texts])

    @property
    def dimensions(self) -> int:
        return 1024  # Qwen3-Embedding-0.6B


class EmbeddingsService:
    """
    Serviço de embeddings.

    Usa factory para criar provider baseado na configuração.
    Provider pode ser trocado via DI para testes ou outros backends.
    """

    def __init__(self, provider: EmbeddingsProvider | None = None):
        if provider is None:
            from app.embeddings.factory import create_embedding_provider

            provider = create_embedding_provider()
        self._provider = provider

    def embed(self, text: str) -> list[float]:
        """Gera embedding para um texto."""
        return self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Gera embeddings para múltiplos textos (usa async internamente)."""
        return self._provider.embed_batch(texts)

    async def embed_batch_async(
        self, texts: list[str], max_workers: int = 4
    ) -> list[list[float]]:
        """Gera embeddings de forma assíncrona."""
        return await self._provider.embed_batch_async(texts, max_workers)

    @property
    def dimensions(self) -> int:
        return self._provider.dimensions


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcula similaridade cosseno entre dois vetores."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
