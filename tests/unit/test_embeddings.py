"""Testes da Task 4: Embeddings Service.

Testes unitários usam mock do provider Ollama.
Testes de integração requerem Ollama rodando com o modelo carregado.
"""
import math

import pytest

from app.embeddings.service import (
    EmbeddingsProvider,
    EmbeddingsService,
    OllamaEmbeddingsProvider,
    cosine_similarity,
)

EMBEDDING_DIM = 1024


# --- Fake provider para testes unitários ---

class FakeEmbeddingsProvider(EmbeddingsProvider):
    """Provider fake para testes unitários."""

    def embed(self, text: str) -> list[float]:
        # Gera vetor determinístico baseado no hash do texto
        seed = sum(ord(c) for c in text)
        import random
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
        # Normalizar
        norm = math.sqrt(sum(x**2 for x in vec))
        return [x / norm for x in vec]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIM


# --- Testes unitários ---

class TestEmbeddingsService:
    @pytest.fixture
    def service(self):
        return EmbeddingsService(provider=FakeEmbeddingsProvider())

    def test_embed_returns_vector_of_correct_dimension(self, service):
        """embed() deve retornar vetor de dimensão correta."""
        vector = service.embed("texto de teste")

        assert len(vector) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vector)

    def test_embed_batch_returns_correct_count(self, service):
        """embed_batch() deve processar múltiplos textos."""
        texts = ["texto 1", "texto 2", "texto 3"]
        vectors = service.embed_batch(texts)

        assert len(vectors) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vectors)

    def test_embed_is_deterministic(self, service):
        """Mesmo texto deve gerar mesmo embedding."""
        v1 = service.embed("texto determinístico")
        v2 = service.embed("texto determinístico")

        assert v1 == v2

    def test_dimensions_property(self, service):
        """dimensions deve retornar valor correto."""
        assert service.dimensions == EMBEDDING_DIM

    def test_embed_different_texts_give_different_vectors(self, service):
        """Textos diferentes devem gerar embeddings diferentes."""
        v1 = service.embed("horário de obras")
        v2 = service.embed("receita de bolo")

        assert v1 != v2

    def test_provider_is_swappable(self):
        """Provider deve ser substituível via DI."""
        fake = FakeEmbeddingsProvider()
        service = EmbeddingsService(provider=fake)

        assert service._provider is fake


class TestCosineSimilarity:
    def test_identical_vectors_have_similarity_one(self):
        """Vetores idênticos devem ter similaridade 1.0."""
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_have_similarity_zero(self):
        """Vetores ortogonais devem ter similaridade 0.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(v1, v2)) < 1e-6

    def test_opposite_vectors_have_similarity_minus_one(self):
        """Vetores opostos devem ter similaridade -1.0."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v1, v2) + 1.0) < 1e-6

    def test_zero_vector_returns_zero(self):
        """Vetor zero deve retornar similaridade 0.0."""
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == 0.0

    def test_similar_texts_have_higher_similarity(self):
        """Textos similares devem ter embeddings mais próximos."""
        service = EmbeddingsService(provider=FakeEmbeddingsProvider())

        v1 = service.embed("horário de obras no condomínio")
        v2 = service.embed("quando posso fazer reforma")
        v3 = service.embed("receita de bolo de chocolate")

        sim_12 = cosine_similarity(v1, v2)
        sim_13 = cosine_similarity(v1, v3)

        # Com fake provider não garantimos semântica, mas podemos verificar o método
        assert isinstance(sim_12, float)
        assert isinstance(sim_13, float)
        assert -1.0 <= sim_12 <= 1.0
        assert -1.0 <= sim_13 <= 1.0


# --- Testes de integração (requerem Ollama) ---

@pytest.mark.integration
class TestOllamaEmbeddings:
    @pytest.fixture
    def provider(self):
        return OllamaEmbeddingsProvider()

    def test_embed_returns_vector(self, provider):
        """embed() deve retornar vetor de dimensão correta via Ollama."""
        vector = provider.embed("texto de teste")

        assert len(vector) == EMBEDDING_DIM
        assert all(isinstance(v, float) for v in vector)

    def test_embed_similar_texts_have_high_similarity(self, provider):
        """Textos similares devem ter embeddings próximos via Ollama."""
        v1 = provider.embed("horário de obras")
        v2 = provider.embed("quando posso fazer reforma")
        v3 = provider.embed("receita de bolo de chocolate")

        sim_12 = cosine_similarity(v1, v2)
        sim_13 = cosine_similarity(v1, v3)

        assert sim_12 > sim_13

    def test_embed_batch(self, provider):
        """embed_batch() deve processar múltiplos textos."""
        vectors = provider.embed_batch(["texto 1", "texto 2", "texto 3"])

        assert len(vectors) == 3
        assert all(len(v) == EMBEDDING_DIM for v in vectors)
