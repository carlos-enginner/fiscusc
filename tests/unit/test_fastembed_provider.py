"""Testes unitários para FastEmbedProvider."""
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


# Mock do módulo fastembed antes de qualquer import
@pytest.fixture(autouse=True)
def mock_fastembed_module():
    """Mock global do módulo fastembed."""
    mock_fastembed = MagicMock()
    mock_text_embedding = MagicMock()
    
    def fake_embed(texts):
        """Gera embeddings fake determinísticos."""
        for text in texts:
            seed = sum(ord(c) for c in text) % 1000
            np.random.seed(seed)
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            yield vec
    
    mock_text_embedding.return_value.embed = fake_embed
    mock_fastembed.TextEmbedding = mock_text_embedding
    
    # Adiciona mock ao sys.modules
    sys.modules["fastembed"] = mock_fastembed
    yield mock_fastembed, mock_text_embedding
    # Remove após o teste
    if "fastembed" in sys.modules:
        del sys.modules["fastembed"]


class TestFastEmbedProvider:
    """Testes para FastEmbedProvider com mocks."""

    @pytest.fixture
    def provider(self, mock_fastembed_module):
        """Provider com modelo mockado."""
        # Força reimport do módulo para usar o mock
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        from app.embeddings.fastembed_provider import FastEmbedProvider
        return FastEmbedProvider(model="intfloat/multilingual-e5-small")

    def test_embed_returns_list_of_floats(self, provider):
        """embed() deve retornar lista de floats."""
        result = provider.embed("texto de teste")
        
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_embed_returns_correct_dimension(self, provider):
        """embed() deve retornar vetor com dimensão correta."""
        result = provider.embed("texto de teste")
        
        assert len(result) == 384

    def test_embed_is_deterministic(self, provider):
        """Mesmo texto deve gerar mesmo embedding."""
        v1 = provider.embed("texto determinístico")
        v2 = provider.embed("texto determinístico")
        
        assert v1 == v2

    def test_embed_different_texts_give_different_vectors(self, provider):
        """Textos diferentes devem gerar embeddings diferentes."""
        v1 = provider.embed("primeiro texto")
        v2 = provider.embed("segundo texto completamente diferente")
        
        assert v1 != v2

    def test_embed_batch_returns_correct_count(self, provider):
        """embed_batch() deve retornar número correto de embeddings."""
        texts = ["texto 1", "texto 2", "texto 3"]
        results = provider.embed_batch(texts)
        
        assert len(results) == 3

    def test_embed_batch_each_has_correct_dimension(self, provider):
        """Cada embedding em batch deve ter dimensão correta."""
        texts = ["texto 1", "texto 2", "texto 3"]
        results = provider.embed_batch(texts)
        
        assert all(len(v) == 384 for v in results)

    def test_embed_batch_empty_list_returns_empty(self, provider):
        """embed_batch([]) deve retornar lista vazia."""
        results = provider.embed_batch([])
        
        assert results == []

    def test_embed_batch_maintains_order(self, provider):
        """Ordem dos embeddings deve ser preservada."""
        texts = ["primeiro", "segundo", "terceiro"]
        batch_results = provider.embed_batch(texts)
        
        for i, text in enumerate(texts):
            single_result = provider.embed(text)
            assert batch_results[i] == single_result

    def test_dimensions_property(self, provider):
        """dimensions deve retornar valor detectado."""
        assert provider.dimensions == 384

    @pytest.mark.asyncio
    async def test_embed_batch_async_returns_correct_results(self, provider):
        """embed_batch_async() deve retornar mesmos resultados que síncrono."""
        texts = ["texto 1", "texto 2", "texto 3"]
        
        sync_results = provider.embed_batch(texts)
        async_results = await provider.embed_batch_async(texts)
        
        assert sync_results == async_results

    @pytest.mark.asyncio
    async def test_embed_batch_async_respects_max_workers(self, provider):
        """embed_batch_async() deve aceitar parâmetro max_workers."""
        texts = ["texto 1", "texto 2"]
        
        # Não deve lançar erro
        results = await provider.embed_batch_async(texts, max_workers=2)
        
        assert len(results) == 2


class TestFastEmbedProviderInitialization:
    """Testes de inicialização do FastEmbedProvider."""

    def test_uses_default_model_from_config(self, mock_fastembed_module):
        """Deve usar modelo da config quando não especificado."""
        _, mock_text_embedding = mock_fastembed_module
        mock_text_embedding.reset_mock()
        
        # Força reimport
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        with patch("app.core.config.get_settings") as mock_settings:
            mock_settings.return_value.fastembed_model = "custom/model"
            
            from app.embeddings.fastembed_provider import FastEmbedProvider
            FastEmbedProvider()
            
            mock_text_embedding.assert_called_with(model_name="custom/model")

    def test_uses_custom_model_when_specified(self, mock_fastembed_module):
        """Deve usar modelo especificado no construtor."""
        _, mock_text_embedding = mock_fastembed_module
        mock_text_embedding.reset_mock()
        
        # Força reimport
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        from app.embeddings.fastembed_provider import FastEmbedProvider
        FastEmbedProvider(model="sentence-transformers/all-MiniLM-L6-v2")
        
        mock_text_embedding.assert_called_with(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def test_auto_detects_dimensions(self, mock_fastembed_module):
        """Deve auto-detectar dimensões via embedding de teste."""
        mock_fastembed, mock_text_embedding = mock_fastembed_module
        mock_text_embedding.reset_mock()
        
        # Configura para retornar 768 dimensões
        def fake_embed_768(texts):
            for _ in texts:
                yield np.zeros(768)
        
        mock_text_embedding.return_value.embed = fake_embed_768
        
        # Força reimport
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        from app.embeddings.fastembed_provider import FastEmbedProvider
        provider = FastEmbedProvider(model="test-model")
        
        assert provider.dimensions == 768


class TestFastEmbedProviderIntegration:
    """Testes que verificam integração com interface EmbeddingsProvider."""

    def test_implements_embeddings_provider_interface(self, mock_fastembed_module):
        """FastEmbedProvider deve implementar EmbeddingsProvider."""
        from app.embeddings.service import EmbeddingsProvider
        
        # Força reimport
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        from app.embeddings.fastembed_provider import FastEmbedProvider
        
        assert issubclass(FastEmbedProvider, EmbeddingsProvider)

    def test_can_be_used_with_embeddings_service(self, mock_fastembed_module):
        """Deve ser usável com EmbeddingsService via DI."""
        # Força reimport
        if "app.embeddings.fastembed_provider" in sys.modules:
            del sys.modules["app.embeddings.fastembed_provider"]
        
        from app.embeddings.fastembed_provider import FastEmbedProvider
        from app.embeddings.service import EmbeddingsService
        
        provider = FastEmbedProvider(model="test-model")
        service = EmbeddingsService(provider=provider)
        
        result = service.embed("teste")
        
        assert len(result) == 384
        assert service.dimensions == 384
