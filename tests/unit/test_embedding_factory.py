"""Testes unitários para factory de embedding providers."""
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_fastembed_module():
    """Mock do módulo fastembed."""
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

    sys.modules["fastembed"] = mock_fastembed
    yield mock_fastembed, mock_text_embedding
    if "fastembed" in sys.modules:
        del sys.modules["fastembed"]


@pytest.fixture
def clear_fastembed_cache():
    """Limpa cache do módulo fastembed_provider."""
    yield
    if "app.embeddings.fastembed_provider" in sys.modules:
        del sys.modules["app.embeddings.fastembed_provider"]


class TestCreateEmbeddingProvider:
    """Testes para create_embedding_provider()."""

    def test_returns_ollama_provider_when_ollama(self):
        """Deve retornar OllamaEmbeddingsProvider quando embedding_provider='ollama'."""
        from app.embeddings.service import OllamaEmbeddingsProvider

        mock_settings = MagicMock()
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "qwen3-embedding:0.6b"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.factory import create_embedding_provider

            provider = create_embedding_provider(mock_settings)

            assert isinstance(provider, OllamaEmbeddingsProvider)

    def test_returns_fastembed_provider_when_fastembed(
        self, mock_fastembed_module, clear_fastembed_cache
    ):
        """Deve retornar FastEmbedProvider quando embedding_provider='fastembed'."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "fastembed"
        mock_settings.fastembed_model = "intfloat/multilingual-e5-small"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.factory import create_embedding_provider
            from app.embeddings.fastembed_provider import FastEmbedProvider

            provider = create_embedding_provider(mock_settings)

            assert isinstance(provider, FastEmbedProvider)

    def test_uses_get_settings_when_settings_is_none(self):
        """Deve usar get_settings() quando settings=None."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "qwen3-embedding:0.6b"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ) as mock_get_settings:
            from app.embeddings.factory import create_embedding_provider

            create_embedding_provider(settings=None)

            mock_get_settings.assert_called_once()

    def test_does_not_call_get_settings_when_settings_provided(self):
        """Não deve chamar get_settings() quando settings é fornecido."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "qwen3-embedding:0.6b"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ) as mock_get_settings:
            from app.embeddings.factory import create_embedding_provider

            create_embedding_provider(settings=mock_settings)

            mock_get_settings.assert_not_called()

    def test_raises_error_for_invalid_provider(self):
        """Deve levantar ValueError para provider inválido."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "invalid_provider"

        from app.embeddings.factory import create_embedding_provider

        with pytest.raises(ValueError) as exc_info:
            create_embedding_provider(mock_settings)

        assert "invalid_provider" in str(exc_info.value)
        assert "fastembed" in str(exc_info.value)
        assert "ollama" in str(exc_info.value)

    def test_ollama_provider_uses_correct_model(self):
        """OllamaEmbeddingsProvider deve usar o modelo da config."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "custom-embedding-model"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.factory import create_embedding_provider

            provider = create_embedding_provider(mock_settings)

            assert provider._model == "custom-embedding-model"

    def test_fastembed_provider_uses_correct_model(
        self, mock_fastembed_module, clear_fastembed_cache
    ):
        """FastEmbedProvider deve usar o modelo da config."""
        _, mock_text_embedding = mock_fastembed_module

        mock_settings = MagicMock()
        mock_settings.embedding_provider = "fastembed"
        mock_settings.fastembed_model = "custom/fastembed-model"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.factory import create_embedding_provider

            provider = create_embedding_provider(mock_settings)

            assert provider._model_name == "custom/fastembed-model"


class TestEmbeddingsServiceWithFactory:
    """Testes de integração entre EmbeddingsService e factory."""

    def test_service_uses_factory_when_provider_is_none(self):
        """EmbeddingsService deve usar factory quando provider=None."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "ollama"
        mock_settings.embedding_model = "qwen3-embedding:0.6b"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.service import (
                EmbeddingsService,
                OllamaEmbeddingsProvider,
            )

            service = EmbeddingsService(provider=None)

            assert isinstance(service._provider, OllamaEmbeddingsProvider)

    def test_service_uses_provided_provider_when_given(self):
        """EmbeddingsService deve usar provider fornecido se especificado."""
        from app.embeddings.service import EmbeddingsProvider

        mock_provider = MagicMock(spec=EmbeddingsProvider)
        mock_provider.dimensions = 512

        from app.embeddings.service import EmbeddingsService

        service = EmbeddingsService(provider=mock_provider)

        assert service._provider is mock_provider

    def test_service_with_fastembed_via_factory(
        self, mock_fastembed_module, clear_fastembed_cache
    ):
        """EmbeddingsService deve usar FastEmbed quando config indica."""
        mock_settings = MagicMock()
        mock_settings.embedding_provider = "fastembed"
        mock_settings.fastembed_model = "intfloat/multilingual-e5-small"

        with patch(
            "app.embeddings.factory.get_settings", return_value=mock_settings
        ):
            from app.embeddings.fastembed_provider import FastEmbedProvider
            from app.embeddings.service import EmbeddingsService

            service = EmbeddingsService(provider=None)

            assert isinstance(service._provider, FastEmbedProvider)
