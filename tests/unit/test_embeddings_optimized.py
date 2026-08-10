"""Testes unitários para EmbeddingCache e IngestionMetrics.

Testes para as classes de otimização de embeddings:
- EmbeddingCache: cache LRU com TTL
- IngestionMetrics: métricas de pipeline de ingestão
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from app.embeddings.cache import EmbeddingCache
from app.rag.metrics import IngestionMetrics


# --- Testes para EmbeddingCache ---


class TestEmbeddingCache:
    """Testes para a classe EmbeddingCache."""

    @pytest.fixture
    def cache(self) -> EmbeddingCache:
        """Cache com configuração padrão."""
        return EmbeddingCache(max_size=100, ttl_seconds=3600.0)

    @pytest.fixture
    def small_cache(self) -> EmbeddingCache:
        """Cache pequeno para testar evição."""
        return EmbeddingCache(max_size=3, ttl_seconds=3600.0)

    @pytest.fixture
    def sample_embedding(self) -> list[float]:
        """Embedding de exemplo."""
        return [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_cache_miss_returns_none(self, cache: EmbeddingCache) -> None:
        """Cache vazio deve retornar None para qualquer conteúdo."""
        result = cache.get("texto inexistente")

        assert result is None
        assert cache._misses == 1
        assert cache._hits == 0

    def test_cache_hit_returns_embedding(
        self, cache: EmbeddingCache, sample_embedding: list[float]
    ) -> None:
        """Após put(), get() deve retornar o embedding armazenado."""
        content = "texto de teste"

        cache.put(content, sample_embedding)
        result = cache.get(content)

        assert result == sample_embedding
        assert cache._hits == 1
        assert cache._misses == 0
        assert len(cache) == 1

    def test_cache_lru_eviction(self, small_cache: EmbeddingCache) -> None:
        """Quando max_size é atingido, item mais antigo deve ser removido."""
        # Inserir 3 itens (max_size = 3)
        small_cache.put("item1", [1.0])
        small_cache.put("item2", [2.0])
        small_cache.put("item3", [3.0])

        assert len(small_cache) == 3

        # Inserir 4º item, deve evictar o item1 (mais antigo)
        small_cache.put("item4", [4.0])

        assert len(small_cache) == 3
        assert small_cache.get("item1") is None  # Evictado
        assert small_cache.get("item2") == [2.0]
        assert small_cache.get("item3") == [3.0]
        assert small_cache.get("item4") == [4.0]

    def test_cache_lru_eviction_respects_access_order(
        self, small_cache: EmbeddingCache
    ) -> None:
        """LRU deve considerar ordem de acesso, não de inserção."""
        small_cache.put("item1", [1.0])
        small_cache.put("item2", [2.0])
        small_cache.put("item3", [3.0])

        # Acessar item1, tornando-o mais recente
        small_cache.get("item1")

        # Inserir 4º item, deve evictar item2 (agora o mais antigo acessado)
        small_cache.put("item4", [4.0])

        assert small_cache.get("item1") == [1.0]  # Ainda presente
        assert small_cache.get("item2") is None  # Evictado
        assert small_cache.get("item3") == [3.0]
        assert small_cache.get("item4") == [4.0]

    def test_cache_ttl_expiration(
        self, sample_embedding: list[float]
    ) -> None:
        """Item expirado deve retornar None."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=100.0)
        content = "texto expirável"

        # Mock time.time() para controlar expiração
        with patch("app.embeddings.cache.time") as mock_time:
            # Tempo inicial: 1000
            mock_time.time.return_value = 1000.0
            cache.put(content, sample_embedding)

            # Tempo ainda dentro do TTL (1000 + 50 = 1050, TTL = 100)
            mock_time.time.return_value = 1050.0
            result = cache.get(content)
            assert result == sample_embedding

            # Tempo após expiração (1000 + 100 + 1 = 1101)
            mock_time.time.return_value = 1101.0
            result = cache.get(content)
            assert result is None

    def test_cache_expired_item_is_removed(
        self, sample_embedding: list[float]
    ) -> None:
        """Item expirado deve ser removido do cache ao ser acessado."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=100.0)

        with patch("app.embeddings.cache.time") as mock_time:
            mock_time.time.return_value = 1000.0
            cache.put("texto", sample_embedding)
            assert len(cache) == 1

            # Após expiração
            mock_time.time.return_value = 1101.0
            cache.get("texto")
            assert len(cache) == 0

    def test_cache_hit_rate_calculation(self, cache: EmbeddingCache) -> None:
        """Hit rate deve ser calculado corretamente."""
        # Sem operações
        assert cache.hit_rate == 0.0

        # 1 miss
        cache.get("inexistente")
        assert cache.hit_rate == 0.0  # 0 hits / 1 total

        # 1 put + 1 hit
        cache.put("existente", [1.0])
        cache.get("existente")
        assert cache.hit_rate == 0.5  # 1 hit / 2 total

        # Mais hits
        cache.get("existente")
        cache.get("existente")
        assert cache.hit_rate == 0.75  # 3 hits / 4 total

    def test_cache_hit_rate_after_clear(self, cache: EmbeddingCache) -> None:
        """Clear deve resetar as estatísticas."""
        cache.put("texto", [1.0])
        cache.get("texto")  # hit
        cache.get("outro")  # miss

        assert cache._hits == 1
        assert cache._misses == 1

        cache.clear()

        assert cache._hits == 0
        assert cache._misses == 0
        assert cache.hit_rate == 0.0
        assert len(cache) == 0

    def test_different_content_different_keys(
        self, cache: EmbeddingCache
    ) -> None:
        """Conteúdos diferentes devem ter keys diferentes no cache."""
        content1 = "primeiro texto"
        content2 = "segundo texto"
        embedding1 = [1.0, 2.0, 3.0]
        embedding2 = [4.0, 5.0, 6.0]

        cache.put(content1, embedding1)
        cache.put(content2, embedding2)

        assert len(cache) == 2
        assert cache.get(content1) == embedding1
        assert cache.get(content2) == embedding2
        assert cache.get(content1) != cache.get(content2)

    def test_same_content_same_key(self, cache: EmbeddingCache) -> None:
        """Mesmo conteúdo deve sobrescrever valor existente."""
        content = "texto repetido"

        cache.put(content, [1.0])
        cache.put(content, [2.0])

        assert len(cache) == 1
        assert cache.get(content) == [2.0]

    def test_unicode_content_handling(
        self, cache: EmbeddingCache, sample_embedding: list[float]
    ) -> None:
        """Cache deve funcionar com conteúdo Unicode."""
        content = "Texto com acentuação: ção, ã, é, ü 日本語 🎉"

        cache.put(content, sample_embedding)
        result = cache.get(content)

        assert result == sample_embedding


# --- Testes para IngestionMetrics ---


class TestIngestionMetrics:
    """Testes para a classe IngestionMetrics."""

    def test_metrics_default_values(self) -> None:
        """Valores padrão devem ser zero/None."""
        metrics = IngestionMetrics()

        assert metrics.extraction_ms == 0.0
        assert metrics.chunking_ms == 0.0
        assert metrics.embedding_ms == 0.0
        assert metrics.db_ms == 0.0
        assert metrics.total_ms == 0.0
        assert metrics.chunks_count == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.incremental_reused == 0
        assert metrics.chunks_per_sec == 0.0
        assert metrics.tokens_per_sec == 0.0
        assert metrics.start_time is not None
        assert metrics.end_time is None

    def test_metrics_finalize_calculates_total(self) -> None:
        """finalize() deve calcular total_ms corretamente."""
        start = datetime(2026, 8, 10, 10, 0, 0)
        end = datetime(2026, 8, 10, 10, 0, 2, 500000)  # 2.5 segundos depois

        metrics = IngestionMetrics(start_time=start, chunks_count=10)
        metrics.end_time = end
        metrics.finalize()

        assert metrics.total_ms == 2500.0  # 2.5 segundos = 2500ms

    def test_metrics_finalize_sets_end_time(self) -> None:
        """finalize() deve setar end_time se não estiver definido."""
        metrics = IngestionMetrics()
        assert metrics.end_time is None

        before = datetime.now()
        metrics.finalize()
        after = datetime.now()

        assert metrics.end_time is not None
        assert before <= metrics.end_time <= after

    def test_metrics_finalize_preserves_existing_end_time(self) -> None:
        """finalize() não deve alterar end_time já definido."""
        specific_end = datetime(2026, 8, 10, 15, 30, 0)
        metrics = IngestionMetrics()
        metrics.end_time = specific_end

        metrics.finalize()

        assert metrics.end_time == specific_end

    def test_chunks_per_sec_calculation(self) -> None:
        """chunks_per_sec deve ser calculado corretamente."""
        start = datetime(2026, 8, 10, 10, 0, 0)
        end = datetime(2026, 8, 10, 10, 0, 2)  # 2 segundos depois

        metrics = IngestionMetrics(start_time=start, chunks_count=100)
        metrics.end_time = end
        metrics.finalize()

        # 100 chunks / 2000ms * 1000 = 50 chunks/sec
        assert metrics.chunks_per_sec == 50.0

    def test_chunks_per_sec_zero_time(self) -> None:
        """chunks_per_sec deve ser 0 se total_ms for 0."""
        start = datetime(2026, 8, 10, 10, 0, 0)
        end = start  # Mesmo tempo

        metrics = IngestionMetrics(start_time=start, chunks_count=100)
        metrics.end_time = end
        metrics.finalize()

        assert metrics.total_ms == 0.0
        assert metrics.chunks_per_sec == 0.0

    def test_metrics_to_dict_structure(self) -> None:
        """to_dict() deve retornar estrutura correta."""
        start = datetime(2026, 8, 10, 10, 0, 0)
        end = datetime(2026, 8, 10, 10, 0, 1)

        metrics = IngestionMetrics(
            extraction_ms=100.0,
            chunking_ms=50.0,
            embedding_ms=200.0,
            db_ms=30.0,
            chunks_count=25,
            cache_hits=10,
            cache_misses=15,
            incremental_reused=5,
            tokens_per_sec=1000.0,
            start_time=start,
        )
        metrics.end_time = end
        metrics.finalize()

        result = metrics.to_dict()

        # Verificar estrutura de primeiro nível
        assert "timings" in result
        assert "counts" in result
        assert "throughput" in result
        assert "timestamps" in result

        # Verificar timings
        assert result["timings"]["extraction_ms"] == 100.0
        assert result["timings"]["chunking_ms"] == 50.0
        assert result["timings"]["embedding_ms"] == 200.0
        assert result["timings"]["db_ms"] == 30.0
        assert result["timings"]["total_ms"] == 1000.0

        # Verificar counts
        assert result["counts"]["chunks_count"] == 25
        assert result["counts"]["cache_hits"] == 10
        assert result["counts"]["cache_misses"] == 15
        assert result["counts"]["incremental_reused"] == 5

        # Verificar throughput
        assert result["throughput"]["chunks_per_sec"] == 25.0
        assert result["throughput"]["tokens_per_sec"] == 1000.0
        assert "cache_hit_rate" in result["throughput"]

        # Verificar timestamps
        assert result["timestamps"]["start_time"] == start.isoformat()
        assert result["timestamps"]["end_time"] == end.isoformat()

    def test_cache_hit_rate_in_dict(self) -> None:
        """cache_hit_rate em to_dict() deve ser calculado corretamente."""
        metrics = IngestionMetrics(cache_hits=30, cache_misses=70)

        result = metrics.to_dict()

        # 30 / (30 + 70) = 0.3
        assert result["throughput"]["cache_hit_rate"] == 0.3

    def test_cache_hit_rate_zero_operations(self) -> None:
        """cache_hit_rate deve ser 0 quando não há operações de cache."""
        metrics = IngestionMetrics(cache_hits=0, cache_misses=0)

        result = metrics.to_dict()

        assert result["throughput"]["cache_hit_rate"] == 0.0

    def test_cache_hit_rate_all_hits(self) -> None:
        """cache_hit_rate deve ser 1.0 quando todos são hits."""
        metrics = IngestionMetrics(cache_hits=100, cache_misses=0)

        result = metrics.to_dict()

        assert result["throughput"]["cache_hit_rate"] == 1.0

    def test_to_dict_with_none_end_time(self) -> None:
        """to_dict() deve funcionar com end_time None."""
        metrics = IngestionMetrics()

        result = metrics.to_dict()

        assert result["timestamps"]["end_time"] is None
        assert result["timestamps"]["start_time"] is not None
