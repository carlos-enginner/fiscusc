"""Testes da Task 3: Camada de Configuração."""
import os

import pytest
from sqlalchemy import text


class TestSettings:
    def test_settings_loads_from_env(self, monkeypatch):
        """Settings deve carregar de variáveis de ambiente."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/testdb")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        # Importar após setar envs (forçar nova instância)
        from importlib import reload

        import app.core.config as cfg_module

        reload(cfg_module)
        cfg_module.get_settings.cache_clear()

        settings = cfg_module.Settings()
        assert "testdb" in settings.database_url
        assert settings.log_level == "DEBUG"

        # Restaurar
        cfg_module.get_settings.cache_clear()

    def test_settings_has_defaults(self):
        """Settings deve ter valores padrão sensatos."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200
        assert settings.top_k_results >= 5  # Can be overridden by .env
        assert settings.min_similarity_score == 0.3
        assert settings.ollama_base_url == "http://localhost:11434"

    def test_settings_log_level_uppercase(self, monkeypatch):
        """log_level deve ser normalizado para uppercase."""
        monkeypatch.setenv("LOG_LEVEL", "warning")

        from app.core.config import Settings

        settings = Settings()
        assert settings.log_level == "WARNING"

    def test_settings_invalid_log_level(self, monkeypatch):
        """log_level inválido deve lançar erro."""
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

        from app.core.config import Settings

        with pytest.raises(Exception):
            Settings()

    def test_settings_chunk_size_minimum(self):
        """chunk_size < 100 deve lançar erro."""
        from app.core.config import Settings

        with pytest.raises(Exception):
            Settings(chunk_size=50)

    def test_get_settings_is_cached(self):
        """get_settings deve retornar mesma instância."""
        from app.core.config import get_settings

        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_new_config_params_have_defaults(self):
        """Novos parâmetros de performance devem ter valores padrão."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.embedding_batch_size == 16
        assert settings.embedding_max_workers == 4
        assert settings.enable_embedding_cache is True
        assert settings.enable_incremental_ingest is True

    def test_batch_size_validation(self):
        """embedding_batch_size deve rejeitar valores inválidos."""
        from app.core.config import Settings

        # Deve rejeitar < 1
        with pytest.raises(Exception):
            Settings(embedding_batch_size=0)

        # Deve rejeitar > 64
        with pytest.raises(Exception):
            Settings(embedding_batch_size=65)

        # Deve aceitar valores válidos
        settings = Settings(embedding_batch_size=16)
        assert settings.embedding_batch_size == 16

    def test_max_workers_validation(self):
        """embedding_max_workers deve rejeitar valores inválidos."""
        from app.core.config import Settings

        # Deve rejeitar < 1
        with pytest.raises(Exception):
            Settings(embedding_max_workers=0)

        # Deve rejeitar > 32
        with pytest.raises(Exception):
            Settings(embedding_max_workers=33)

        # Deve aceitar valores válidos
        settings = Settings(embedding_max_workers=4)
        assert settings.embedding_max_workers == 4

    def test_embedding_provider_defaults_to_ollama(self):
        """embedding_provider deve ter valor padrão 'ollama'."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.embedding_provider == "ollama"

    def test_embedding_provider_accepts_fastembed(self):
        """embedding_provider deve aceitar 'fastembed'."""
        from app.core.config import Settings

        settings = Settings(embedding_provider="fastembed")
        assert settings.embedding_provider == "fastembed"

    def test_embedding_provider_normalizes_to_lowercase(self):
        """embedding_provider deve normalizar para lowercase."""
        from app.core.config import Settings

        settings = Settings(embedding_provider="FASTEMBED")
        assert settings.embedding_provider == "fastembed"

        settings2 = Settings(embedding_provider="OlLaMa")
        assert settings2.embedding_provider == "ollama"

    def test_embedding_provider_rejects_invalid_value(self):
        """embedding_provider deve rejeitar valores inválidos."""
        from app.core.config import Settings

        with pytest.raises(Exception):
            Settings(embedding_provider="invalid_provider")

    def test_fastembed_model_has_default(self):
        """fastembed_model deve ter valor padrão."""
        from app.core.config import Settings

        settings = Settings()
        assert settings.fastembed_model == "intfloat/multilingual-e5-small"


@pytest.mark.integration
class TestDatabaseConnection:
    def test_database_connection(self):
        """Deve conectar ao banco de dados."""
        from app.core.database import get_db

        db = next(get_db())
        result = db.execute(text("SELECT 1"))
        assert result.scalar() == 1

    def test_check_db_connection_returns_true(self):
        """check_db_connection deve retornar True quando banco acessível."""
        from app.core.database import check_db_connection

        assert check_db_connection() is True

    def test_get_db_is_generator(self):
        """get_db deve ser um generator."""
        import inspect

        from app.core.database import get_db

        assert inspect.isgeneratorfunction(get_db)
