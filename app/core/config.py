"""Configurações da aplicação via pydantic-settings."""
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações carregadas de variáveis de ambiente / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql://fiscusc:fiscusc@localhost:5432/fiscusc"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    llm_model: str = "qwen3:8b"

    # LLM Provider: "ollama" ou "gemini"
    llm_provider: str = "ollama"
    google_api_key: str = ""

    # App
    log_level: str = "INFO"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_results: int = 10
    min_similarity_score: float = 0.3

    # Performance Tuning
    embedding_batch_size: int = 16
    embedding_max_workers: int = 4
    enable_embedding_cache: bool = True
    enable_incremental_ingest: bool = True

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level deve ser um de: {allowed}")
        return upper

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v < 100:
            raise ValueError("chunk_size deve ser >= 100")
        return v

    @field_validator("top_k_results")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k_results deve ser >= 1")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v < 1 or v > 64:
            raise ValueError("embedding_batch_size deve estar entre 1 e 64")
        return v

    @field_validator("embedding_max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        if v < 1 or v > 32:
            raise ValueError("embedding_max_workers deve estar entre 1 e 32")
        return v


@lru_cache
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
