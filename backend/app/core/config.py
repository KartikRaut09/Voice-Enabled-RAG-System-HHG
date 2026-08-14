"""Application configuration using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "HHGoa-RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # STT Provider (Phase 9)
    SARVAM_API_KEY: str = ""

    # LLM Provider (Phase 6 — selected via benchmarks)
    LLM_API_KEY: str = ""
    LLM_PROVIDER: str = ""  # Not locked to any provider

    # Embeddings (Phase 3 — selected via benchmarks)
    EMBEDDING_MODEL: str = ""  # Not locked to any model

    # Vector Database
    VECTOR_DB_PATH: str = "data/indexes"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
