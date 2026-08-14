"""Application configuration and structured logging."""

from __future__ import annotations

from functools import lru_cache
import logging
import sys

from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "HHGoa-RAG"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Secrets / API Keys (loaded via env, never committed)
    GROQ_API_KEY: str = ""
    SARVAM_API_KEY: str = ""
    LLM_API_KEY: str = ""
    LLM_PROVIDER: str = ""
    EMBEDDING_MODEL: str = ""
    VECTOR_DB_PATH: str = "data/indexes"



@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog for structured logging."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer()
            if log_level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a bound structlog logger."""
    return structlog.get_logger(name)
