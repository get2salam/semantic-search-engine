"""
Configuration Management
========================
Centralized configuration using pydantic-settings with environment variable support.
Follows 12-factor app methodology for configuration management.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- Application ---
    app_name: str = Field(default="Semantic Search Engine", description="Application name")
    app_version: str = Field(default="1.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # --- Model ---
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformer model name",
    )
    use_faiss: bool = Field(default=True, description="Use FAISS for similarity search")
    normalize_embeddings: bool = Field(
        default=True, description="L2-normalize embeddings for cosine similarity"
    )
    default_top_k: int = Field(default=5, ge=1, le=100, description="Default number of results")
    max_top_k: int = Field(default=50, ge=1, le=500, description="Maximum allowed top_k")
    max_batch_size: int = Field(
        default=100, ge=1, le=1000, description="Maximum batch query size"
    )

    # --- API Server ---
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of worker processes")
    cors_origins: list[str] = Field(default=["*"], description="Allowed CORS origins")
    rate_limit_per_minute: int = Field(
        default=60, ge=1, description="Max requests per minute per client"
    )

    # --- Persistence ---
    index_path: Optional[str] = Field(
        default=None, description="Path to load a pre-built index on startup"
    )
    auto_save_path: Optional[str] = Field(
        default=None, description="Path to auto-save the index after modifications"
    )

    model_config = {
        "env_prefix": "SSE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings singleton.

    Returns the same Settings instance across calls to avoid
    re-reading environment variables on every request.
    """
    return Settings()
