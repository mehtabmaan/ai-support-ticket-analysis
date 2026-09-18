"""
Configuration module for AI Support Ticket Analysis System.
Provides strongly-typed settings using pydantic-settings, OS-agnostic path resolution,
and dynamic runtime reference timestamp management.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_CSV_PATH = DATA_DIR / "support_tickets.csv"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Service & Network
    FASTAPI_HOST: str = "127.0.0.1"
    FASTAPI_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    LOG_LEVEL: str = "INFO"

    # Dataset file path
    CSV_FILE_PATH: Path = DEFAULT_CSV_PATH

    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "groq", "ollama", or "fallback"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:latest"
    LLM_TIMEOUT_SECONDS: float = 30.0

    # Anomaly Thresholds
    IQR_MULTIPLIER: float = 1.5
    SLA_BREACH_HOURS: float = 24.0
    RESPONSE_SLA_HOURS: float = 4.0


# Global settings instance
settings = Settings()

# Dynamic Runtime Reference Timestamp State
# This is NEVER hardcoded. It is calculated dynamically at startup
# from MAX(created_at, resolved_at) of the ingested dataset.
_RUNTIME_REFERENCE_NOW: Optional[datetime] = None
_RUNTIME_REFERENCE_SOURCE: str = "Uninitialized"


def set_reference_timestamp(timestamp: datetime, source_info: str = "Derived from dataset") -> None:
    """Dynamically set the reference timestamp based on the loaded dataset."""
    global _RUNTIME_REFERENCE_NOW, _RUNTIME_REFERENCE_SOURCE
    _RUNTIME_REFERENCE_NOW = timestamp
    _RUNTIME_REFERENCE_SOURCE = source_info


def get_reference_timestamp() -> datetime:
    """
    Retrieve the dynamic reference timestamp.
    Raises RuntimeError if accessed before dataset ingestion.
    """
    global _RUNTIME_REFERENCE_NOW
    if _RUNTIME_REFERENCE_NOW is None:
        raise RuntimeError(
            "Reference timestamp has not been initialized. "
            "Ensure dataset is loaded before accessing reference time."
        )
    return _RUNTIME_REFERENCE_NOW


def get_reference_timestamp_str() -> str:
    """Get the reference timestamp as a minute-precision string (YYYY-MM-DD HH:MM)."""
    return get_reference_timestamp().strftime("%Y-%m-%d %H:%M")


def get_reference_month_str() -> str:
    """Get the reference month string (YYYY-MM)."""
    return get_reference_timestamp().strftime("%Y-%m")


def get_reference_source() -> str:
    """Get explanation of how the reference timestamp was derived."""
    global _RUNTIME_REFERENCE_SOURCE
    return _RUNTIME_REFERENCE_SOURCE
