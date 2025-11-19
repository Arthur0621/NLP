"""Application configuration utilities."""
from functools import lru_cache
import os
from pydantic import BaseModel


class Settings(BaseModel):
    """Centralized application settings."""

    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:postgres@localhost:5432/newsdb",
    )
    model_path: str = os.getenv("MODEL_PATH", "models/best_model")
    top_k_predictions: int = int(os.getenv("TOP_K", "3"))
    summary_model: str = os.getenv(
        "SUMMARY_MODEL",
        "VietAI/vit5-base-vietnews-summarization",
    )
    auto_crawl_enabled: bool = os.getenv("AUTO_CRAWL_ENABLED", "true").lower() == "true"
    auto_crawl_interval_minutes: int = int(os.getenv("AUTO_CRAWL_INTERVAL_MINUTES", "10"))
    auto_crawl_limit_per_feed: int = int(os.getenv("AUTO_CRAWL_LIMIT_PER_FEED", "50"))


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
