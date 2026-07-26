from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tg_token: SecretStr = SecretStr("")
    logging_level: str = "INFO"

    database_url: str | None = None
    postgres_host: str = "db"
    postgres_port: int = 5432
    postgres_db: str = "mediaarchivist"
    postgres_user: str = "mediaarchivist"
    postgres_password: SecretStr = SecretStr("")

    vision_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("VISION_API_KEY", "OPENROUTER_API_KEY"),
    )
    vision_base_url: str = "https://openrouter.ai/api/v1"
    vision_model: str = "google/gemini-3-flash-preview"
    vision_timeout_seconds: float = 120
    vision_max_retries: int = 3

    ollama_host: str = "http://ollama:11434"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = 1024

    whisper_enabled: bool = True
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    max_file_size: int = 20_000_000
    max_video_frames: int = Field(default=12, ge=4, le=24)
    index_bot_media: bool = False
    indexing_workers: int = Field(default=1, ge=1, le=4)
    membership_concurrency: int = Field(default=5, ge=1, le=20)
    search_candidates: int = Field(default=40, ge=10, le=200)
    search_results: int = Field(default=5, ge=1, le=10)
    max_cosine_distance: float = Field(default=0.65, ge=0, le=2)
    temp_dir: Path = Path("/tmp/mediaarchivist")

    @property
    def sqlalchemy_url(self) -> str:
        if self.database_url:
            return self.database_url
        raw_password = self.postgres_password.get_secret_value()
        if not raw_password:
            raise RuntimeError(
                "Missing required environment variable: POSTGRES_PASSWORD"
            )
        password = quote_plus(raw_password)
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def validate_runtime(self) -> None:
        missing: list[str] = []
        if not self.tg_token.get_secret_value():
            missing.append("TG_TOKEN")
        if not self.vision_api_key.get_secret_value():
            missing.append("VISION_API_KEY")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


@lru_cache
def get_settings() -> Settings:
    return Settings()
