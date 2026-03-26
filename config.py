from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API & Model
    api_token: Optional[str] = None
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 8192
    chunk_size: int = 64

    # Device: auto, cpu, cuda
    device: Literal["auto", "cpu", "cuda"] = "auto"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
