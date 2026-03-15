from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API & Model
    api_token: Optional[str] = None
    model_name: str = "BAAI/bge-m3"
    max_seq_length: int = 8192
    
    # Device: auto, cpu, cuda
    device: Literal["auto", "cpu", "cuda"] = "auto"
    
    # WebSocket / Reverb configuration
    reverb_app_key: Optional[str] = None
    reverb_host: str = "reverb"
    reverb_port: int = 8080
    reverb_scheme: Literal["http", "https"] = "http"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
