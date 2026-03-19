import pytest
from pydantic import ValidationError

from config import Settings


def test_default_config():
    settings = Settings()
    assert settings.model_name == "BAAI/bge-m3"
    assert settings.device == "auto"
    assert settings.max_seq_length == 8192


def test_override_config(monkeypatch):
    monkeypatch.setenv("model_name", "custom/model")
    monkeypatch.setenv("device", "cpu")
    monkeypatch.setenv("api_token", "supersecret123")

    settings = Settings()

    assert settings.model_name == "custom/model"
    assert settings.device == "cpu"
    assert settings.api_token == "supersecret123"


def test_invalid_device_raises_error(monkeypatch):
    monkeypatch.setenv("device", "invalid_device")
    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "Input should be 'auto', 'cpu' or 'cuda'" in str(exc_info.value)
