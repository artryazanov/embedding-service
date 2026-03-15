from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from engine import EmbeddingEngine


@pytest.fixture
def mock_settings():
    with patch("engine.settings") as mock:
        mock.device = "auto"
        mock.model_name = "BAAI/bge-m3"
        mock.max_seq_length = 8192
        yield mock


def test_device_selection_auto_cpu(mock_settings):
    with patch("torch.cuda.is_available", return_value=False):
        engine = EmbeddingEngine()
        assert engine.device == "cpu"
        assert not engine.is_gpu


def test_device_selection_auto_cuda(mock_settings):
    with patch("torch.cuda.is_available", return_value=True):
        engine = EmbeddingEngine()
        assert engine.device == "cuda"
        assert engine.is_gpu


def test_device_selection_manual_cpu(mock_settings):
    mock_settings.device = "cpu"
    engine = EmbeddingEngine()
    assert engine.device == "cpu"


def test_engine_load_success(mock_settings):
    engine = EmbeddingEngine()
    with (
        patch("engine.SentenceTransformer") as MockST,
        patch("os.path.exists", return_value=True),
    ):

        mock_model = MagicMock()
        MockST.return_value = mock_model

        engine.load()
        assert engine.model is not None
        MockST.assert_called_once_with("./models/BAAI_bge-m3", device=engine.device)
        assert engine.model.max_seq_length == 8192


def test_engine_unload(mock_settings):
    engine = EmbeddingEngine()
    engine.model = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache") as mock_empty_cache,
        patch("torch.cuda.ipc_collect") as mock_ipc_collect,
    ):

        engine.unload()

        assert engine.model is None
        mock_empty_cache.assert_called_once()
        mock_ipc_collect.assert_called_once()


def test_engine_encode_success(mock_settings):
    engine = EmbeddingEngine()
    engine.model = MagicMock()

    # Mock return ndarray
    engine.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

    result = engine.encode(["test text"])
    assert result == [[0.1, 0.2, 0.3]]
    engine.model.encode.assert_called_once_with(
        ["test text"], batch_size=32, convert_to_numpy=True
    )


def test_engine_encode_unloaded():
    engine = EmbeddingEngine()
    engine.model = None
    with pytest.raises(RuntimeError) as exc_info:
        engine.encode(["test text"])
    assert "Model is not initialized" in str(exc_info.value)
