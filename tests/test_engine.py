from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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


@pytest.mark.asyncio
async def test_engine_encode_async_success(mock_settings):
    engine = EmbeddingEngine()
    engine.model = MagicMock()
    # Mock synchronous method
    engine.encode = MagicMock(return_value=[[0.1, 0.2, 0.3]])

    engine.start_queue_worker()
    result = await engine.encode_async(["test text_async"], batch_size=32)
    await engine.stop_queue_worker()

    assert result == [[0.1, 0.2, 0.3]]
    engine.encode.assert_called_once_with(["test text_async"], batch_size=1)


@pytest.mark.asyncio
async def test_engine_encode_batch_chunked_async_success(mock_settings):
    engine = EmbeddingEngine()
    engine.model = MagicMock()

    # Each chunk call returns a list of vectors
    engine.encode = MagicMock(side_effect=[[[0.1]], [[0.2]]])

    engine.start_queue_worker()
    # We do a batch of 2 elements, chunk size 1
    result = await engine.encode_batch_chunked_async(["a", "b"], chunk_size=1)
    await engine.stop_queue_worker()

    assert result == [[0.1], [0.2]]
    assert engine.encode.call_count == 2
    engine.encode.assert_any_call(["a"], batch_size=1)
    engine.encode.assert_any_call(["b"], batch_size=1)


def test_engine_load_saves_locally(mock_settings):
    # Cover the condition where load_source == settings.model_name
    engine = EmbeddingEngine()
    with (
        patch("engine.SentenceTransformer") as MockST,
        patch("os.path.exists", return_value=False),
        patch("os.makedirs") as mock_makedirs,
    ):
        mock_model = MagicMock()
        MockST.return_value = mock_model
        engine.load()

        mock_makedirs.assert_called_once_with("./models", exist_ok=True)
        mock_model.save.assert_called_once_with("./models/BAAI_bge-m3")


def test_engine_load_exception(mock_settings):
    engine = EmbeddingEngine()
    with patch("engine.SentenceTransformer", side_effect=Exception("Test error")):
        with pytest.raises(Exception, match="Test error"):
            engine.load()
