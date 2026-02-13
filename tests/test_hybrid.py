from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import BatchHybridVectorResponse, HybridVectorResponse, app, engine

client = TestClient(app)


@pytest.fixture
def mock_bge_engine():
    """
    Mocks the engine to simulate BGE-M3 loaded state.
    """
    original_model = engine.model
    original_bge = engine.bge_model
    original_profile = engine.profile

    # Mock BGE model
    mock_bge = MagicMock()
    # Mock encode output: dictionary with dense, sparse, colbert
    mock_bge.encode.return_value = {
        "dense_vecs": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "lexical_weights": [{"hello": 0.5}, {"world": 0.6}],
        "colbert_vecs": [
            np.array([[0.1, 0.1], [0.2, 0.2]]),
            np.array([[0.3, 0.3], [0.4, 0.4]]),
        ],
    }

    engine.bge_model = mock_bge
    engine.model = MagicMock()  # Mock standard model too just in case

    # Mock profile
    mock_profile = MagicMock()
    mock_profile.max_seq_length = 8192
    engine.profile = mock_profile

    yield mock_bge

    # Cleanup
    engine.model = original_model
    engine.bge_model = original_bge
    engine.profile = original_profile


def test_vectorize_hybrid_endpoint(mock_bge_engine):
    """
    Test POST /vectorize-hybrid with return_colbert=True
    """
    payload = {"text": "Hello world", "return_colbert": True}
    response = client.post("/vectorize-hybrid", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "hybrid_vector" in data
    vec = data["hybrid_vector"]

    # Verify structure matches HybridVector DTO
    assert "dense" in vec
    assert "sparse" in vec
    assert "colbert" in vec

    # Verify mocked values
    assert vec["dense"] == [0.1, 0.2, 0.3]
    assert vec["sparse"] == {"hello": 0.5}
    assert vec["colbert"] == [[0.1, 0.1], [0.2, 0.2]]

    # Verify engine call
    mock_bge_engine.encode.assert_called_once()
    args, kwargs = mock_bge_engine.encode.call_args
    assert args[0] == ["Hello world"]
    assert kwargs["return_colbert_vecs"] is True


def test_vectorize_hybrid_no_colbert(mock_bge_engine):
    """
    Test POST /vectorize-hybrid with return_colbert=False
    """
    # Mock return where colbert_vecs is None (as FlagEmbedding does when flag is False)
    mock_bge_engine.encode.return_value = {
        "dense_vecs": np.array([[0.1, 0.2, 0.3]]),
        "lexical_weights": [{"hello": 0.5}],
        "colbert_vecs": None,
    }

    payload = {"text": "Hello world", "return_colbert": False}
    response = client.post("/vectorize-hybrid", json=payload)
    assert response.status_code == 200

    data = response.json()
    vec = data["hybrid_vector"]

    assert vec["colbert"] is None

    args, kwargs = mock_bge_engine.encode.call_args
    assert kwargs["return_colbert_vecs"] is False


def test_vectorize_batch_hybrid(mock_bge_engine):
    """
    Test POST /vectorize-batch-hybrid
    """
    payload = {"items": ["Text 1", "Text 2"], "return_colbert": True}
    response = client.post("/vectorize-batch-hybrid", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "hybrid_vectors" in data
    assert len(data["hybrid_vectors"]) == 2

    v1 = data["hybrid_vectors"][0]
    assert v1["dense"] == [0.1, 0.2, 0.3]
    assert v1["colbert"] is not None


def test_hybrid_not_available():
    """
    Test that endpoints return 400/503 if bge_model is not loaded.
    """
    # Force engine state to have no BGE model
    original_bge = engine.bge_model
    engine.bge_model = None

    try:
        # Case 1: Model completely missing -> 503
        original_model = engine.model
        engine.model = None

        response = client.post("/vectorize-hybrid", json={"text": "test"})
        assert response.status_code == 503

        # Case 2: Standard model loaded, but not BGE -> 400
        engine.model = MagicMock()
        response = client.post("/vectorize-hybrid", json={"text": "test"})
        assert response.status_code == 400
        assert "available only for BGE-M3" in response.json()["detail"]

    finally:
        # Restore
        engine.bge_model = original_bge
        if "original_model" in locals():
            engine.model = original_model
