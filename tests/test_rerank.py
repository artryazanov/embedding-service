import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from main import engine, app
from fastapi.testclient import TestClient

client = TestClient(app)


@pytest.fixture
def mock_bge_model():
    # Since FlagEmbedding might not be installed in the test environment,
    # main.BGEM3FlagModel might not exist.
    # We can inject a dummy class into main module if it's missing, just so patch works
    import main

    if not hasattr(main, "BGEM3FlagModel"):
        main.BGEM3FlagModel = MagicMock()

    with patch("main.BGEM3FlagModel") as mock:
        mock_instance = MagicMock()

        # Mock behavior for encode
        def side_effect_encode(texts, **kwargs):
            # Return dict with colbert_vecs
            # colbert_vecs is a list of numpy arrays
            # let's say dimension is 4 (small)
            # length of text ~ number of tokens
            batch_size = len(texts)
            dim = 4
            seq_len = 5  # arbitrary

            vecs = []
            for _ in range(batch_size):
                # Create random vectors
                vecs.append(np.random.rand(seq_len, dim).astype(np.float32))

            return {
                "colbert_vecs": vecs,
                "dense_vecs": np.random.rand(batch_size, dim),
                "lexical_weights": [{} for _ in range(batch_size)],
            }

        mock_instance.encode.side_effect = side_effect_encode
        yield mock_instance


def test_rerank_success(mock_bge_model):
    # Setup engine with mock bge model
    engine.bge_model = mock_bge_model
    # Ensure profile is set
    engine.profile = MagicMock()
    engine.profile.max_seq_length = 512
    engine.model = (
        MagicMock()
    )  # To avoid "Model is initializing" error if bge is None (logic check)

    payload = {
        "query": "test query",
        "candidates": ["candidate 1", "candidate 2"],
        "batch_size": 2,
    }

    response = client.post("/rerank", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "scores" in data
    assert len(data["scores"]) == 2
    assert isinstance(data["scores"][0], float)


def test_rerank_no_model():
    # Simulate no BGE model loaded
    engine.bge_model = None
    engine.model = MagicMock()  # Some other model is loaded

    payload = {"query": "test query", "candidates": ["candidate 1"]}

    response = client.post("/rerank", json=payload)
    assert response.status_code == 400
    assert "available only for BGE-M3" in response.json()["detail"]


def test_rerank_service_unavailable():
    # Simulate initialization state
    engine.bge_model = None
    engine.model = None

    payload = {"query": "test query", "candidates": ["candidate 1"]}

    response = client.post("/rerank", json=payload)
    assert response.status_code == 503
    assert "Model is initializing" in response.json()["detail"]


def test_rerank_logic_correctness():
    """
    Verify MaxSim calculation with known vectors.
    """
    # Mock engine locally
    mock_bge = MagicMock()
    engine.bge_model = mock_bge
    engine.profile = MagicMock()
    engine.device = "cpu"  # Force CPU for test

    # Query: 1 token, dimensionality 2
    # Q = [[1, 0]]
    q_vecs = [np.array([[1.0, 0.0]])]

    # Candidate 1: 1 token, exact match
    # D1 = [[1, 0]] -> Dot = [[1]], Max = 1, Sum = 1
    d1 = np.array([[1.0, 0.0]])

    # Candidate 2: 1 token, orthogonal
    # D2 = [[0, 1]] -> Dot = [[0]], Max = 0, Sum = 0
    d2 = np.array([[0.0, 1.0]])

    # Candidate 3: 2 tokens
    # D3 = [[1, 0], [0, 1]]
    # Dot Q(1x2) @ D3.T(2x2) = [[1, 0]] @ [[1, 0], [0, 1]] = [[1, 0]]
    # Max along dim 1 (doc tokens) = 1
    # Sum = 1
    d3 = np.array([[1.0, 0.0], [0.0, 1.0]])

    # Encode Query Return
    mock_bge.encode.side_effect = [
        {"colbert_vecs": q_vecs},  # First call for query
        {"colbert_vecs": [d1, d2, d3]},  # Second call for candidates
    ]

    scores = engine.rerank_colbert("q", ["c1", "c2", "c3"])

    # Check close values
    assert pytest.approx(scores[0], 0.001) == 1.0
    assert pytest.approx(scores[1], 0.001) == 0.0
    assert pytest.approx(scores[2], 0.001) == 1.0
