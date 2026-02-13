from unittest.mock import patch
import numpy as np

from main import engine


def test_vectorize_success(client, mock_sentence_transformer):
    # Ensure model is "loaded" (mocked)
    # The client fixture should have triggered startup, but we can verify or force it
    # Because client fixture depends on mock_sentence_transformer,
    # but currently it doesn't. Let's force load in the test or modify fixture.
    # Ideally, we rely on startup event.

    # We can also explicitly calling engine.load() to set the global
    engine.load()

    response = client.post("/vectorize", json={"text": "hello world", "is_query": True})
    assert response.status_code == 200
    data = response.json()
    assert "vector" in data
    assert isinstance(data["vector"], list)
    # mock returns [0.1, 0.2, 0.3]
    assert data["vector"] == [0.1, 0.2, 0.3]


def test_vectorize_model_unloaded(client):
    # Determine state: unload model
    engine.unload()

    response = client.post("/vectorize", json={"text": "hello world"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is currently training or initializing."


def test_vectorize_batch_success(client, mock_sentence_transformer):
    engine.load()

    # Mock return value for batch (main.py:102 calls encode on list)
    # The mock in conftest sets return_value for single call.
    # For batch, encode returns a list of embeddings (or ndarray).
    # Let's update the mock behavior for this test if needed,
    # but the default mock returns [0.1, 0.2, 0.3].
    # SentenceTransformer.encode(list) returns a list of embeddings (numpy arrays).

    # We need to make sure our mock returns a list of lists for batch inputs
    mock_sentence_transformer.encode.return_value = np.array([[0.1, 0.1], [0.2, 0.2]])

    response = client.post(
        "/vectorize-batch", json={"items": ["text1", "text2"], "is_query": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "vectors" in data
    assert len(data["vectors"]) == 2
    assert data["vectors"][0] == [0.1, 0.1]


def test_fine_tune_start(client, mock_trainer, mock_sentence_transformer):
    engine.load()

    # Mock BackgroundTasks so we don't executed worker immediately.
    # But FastAPI TestClient runs background tasks synchronously by default.
    # So train_worker will be called.
    # train_worker calls unload_model, then SentenceTransformerTrainer...

    # We need to mock 'datasets.Dataset.from_dict' or just ensure input is valid
    # train_worker will try to split lines.

    payload = {
        "text_content": "query1\tpassage1\nquery2\tpassage2",
        "model_name": "test_model",
    }

    # We need to ensure we don't delete models or modify files system significantly
    # train_worker does:
    # 1. unload_model()
    # 2. parses numbers
    # 3. loads train model
    # 4. runs trainer
    # 5. saves model
    # 6. reloads serving model

    with patch("main.train_worker") as mock_worker:
        response = client.post("/fine-tune", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"

        # Verify worker was added to background tasks
        mock_worker.assert_called_once()
        # Note: TestClient runs background tasks. If we patch it, it executes the mock.


def test_train_status_404(client):
    response = client.get("/train-status/non_existent_id")
    assert response.status_code == 404
