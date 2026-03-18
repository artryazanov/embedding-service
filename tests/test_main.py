from unittest.mock import AsyncMock, patch

from main import engine


def test_health_success(client):
    # Model should be mocked as loaded by the mock_engine_load_unload
    # fixture in conftest
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data


def test_health_unloaded(client):
    # Manually unload
    engine.unload()
    response = client.get("/health")
    assert response.status_code == 503
    assert "Model not loaded" in response.json()["detail"]


def test_vectorize_success(client):
    # Manually ensure it's loaded
    engine.model = True  # Dummy truthy value to pass "is None" check
    with patch(
        "main.engine.encode_async",
        new_callable=AsyncMock,
        return_value=[[0.1, 0.2, 0.3]],
    ) as mock_encode:
        response = client.post("/vectorize", json={"text": "hello world"})

        assert response.status_code == 200
        data = response.json()
        assert "vector" in data
        assert isinstance(data["vector"], list)
        assert data["vector"] == [0.1, 0.2, 0.3]
        mock_encode.assert_called_once_with(["hello world"])


def test_vectorize_model_unloaded(client):
    engine.unload()

    response = client.post("/vectorize", json={"text": "hello world"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is initializing."


def test_vectorize_batch_success(client):
    engine.model = True

    with patch(
        "main.engine.encode_batch_chunked_async",
        new_callable=AsyncMock,
        return_value=[[0.1, 0.1], [0.2, 0.2]],
    ) as mock_encode:
        response = client.post("/vectorize-batch", json={"items": ["text1", "text2"]})
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data
        assert len(data["vectors"]) == 2
        assert data["vectors"][0] == [0.1, 0.1]
        mock_encode.assert_called_once_with(["text1", "text2"], chunk_size=8)


def test_verify_token(client):
    with patch("main.settings.api_token", new="supersecret"):
        response = client.get("/health")
        # No token provided
        assert response.status_code == 401

        # Wrong token
        response = client.get("/health", headers={"Authorization": "Bearer badtoken"})
        assert response.status_code == 401

        # Correct token
        engine.model = True
        response = client.get(
            "/health", headers={"Authorization": "Bearer supersecret"}
        )
        assert response.status_code == 200


def test_vectorize_exception(client):
    engine.model = True
    with patch(
        "main.engine.encode_async",
        new_callable=AsyncMock,
        side_effect=Exception("Inference error"),
    ):
        response = client.post("/vectorize", json={"text": "error string"})
        assert response.status_code == 500
        assert response.json()["detail"] == "Inference error"


def test_vectorize_batch_model_unloaded(client):
    engine.unload()
    response = client.post("/vectorize-batch", json={"items": ["text"]})
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is initializing."


def test_vectorize_batch_exception(client):
    engine.model = True
    with patch(
        "main.engine.encode_batch_chunked_async",
        new_callable=AsyncMock,
        side_effect=Exception("Batch inference error"),
    ):
        response = client.post("/vectorize-batch", json={"items": ["text"]})
        assert response.status_code == 500
        assert response.json()["detail"] == "Batch inference error"
