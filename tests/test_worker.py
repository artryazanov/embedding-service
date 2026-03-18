import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from worker import websocket_worker_task


@pytest.fixture
def mock_settings():
    with patch("worker.settings") as mock:
        mock.reverb_app_key = "test_key"
        mock.reverb_host = "localhost"
        mock.reverb_scheme = "ws"
        mock.reverb_port = 8080
        yield mock


@pytest.fixture
def mock_engine():
    with patch("worker.engine") as mock:
        mock.is_gpu = False
        mock.encode_async = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock.encode_batch_chunked_async = AsyncMock(return_value=[[0.1], [0.2]])
        yield mock


@pytest.mark.asyncio
async def test_worker_disabled_if_no_key():
    with (
        patch("worker.settings") as mock_settings,
        patch("worker.logger.warning") as mock_warn,
    ):
        mock_settings.reverb_app_key = None

        await websocket_worker_task()
        mock_warn.assert_called_once()


class MockWebsocket:
    def __init__(self, messages):
        self.send = AsyncMock()
        self.messages = messages
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.messages):
            msg = self.messages[self.index]
            self.index += 1
            return msg
        raise Exception("Force Sleep Branch")


@pytest.mark.asyncio
async def test_worker_connects_and_processes(mock_settings, mock_engine):
    messages = [
        json.dumps(
            {
                "event": "VectorizeTaskEvent",
                "data": {"requestId": "req-123", "text": "test"},
            }
        )
    ]

    with (
        patch(
            "websockets.connect", return_value=MockWebsocket(messages)
        ) as mock_connect,
        patch("worker.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)),
    ):

        try:
            await websocket_worker_task()
        except asyncio.CancelledError:
            pass

        mock_connect.assert_called_once()
        mock_engine.encode_async.assert_called_once_with(["test"])


@pytest.mark.asyncio
async def test_worker_batch_process(mock_settings, mock_engine):
    mock_engine.encode_batch_chunked_async.return_value = [[0.1], [0.2]]

    messages = [
        json.dumps(
            {
                "event": "BatchVectorizeTaskEvent",
                "data": '{"requestId": "req-batch", "items": ["t1", "t2"]}',
            }
        )
    ]

    with (
        patch(
            "websockets.connect", return_value=MockWebsocket(messages)
        ) as mock_connect,
        patch("worker.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)),
    ):

        try:
            await websocket_worker_task()
        except asyncio.CancelledError:
            pass

        mock_connect.assert_called_once()
        mock_engine.encode_batch_chunked_async.assert_called_once_with(
            ["t1", "t2"], chunk_size=64
        )


from websockets.exceptions import ConnectionClosed


@pytest.mark.asyncio
async def test_worker_connection_closed(mock_settings, mock_engine):
    class MockWebsocketClosed(MockWebsocket):
        async def __anext__(self):
            raise ConnectionClosed(None, None)

    with (
        patch(
            "websockets.connect",
            side_effect=[MockWebsocketClosed([]), Exception("Stop loop!")],
        ) as mock_connect,
        patch("worker.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)),
        patch("worker.logger.warning") as mock_warn,
    ):
        try:
            await websocket_worker_task()
        except asyncio.CancelledError:
            pass
        mock_warn.assert_any_call("Connection closed by server.")


@pytest.mark.asyncio
async def test_worker_inference_error(mock_settings, mock_engine):
    messages = [
        json.dumps(
            {
                "event": "VectorizeTaskEvent",
                "data": {"requestId": "req-123", "text": "test_error"},
            }
        )
    ]
    mock_engine.encode_async.side_effect = Exception("Inference failed")

    with (
        patch("websockets.connect", return_value=MockWebsocket(messages)),
        patch("worker.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)),
        patch("worker.logger.error") as mock_error,
    ):
        try:
            await websocket_worker_task()
        except asyncio.CancelledError:
            pass
        mock_error.assert_any_call("Inference task execution error: Inference failed")


@pytest.mark.asyncio
async def test_worker_outer_exception(mock_settings, mock_engine):
    with (
        patch("websockets.connect", side_effect=Exception("Outer error")),
        patch("worker.asyncio.sleep", AsyncMock(side_effect=asyncio.CancelledError)),
        patch("worker.logger.error") as mock_error,
    ):
        try:
            await websocket_worker_task()
        except asyncio.CancelledError:
            pass
        mock_error.assert_any_call(
            "WebSocket failure: Outer error. Retrying in 1.0 seconds..."
        )
