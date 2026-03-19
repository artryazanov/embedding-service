import asyncio
import json
import logging
import uuid

import websockets
from websockets.exceptions import ConnectionClosed

from config import settings
from engine import engine

logger = logging.getLogger(__name__)


async def websocket_worker_task():
    if not settings.reverb_app_key or not settings.reverb_host:
        logger.warning(
            "WebSocket disabled: REVERB_APP_KEY or REVERB_HOST not configured."
        )
        return

    worker_id = str(uuid.uuid4())
    worker_channel = f"worker-python-{worker_id}"
    ws_scheme = "wss" if settings.reverb_scheme == "https" else "ws"
    uri = f"{ws_scheme}://{settings.reverb_host}:{settings.reverb_port}/app/{settings.reverb_app_key}?protocol=7&client=python&version=1.0.0"  # noqa: E501

    dev_type = "gpu" if engine.is_gpu else "cpu"
    retry_delay = 1.0

    while True:
        try:
            logger.info(f"Connecting to Reverb WebSocket: {uri}")
            async with websockets.connect(uri) as websocket:
                logger.info("Successfully connected to WebSocket.")
                retry_delay = 1.0  # Reset delay upon successful connection

                # Subscribe
                await websocket.send(
                    json.dumps(
                        {
                            "event": "pusher:subscribe",
                            "data": {"channel": worker_channel},
                        }
                    )
                )

                # Ping task
                async def ping_loop():
                    while True:
                        try:
                            await websocket.send(
                                json.dumps(
                                    {
                                        "event": "worker_ping",
                                        "channel": worker_channel,
                                        "data": {
                                            "worker_id": worker_id,
                                            "channel": worker_channel,
                                            "type": dev_type,
                                            "model": settings.model_name,
                                        },
                                    }
                                )
                            )
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            break
                        except Exception:
                            break

                ping_task = asyncio.create_task(ping_loop())

                try:
                    async for message in websocket:
                        data = json.loads(message)
                        event = data.get("event", "")

                        if (
                            "VectorizeTaskEvent" in event
                            or "BatchVectorizeTaskEvent" in event
                        ):
                            payload_str = data.get("data", "{}")
                            payload = (
                                json.loads(payload_str)
                                if isinstance(payload_str, str)
                                else payload_str
                            )
                            req_id = payload.get("requestId")

                            try:
                                if "BatchVectorizeTaskEvent" in event:
                                    result = await engine.encode_batch_chunked_async(
                                        payload.get("items", [])
                                    )
                                else:
                                    vectors = await engine.encode_async(
                                        [payload.get("text", "")]
                                    )
                                    result = vectors[0]

                                await websocket.send(
                                    json.dumps(
                                        {
                                            "event": "worker_result",
                                            "channel": worker_channel,
                                            "data": {
                                                "request_id": req_id,
                                                "result": result,
                                            },
                                        }
                                    )
                                )
                            except Exception as e:
                                logger.error(f"Inference task execution error: {e}")

                except ConnectionClosed:
                    logger.warning("Connection closed by server.")
                finally:
                    ping_task.cancel()

        except asyncio.CancelledError:
            logger.info("Stopping WebSocket worker...")
            break
        except Exception as e:
            logger.error(
                f"WebSocket failure: {e}. Retrying in {retry_delay} seconds..."
            )
            await asyncio.sleep(retry_delay)
            retry_delay = min(
                retry_delay * 2, 60.0
            )  # Exponential backoff up to 60 seconds
