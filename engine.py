import asyncio
import logging
import os
import time
import dataclasses
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


@dataclasses.dataclass(order=True)
class PrioritizedTask:
    priority: int
    timestamp: float
    texts: List[str] = dataclasses.field(compare=False)
    future: asyncio.Future = dataclasses.field(compare=False)


class EmbeddingEngine:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self._set_device()
        self.queue: Optional[asyncio.PriorityQueue] = None
        self.worker_task: Optional[asyncio.Task] = None

    def start_queue_worker(self):
        """Starts the background worker for processing the GPU inference queue."""
        if self.queue is None:
            self.queue = asyncio.PriorityQueue()
        self.worker_task = asyncio.create_task(self._process_queue())

    async def stop_queue_worker(self):
        """Stops the background worker when the application shuts down."""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            self.queue = None

    async def _process_queue(self):
        """A background loop that retrieves chunks one by one based on priority."""
        while True:
            try:
                task: PrioritizedTask = await self.queue.get()

                try:
                    if self.model is None:
                        raise RuntimeError("Model is not initialized")

                    result = await asyncio.to_thread(
                        self.encode, task.texts, batch_size=len(task.texts)
                    )
                    task.future.set_result(result)
                except Exception as e:
                    task.future.set_exception(e)
                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue worker error: {e}")

    def _set_device(self):
        if settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.device

        self.is_gpu = "cuda" in self.device
        logger.info(f"Selected inference device: {self.device}")

    def load(self) -> None:
        self._cleanup_memory()
        try:
            local_path = f"./models/{settings.model_name.replace('/', '_')}"
            load_source = (
                local_path if os.path.exists(local_path) else settings.model_name
            )

            logger.info(f"Loading model {load_source} on {self.device}...")
            self.model = SentenceTransformer(load_source, device=self.device)

            # Save locally if downloaded from hub
            if load_source == settings.model_name:
                os.makedirs("./models", exist_ok=True)
                self.model.save(local_path)

            self.model.max_seq_length = settings.max_seq_length
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.critical(f"Critical error loading model: {e}")
            raise

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._cleanup_memory()
        logger.info("Model unloaded from memory.")

    def _cleanup_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        # BGE-M3 doesn't require prefixes for base vectorization
        embeddings = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True
        )
        return embeddings.tolist()

    async def encode_async(
        self, texts: List[str], batch_size: int = 32
    ) -> List[List[float]]:
        """Asynchronous wrapper for fast/single requests."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        await self.queue.put(
            PrioritizedTask(
                priority=1, timestamp=time.time(), texts=texts, future=future
            )
        )
        return await future

    async def encode_batch_chunked_async(
        self, texts: List[str], chunk_size: int = 64
    ) -> List[List[float]]:
        """Splits a massive batch into chunks without blocking the event loop."""
        results = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]

            loop = asyncio.get_running_loop()
            future = loop.create_future()

            await self.queue.put(
                PrioritizedTask(
                    priority=2, timestamp=time.time(), texts=chunk, future=future
                )
            )

            chunk_result = await future
            results.extend(chunk_result)

        return results


# Global engine instance
engine = EmbeddingEngine()
