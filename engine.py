import asyncio
import logging
import os
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self._set_device()
        self._lock = asyncio.Lock()

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
        async with self._lock:
            return await asyncio.to_thread(self.encode, texts, batch_size)

    async def encode_batch_chunked_async(
        self, texts: List[str], chunk_size: int = 8
    ) -> List[List[float]]:
        """Splits a massive batch into chunks without blocking the event loop."""
        results = []
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i : i + chunk_size]
            async with self._lock:
                chunk_result = await asyncio.to_thread(
                    self.encode, chunk, batch_size=chunk_size
                )
            results.extend(chunk_result)

            # Important: yield control back to the event loop.
            await asyncio.sleep(0.001)

        return results


# Global engine instance
engine = EmbeddingEngine()
