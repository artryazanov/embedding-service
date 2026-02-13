import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, cast

import datasets
import torch
import torch.nn as nn
from fastapi import BackgroundTasks, FastAPI, HTTPException, status
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from pydantic import BaseModel
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from transformers import BitsAndBytesConfig, TrainerCallback

# Try to import FlagEmbedding for BGE-M3 hybrid features
try:
    from FlagEmbedding import BGEM3FlagModel

    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Configuration Strategy Pattern ---


class ModelProfile:
    """Strategy interface for model-specific configurations."""

    def __init__(self, name: str, max_seq_length: int = 512):
        self.name = name
        self.max_seq_length = max_seq_length

    def format_text(self, text: str, is_query: bool = True) -> str:
        """Formats text according to model requirements (e.g. adding prefixes)."""
        return text


class E5ModelProfile(ModelProfile):
    """Configuration for E5 models which require prefixes."""

    def format_text(self, text: str, is_query: bool = True) -> str:
        prefix = "query: " if is_query else "passage: "
        return prefix + text


class GenericModelProfile(ModelProfile):
    """Default configuration for models that don't need specific formatting."""

    pass


def detect_model_profile(model_name: str) -> ModelProfile:
    """Factory function to select the correct profile based on model name."""
    lower_name = model_name.lower()
    if "e5" in lower_name:
        return E5ModelProfile(model_name, max_seq_length=512)
    elif "bge-m3" in lower_name:
        # BGE-M3 supports up to 8192, but we default to 8192 or less depending on resources
        return GenericModelProfile(model_name, max_seq_length=8192)
    else:
        return GenericModelProfile(model_name)


# --- DTOs ---


class TextRequest(BaseModel):
    text: str
    is_query: bool = True


class BatchTextRequest(BaseModel):
    items: List[str]
    is_query: bool = False


class VectorResponse(BaseModel):
    vector: List[float]


class BatchVectorResponse(BaseModel):
    vectors: List[List[float]]


# --- New DTOs for Hybrid Search (BGE-M3) ---


class HybridTextRequest(BaseModel):
    text: str
    return_colbert: bool = False


class HybridBatchTextRequest(BaseModel):
    items: List[str]
    return_colbert: bool = False


class HybridVector(BaseModel):
    dense: List[float]
    sparse: Dict[str, float]
    colbert: Optional[List[List[float]]] = None


class HybridVectorResponse(BaseModel):
    hybrid_vector: HybridVector


class BatchHybridVectorResponse(BaseModel):
    hybrid_vectors: List[HybridVector]


# --- Fine-tuning DTOs ---


class FineTuneExample(BaseModel):
    query: str
    pos: List[str]
    neg: List[str]


class FineTuneRequest(BaseModel):
    examples: List[FineTuneExample]
    num_epochs: int = 1
    batch_size: int = 16
    warmup_steps: float = 0.1
    learning_rate: float = 2e-5


class TrainRequest(BaseModel):
    text_content: str
    model_name: str


class LoraTrainRequest(BaseModel):
    text_content: str
    model_name: str
    # LoRA parameters
    r: int = 16  # Adapter rank
    lora_alpha: int = 32  # Scaling
    lora_dropout: float = 0.05
    use_qlora: bool = True  # Use 4-bit quantization


# --- Engine ---


class EmbeddingEngine:
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.bge_model: Optional[Any] = None  # Slot for BGEM3FlagModel
        self.profile: Optional[ModelProfile] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name_env = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")

    def load(self):
        self._cleanup_memory()
        try:
            self.profile = detect_model_profile(self.model_name_env)
            local_path = f"./models/{self.model_name_env}"
            load_source = (
                local_path if os.path.exists(local_path) else self.model_name_env
            )

            logger.info(f"Loading model from {load_source} to {self.device}...")

            # Logic to choose loader
            # If it is BGE-M3 and we have FlagEmbedding library, load via it
            if "bge-m3" in self.profile.name.lower() and FLAG_EMBEDDING_AVAILABLE:
                logger.info("Initializing BGEM3FlagModel for hybrid capabilities...")
                self.bge_model = BGEM3FlagModel(
                    load_source,
                    use_fp16=(self.device == "cuda"),
                    device=self.device,
                )

                # To support legacy methods and fine-tuning (which requires SentenceTransformer),
                # we also load SentenceTransformer as self.model.
                # NOTE: This doubles memory usage if loaded simultaneously.
                # Ideally, one should migrate completely, but for backward compatibility
                # we keep both for now if needed.
                self.model = SentenceTransformer(load_source, device=self.device)
            else:
                self.model = SentenceTransformer(load_source, device=self.device)
                self.bge_model = None

            if self.model:
                # Save to local cache if downloaded from Hub
                if load_source == self.model_name_env:
                    self.model.save(local_path)
                self.model.max_seq_length = self.profile.max_seq_length

            logger.info("Model loaded successfully.")

        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise e

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.bge_model is not None:
            del self.bge_model
            self.bge_model = None
        self._cleanup_memory()
        logger.info("Model unloaded.")

    def _cleanup_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def encode(
        self, texts: List[str], is_query: bool = True, batch_size: int = 32
    ) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # Apply formatting (prefixes) if needed
        formatted_texts = [self.profile.format_text(t, is_query) for t in texts]

        embeddings = self.model.encode(
            formatted_texts, batch_size=batch_size, convert_to_numpy=True
        )
        return embeddings.tolist()

    def encode_hybrid(
        self, texts: List[str], return_colbert: bool = False, batch_size: int = 12
    ) -> List[Dict]:
        """
        New method specifically for BGE-M3, returning 3 types of vectors.
        """
        if self.bge_model is None:
            raise RuntimeError(
                "BGE-M3 model is not loaded or FlagEmbedding is missing."
            )

        # FlagEmbedding handles tokenization and batching internally
        output = self.bge_model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.profile.max_seq_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=return_colbert,
        )

        results = []
        count = len(texts)

        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]
        # If return_colbert=False, output['colbert_vecs'] is None
        colbert_vecs = output.get("colbert_vecs")

        for i in range(count):
            # Prepare colbert data only if requested and available
            c_vecs = None
            if return_colbert and colbert_vecs is not None:
                c_vecs = colbert_vecs[i].tolist()

            results.append(
                {
                    "dense": dense_vecs[i].tolist(),
                    "sparse": lexical_weights[i],
                    "colbert": c_vecs,
                }
            )

        return results


engine = EmbeddingEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        engine.load()
    except Exception:
        pass  # Logged in load()
    yield
    # Shutdown
    engine.unload()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if engine.model is None and engine.bge_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return {
        "status": "ok",
        "model": engine.profile.name if engine.profile else "unknown",
    }


@app.post("/vectorize", response_model=VectorResponse)
async def vectorize(req: TextRequest):
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training or initializing.",
        )

    try:
        embedding = engine.encode([req.text], is_query=req.is_query)[0]
        return {"vector": embedding}
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorize-batch", response_model=BatchVectorResponse)
async def vectorize_batch(req: BatchTextRequest):
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training or initializing.",
        )

    try:
        # BGE-M3 supports long contexts but consumes more memory.
        # Can dynamically change batch_size depending on engine.profile.max_seq_length,
        # but for now execute 64 as a reasonable default for 4080.
        vectors = engine.encode(req.items, is_query=req.is_query, batch_size=64)
        return {"vectors": vectors}
    except Exception as e:
        logger.error(f"Batch vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Training Logic ---

jobs: Dict[str, Dict[str, Any]] = {}


class JobProgressCallback(TrainerCallback):
    def __init__(self, job_id):
        self.job_id = job_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.job_id in jobs and state.max_steps > 0:
            progress = int((state.global_step / state.max_steps) * 100)
            jobs[self.job_id]["progress"] = progress
            jobs[self.job_id]["steps"] = state.global_step


def train_worker(job_id: str, req: TrainRequest):
    """
    Training worker.
    Note: Training BGE-M3 might require a different loss function or approach,
    but here we preserve the Contrastive Learning logic, which is universal.
    """
    try:
        jobs[job_id]["status"] = "running"
        logger.info(f"[{job_id}] Starting training process...")

        logger.info(f"[{job_id}] Unloading serving model...")
        engine.unload()
        engine._cleanup_memory()  # Ensure explicit cleanup

        # Parse data
        raw_lines = req.text_content.splitlines()
        train_examples = []

        # IMPORTANT: When training, we must also consider whether prefixes are needed.
        # Usually E5 is fine-tuned with prefixes, while BGE is not.
        # For simplicity here, we assume the user provides "clean" text,
        # and prefixes are added by the code below (if we were using engine.profile).
        # BUT: since the model is unloaded, profile might be unavailable.
        # Here we follow the old logic: accept data as is, but form InputExample
        # with explicit prefix addition only if we are sure about the architecture.

        # For reliability, training is better conducted with the same rule:
        # Determine the base architecture.
        base_profile = detect_model_profile(engine.model_name_env)

        q_prefix = base_profile.query_prefix
        p_prefix = base_profile.passage_prefix

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                # Add prefixes to training data ONLY if the model requires it
                q = f"{q_prefix}{parts[0].strip()}"
                p = f"{p_prefix}{parts[1].strip()}"
                train_examples.append(InputExample(texts=[q, p]))

        if not train_examples:
            raise ValueError("No valid pairs found")

        logger.info(
            f"[{job_id}] Data prepared: {len(train_examples)} pairs. "
            f"Prefixes used: {base_profile.requires_prefix}"
        )

        # Prepare columns for dataset, ensuring 'texts' is not None
        params_list = [e.texts for e in train_examples if e.texts is not None]
        train_dataset = datasets.Dataset.from_dict(
            {
                "sentence_0": [p[0] for p in params_list],
                "sentence_1": [p[1] for p in params_list],
            }
        )

        # Load model for training
        load_path = f"./models/{engine.model_name_env}"
        if not os.path.exists(load_path):
            load_path = engine.model_name_env

        logger.info(f"[{job_id}] Loading base model from {load_path}...")
        # FIX: Force device="cuda:0" to avoid ambiguity.
        # This helps prevent DataParallel usage.
        train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train_model = SentenceTransformer(load_path, device=train_device)
        train_model.max_seq_length = base_profile.max_seq_length  # Use correct length

        # Config
        output_dir = f"./models/{req.model_name}-tmp"
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=32,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            optim="adafactor",
        )

        # FIX: Force single-GPU (n_gpu=1) to prevent Trainer from using DataParallel.
        if torch.cuda.device_count() > 1:
            logger.info(
                f"[{job_id}] Multiple GPUs detected. Forcing single-GPU (n_gpu=1)."
            )
            args._n_gpu = 1

        loss = losses.MultipleNegativesRankingLoss(train_model)

        trainer = SentenceTransformerTrainer(
            model=train_model,
            args=args,
            train_dataset={"default": train_dataset},
            loss={"default": cast(nn.Module, loss)},
            callbacks=[JobProgressCallback(job_id)],
        )

        trainer.train()

        save_path = f"./models/{req.model_name}"
        if not os.path.exists("./models"):
            os.makedirs("./models")

        train_model.save(save_path)
        logger.info(f"[{job_id}] Training finished. Saved to {save_path}")

        del train_model
        engine._cleanup_memory()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"[{job_id}] Training failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

    finally:
        logger.info(f"[{job_id}] Reloading serving model...")
        try:
            engine.load()
        except Exception:
            pass


def train_lora_worker(job_id: str, req: LoraTrainRequest):
    try:
        jobs[job_id]["status"] = "running"
        logger.info(f"[{job_id}] Starting LoRA training...")

        # 1. Clear memory
        engine.unload()
        engine._cleanup_memory()

        # 2. Prepare data (copying logic from train_worker)
        # --- (Start data prep block) ---
        raw_lines = req.text_content.splitlines()
        train_examples = []

        # Determine base model profile for prefixes
        base_profile = detect_model_profile(engine.model_name_env)
        q_prefix = base_profile.query_prefix
        p_prefix = base_profile.passage_prefix

        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                q = f"{q_prefix}{parts[0].strip()}"
                p = f"{p_prefix}{parts[1].strip()}"
                train_examples.append(InputExample(texts=[q, p]))

        if not train_examples:
            raise ValueError("No valid pairs found")

        params_list = [e.texts for e in train_examples if e.texts is not None]
        train_dataset = datasets.Dataset.from_dict(
            {
                "sentence_0": [p[0] for p in params_list],
                "sentence_1": [p[1] for p in params_list],
            }
        )
        # --- (End data prep block) ---

        # 3. Configure Q-LoRA (4-bit) or standard LoRA
        load_path = f"./models/{engine.model_name_env}"
        if not os.path.exists(load_path):
            load_path = engine.model_name_env

        model_kwargs: Dict[str, Any] = {}
        if req.use_qlora:
            logger.info(f"[{job_id}] Using Q-LoRA (4-bit quantization)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["torch_dtype"] = torch.float16

        logger.info(f"[{job_id}] Loading base model for LoRA...")

        # Load model with quantization kwargs
        # FIX: Force device="cuda:0" to avoid ambiguity.
        # This helps prevent DataParallel usage.
        train_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        train_model = SentenceTransformer(
            load_path,
            device=train_device,
            model_kwargs=model_kwargs,
        )
        train_model.max_seq_length = base_profile.max_seq_length  # Use correct length

        # 4. Apply LoRA to the internal transformer model
        # SentenceTransformer wraps HuggingFace model in modules.
        # Usually the first module is Transformer.
        from transformers import PreTrainedModel

        transformer_module = cast(
            PreTrainedModel, train_model._first_module().auto_model
        )

        # Prepare for k-bit training (if QLoRA)
        if req.use_qlora:
            transformer_module = prepare_model_for_kbit_training(transformer_module)

        # LoRA Config
        # target_modules depends on architecture,
        # but for BERT/RoBERTa (E5/BGE) it's query/key/value
        peft_config = LoraConfig(
            r=req.r,
            lora_alpha=req.lora_alpha,
            lora_dropout=req.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=[
                "query",
                "key",
                "value",
                "dense",
            ],  # Extended set for better quality
        )

        # Wrap model in PEFT
        peft_model = get_peft_model(
            transformer_module,
            peft_config,
        )
        train_model._first_module().auto_model = peft_model
        peft_model.print_trainable_parameters()

        # 5. Trainer parameters
        output_dir = f"./models/{req.model_name}-tmp"
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=4,  # With LoRA we can increase batch size!
            gradient_accumulation_steps=4,
            learning_rate=2e-4,  # LoRA needs higher LR (usually 1e-4 ... 2e-4)
            warmup_ratio=0.05,
            fp16=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            # Optimizer for memory savings
            optim="paged_adamw_8bit" if req.use_qlora else "adamw_torch",
        )

        # FIX: Force single-GPU (n_gpu=1) to prevent Trainer from using DataParallel.
        # DataParallel is incompatible with bitsandbytes 4-bit quantization.
        if torch.cuda.device_count() > 1:
            logger.info(
                f"[{job_id}] Multiple GPUs detected. "
                "Forcing single-GPU (n_gpu=1) for LoRA compatibility."
            )
            args._n_gpu = 1

        loss = losses.MultipleNegativesRankingLoss(train_model)

        trainer = SentenceTransformerTrainer(
            model=train_model,
            args=args,
            train_dataset={"default": train_dataset},
            loss={"default": cast(nn.Module, loss)},
            callbacks=[JobProgressCallback(job_id)],
        )

        trainer.train()

        # 6. Saving
        # With LoRA we only save adapters.
        save_path = f"./models/{req.model_name}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        logger.info(f"[{job_id}] Saving LoRA adapter to {save_path}")
        train_model.save(save_path)

        # Cleanup
        del train_model
        del trainer
        engine._cleanup_memory()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        logger.error(f"[{job_id}] LoRA Training failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        # Try to reload serving model
        try:
            engine.load()
        except Exception:
            pass


@app.post("/vectorize-hybrid", response_model=HybridVectorResponse)
async def vectorize_hybrid(req: HybridTextRequest):
    """
    Endpoint for BGE-M3. Returns Dense, Sparse, and optionally ColBERT vectors.
    """
    if engine.bge_model is None:
        if engine.model is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hybrid encoding is available only for BGE-M3 models using "
                    "FlagEmbedding."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is initializing.",
        )

    try:
        # Pass only text and colbert flag
        result_list = engine.encode_hybrid(
            [req.text], return_colbert=req.return_colbert, batch_size=1
        )
        # result_list is a list of dicts, but we need to convert it to HybridVector model
        # However, pydantic should handle dict to model conversion
        return {"hybrid_vector": result_list[0]}
    except Exception as e:
        logger.error(f"Hybrid vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorize-batch-hybrid", response_model=BatchHybridVectorResponse)
async def vectorize_batch_hybrid(req: HybridBatchTextRequest):
    if engine.bge_model is None:
        if engine.model is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hybrid encoding is available only for BGE-M3 models using "
                    "FlagEmbedding."
                ),
            )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is initializing.",
        )

    try:
        results = engine.encode_hybrid(
            req.items, return_colbert=req.return_colbert, batch_size=16
        )
        return {"hybrid_vectors": results}
    except Exception as e:
        logger.error(f"Batch hybrid vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fine-tune")
async def fine_tune(req: TrainRequest, background_tasks: BackgroundTasks):
    # Validation before starting the task
    try:
        # Check if the current base model is supported for fine-tuning
        detect_model_profile(engine.model_name_env)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued"}
    background_tasks.add_task(train_worker, job_id, req)
    return {"job_id": job_id, "status": "pending"}


@app.post("/fine-tune-lora")
async def fine_tune_lora(req: LoraTrainRequest, background_tasks: BackgroundTasks):
    """
    Starts lightweight training (LoRA/Q-LoRA).
    Requires less memory, works faster.
    Result is saved as an adapter.
    """
    try:
        detect_model_profile(engine.model_name_env)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending", "progress": 0, "message": "Queued (LoRA)"}

    background_tasks.add_task(train_lora_worker, job_id, req)

    return {"job_id": job_id, "status": "pending", "type": "lora"}


@app.get("/train-status/{job_id}")
async def get_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
