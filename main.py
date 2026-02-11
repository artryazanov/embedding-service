import gc
import logging
import os
import uuid
from dataclasses import dataclass
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

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Fix CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- Domain Models ---


@dataclass
class ModelProfile:
    """Describes the characteristics and requirements of a specific model."""

    name: str
    max_seq_length: int
    requires_prefix: bool

    @property
    def query_prefix(self) -> str:
        return "query: " if self.requires_prefix else ""

    @property
    def passage_prefix(self) -> str:
        return "passage: " if self.requires_prefix else ""


def detect_model_profile(model_name_path: str) -> ModelProfile:
    """
    Configuration factory. Determines settings based on the model name.
    Supports: E5 family, BGE-M3 family.
    """
    normalized_name = model_name_path.lower()

    if "e5" in normalized_name:
        logger.info(f"Detected E5-based model architecture for '{model_name_path}'")
        return ModelProfile(
            name=model_name_path,
            max_seq_length=512,
            requires_prefix=True,
        )

    if "bge-m3" in normalized_name:
        logger.info(f"Detected BGE-M3 architecture for '{model_name_path}'")
        return ModelProfile(
            name=model_name_path,
            max_seq_length=8192,
            requires_prefix=False,
        )

    # If the model is not recognized, raise an error to prevent incorrect operation
    error_msg = (
        f"Unsupported model architecture: '{model_name_path}'. "
        "System currently supports only 'multilingual-e5' and 'bge-m3' variants."
    )
    logger.error(error_msg)
    raise ValueError(error_msg)


# --- Service Layer ---


class EmbeddingEngine:
    """
    Manager class controlling the model lifecycle.
    Implements the Singleton pattern at the module level (via instance).
    """

    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.profile: Optional[ModelProfile] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name_env = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-large")

    def load(self):
        """Loads the model, applying settings appropriate to the profile."""
        self._cleanup_memory()

        try:
            # First determine the profile to ensure the model is supported
            self.profile = detect_model_profile(self.model_name_env)

            local_path = f"./models/{self.model_name_env}"
            load_source = (
                local_path if os.path.exists(local_path) else self.model_name_env
            )

            logger.info(f"Loading model from {load_source} to {self.device}...")

            self.model = SentenceTransformer(load_source, device=self.device)

            # If loading from network, save locally to cache
            if load_source == self.model_name_env:
                logger.info(f"Saving model to {local_path} for future use...")
                self.model.save(local_path)

            # Apply profile settings
            self.model.max_seq_length = self.profile.max_seq_length
            logger.info(
                f"Model loaded. Max Seq Len: {self.model.max_seq_length}, "
                f"Prefix: {self.profile.requires_prefix}"
            )

        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise e

    def unload(self):
        """Completely unloads the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        self._cleanup_memory()
        logger.info("Model unloaded. VRAM cleared.")

    def _cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(
        self,
        texts: List[str],
        is_query: bool,
        batch_size: int = 64,
    ) -> List[List[float]]:
        if self.model is None or self.profile is None:
            raise RuntimeError("Model is not loaded")

        # Logic for adding prefixes is encapsulated here
        prefix = self.profile.query_prefix if is_query else self.profile.passage_prefix

        # If prefix is empty (BGE-M3), just take the text, otherwise concatenate
        processed_texts = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            processed_texts,
            normalize_embeddings=True,
            convert_to_tensor=False,
            batch_size=batch_size,
            show_progress_bar=False,
        )

        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()

        return cast(List[List[float]], embeddings)


# Initialize Engine
engine = EmbeddingEngine()
app = FastAPI()

# --- API Events ---


@app.on_event("startup")
async def startup_event():
    try:
        engine.load()
    except ValueError as e:
        # If the model is not supported, the app should not start silently
        logger.error(f"Startup failed: {e}")
        # In real prod we could do os._exit(1), but here we leave it to see logs
        pass


# --- DTOs ---


class TextRequest(BaseModel):
    text: str
    is_query: bool = True


class BatchTextRequest(BaseModel):
    items: List[str]
    is_query: bool = False


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


# --- Endpoints ---


@app.post("/vectorize")
async def vectorize(req: TextRequest):
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training or initializing.",
        )

    try:
        # For a single request, wrap in a list, then extract the first element
        vector = engine.encode([req.text], is_query=req.is_query, batch_size=1)[0]
        return {"vector": vector}
    except Exception as e:
        logger.error(f"Vectorization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectorize-batch")
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
