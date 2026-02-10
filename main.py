import gc
import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from sentence_transformers import (
    SentenceTransformer, 
    losses, 
    InputExample, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments
)
from transformers import TrainerCallback
import datasets

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
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
            requires_prefix=True
        )
    
    if "bge-m3" in normalized_name:
        logger.info(f"Detected BGE-M3 architecture for '{model_name_path}'")
        return ModelProfile(
            name=model_name_path,
            max_seq_length=8192,
            requires_prefix=False
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name_env = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-large')

    def load(self):
        """Loads the model, applying settings appropriate to the profile."""
        self._cleanup_memory()
        
        try:
            # First determine the profile to ensure the model is supported
            self.profile = detect_model_profile(self.model_name_env)
            
            local_path = f"./models/{self.model_name_env}"
            load_source = local_path if os.path.exists(local_path) else self.model_name_env
            
            logger.info(f"Loading model from {load_source} to {self.device}...")
            
            self.model = SentenceTransformer(load_source, device=self.device)
            
            # If loading from network, save locally to cache
            if load_source == self.model_name_env:
                logger.info(f"Saving model to {local_path} for future use...")
                self.model.save(local_path)

            # Apply profile settings
            self.model.max_seq_length = self.profile.max_seq_length
            logger.info(f"Model loaded. Max Sequence Length: {self.model.max_seq_length}, Prefix Required: {self.profile.requires_prefix}")

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

    def encode(self, texts: List[str], is_query: bool, batch_size: int = 64) -> List[List[float]]:
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
            show_progress_bar=False
        )
        
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
            
        return embeddings

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

# --- Endpoints ---

@app.post("/vectorize")
async def vectorize(req: TextRequest):
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training or initializing."
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
            detail="Model is currently training or initializing."
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

        # Parse data
        raw_lines = req.text_content.splitlines()
        train_examples = []
        
        # IMPORTANT: When training, we must also consider whether prefixes are needed.
        # Usually E5 is fine-tuned with prefixes, while BGE is not.
        # For simplicity here, we assume the user provides "clean" text,
        # and prefixes are added by the code below (if we were using engine.profile).
        # BUT: since the model is unloaded, profile might be unavailable or we are training a new model.
        # Here we follow the old logic: accept data as is, but form InputExample
        # with explicit prefix addition only if we are sure about the architecture.
        
        # For reliability within this task, training is better conducted with the same rule:
        # Determine the base architecture.
        base_profile = detect_model_profile(engine.model_name_env) 
        
        q_prefix = base_profile.query_prefix
        p_prefix = base_profile.passage_prefix

        for line in raw_lines:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) == 2:
                # Add prefixes to training data ONLY if the model requires it
                q = f"{q_prefix}{parts[0].strip()}"
                p = f"{p_prefix}{parts[1].strip()}"
                train_examples.append(InputExample(texts=[q, p]))

        if not train_examples:
            raise ValueError("No valid pairs found")
        
        logger.info(f"[{job_id}] Data prepared: {len(train_examples)} pairs. Prefixes used: {base_profile.requires_prefix}")

        train_dataset = datasets.Dataset.from_dict({
            "sentence_0": [e.texts[0] for e in train_examples],
            "sentence_1": [e.texts[1] for e in train_examples],
        })

        # Load model for training
        load_path = f"./models/{engine.model_name_env}"
        if not os.path.exists(load_path):
            load_path = engine.model_name_env
            
        logger.info(f"[{job_id}] Loading base model from {load_path}...")
        train_model = SentenceTransformer(load_path, device=engine.device)
        train_model.max_seq_length = base_profile.max_seq_length # Use correct length
        
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
            optim="adafactor"
        )

        loss = losses.MultipleNegativesRankingLoss(train_model)

        trainer = SentenceTransformerTrainer(
            model=train_model,
            args=args,
            train_dataset={"default": train_dataset},
            loss={"default": loss},
            callbacks=[JobProgressCallback(job_id)]
        )

        trainer.train()

        save_path = f"./models/{req.model_name}"
        if not os.path.exists("./models"): os.makedirs("./models")
        
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

@app.post("/fine-tune")
async def fine_tune(req: TrainRequest, background_tasks: BackgroundTasks):
    # Validation before starting the task
    try:
        # Check if the current base model is supported for fine-tuning
        detect_model_profile(engine.model_name_env)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Queued"
    }
    background_tasks.add_task(train_worker, job_id, req)
    return {"job_id": job_id, "status": "pending"}

@app.get("/train-status/{job_id}")
async def get_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]