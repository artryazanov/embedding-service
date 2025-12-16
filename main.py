import gc
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import uuid
import torch
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, losses, InputExample, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from transformers import TrainerCallback
import datasets

app = FastAPI()

# --- Global Variables ---
# Store the model here. If None, the model is unloaded (training in progress)
serving_model: Optional[SentenceTransformer] = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name_env = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-large')

# --- Memory Management ---

def load_serving_model():
    """Loads the serving model into the global variable."""
    global serving_model
    
    # First, clean up memory just in case
    cleanup_memory()
    
    local_path = f"./models/{model_name_env}"
    path_to_load = local_path if os.path.exists(local_path) else model_name_env
    
    print(f"[SYSTEM] Loading serving model from {path_to_load} to {device}...")
    serving_model = SentenceTransformer(path_to_load, device=device)
    serving_model.max_seq_length = 512
    print("[SYSTEM] Serving model ready.")

def unload_model():
    """Completely removes the model from memory."""
    global serving_model
    if serving_model is not None:
        del serving_model
        serving_model = None
    cleanup_memory()
    print("[SYSTEM] Model unloaded. VRAM cleared.")

def cleanup_memory():
    """Forced garbage collection and CUDA cache cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Load the model at application startup
@app.on_event("startup")
async def startup_event():
    load_serving_model()

# --- Model API ---

class TextRequest(BaseModel):
    text: str
    is_query: bool = True

class BatchTextRequest(BaseModel):
    items: List[str]
    is_query: bool = False

@app.post("/vectorize")
async def vectorize(req: TextRequest):
    if serving_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training. Please try again later."
        )
    
    prefix = "query: " if req.is_query else "passage: "
    embedding = serving_model.encode(prefix + req.text, normalize_embeddings=True, convert_to_tensor=False)
    
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return {"vector": embedding}

@app.post("/vectorize-batch")
async def vectorize_batch(req: BatchTextRequest):
    if serving_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is currently training. Please try again later."
        )

    prefix = "query: " if req.is_query else "passage: "
    full_texts = [prefix + text for text in req.items]

    # Batch size 32-64 is optimal for inference on 4080
    embeddings = serving_model.encode(
        full_texts,
        normalize_embeddings=True,
        convert_to_tensor=False,
        batch_size=64, 
        show_progress_bar=False
    )

    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return {"vectors": embeddings}

# --- Training Logic ---

jobs: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    text_content: str
    model_name: str # Name of the directory to save the model to

class JobProgressCallback(TrainerCallback):
    """
    Custom callback to update the global jobs dictionary with training progress.
    """
    def __init__(self, job_id):
        self.job_id = job_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.job_id in jobs and state.max_steps > 0:
            # state.global_step is the number of update steps
            progress = int((state.global_step / state.max_steps) * 100)
            jobs[self.job_id]["progress"] = progress
            jobs[self.job_id]["steps"] = state.global_step

def train_worker(job_id: str, req: TrainRequest):
    try:
        jobs[job_id]["status"] = "running"
        print(f"[{job_id}] Starting training process...")

        # 1. MEMORY CLEANUP
        print(f"[{job_id}] Unloading serving model...")
        unload_model() # Your cleanup function

        # 2. Data
        raw_lines = req.text_content.splitlines()
        train_examples = []
        for line in raw_lines:
            line = line.strip()
            if not line: continue
            parts = line.split(" Характеристики: ")
            if len(parts) == 2:
                q = f"query: {parts[0].strip()}"
                p = f"passage: {parts[1].strip()}"
                train_examples.append(InputExample(texts=[q, p]))

        if not train_examples:
            raise ValueError("No valid pairs found")
        
        print(f"[{job_id}] Data: {len(train_examples)} pairs.")

        # Convert to Hugging Face Dataset
        train_dataset = datasets.Dataset.from_dict({
            "sentence_0": [e.texts[0] for e in train_examples],
            "sentence_1": [e.texts[1] for e in train_examples],
        })

        # 3. Loading
        local_base = f"./models/{model_name_env}"
        load_path = local_base if os.path.exists(local_base) else model_name_env
        print(f"[{job_id}] Loading train model from {load_path}...")
        
        train_model = SentenceTransformer(load_path, device=device)
        train_model.max_seq_length = 512
        
        # 4. TRAINING CONFIGURATION
        # We use a very small physical batch size (1) to avoid OOM, 
        # but a high accumulation (32) to get an effective batch size of 32.
        PHYSICAL_BATCH_SIZE = 1
        ACCUMULATION_STEPS = 32 
        
        output_dir = f"./models/{req.model_name}-tmp"
        
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=PHYSICAL_BATCH_SIZE,
            gradient_accumulation_steps=ACCUMULATION_STEPS,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            fp16=True,  # Enable mixed precision for memory saving
            gradient_checkpointing=True, # Save even more memory
            gradient_checkpointing_kwargs={"use_reentrant": False}, # Fix warning and stability
            logging_steps=10,
            save_strategy="no",
            report_to="none", # Disable wandb etc
            dataloader_num_workers=0, # Ensure no multiprocessing overhead
            optim="adafactor" # Use Adafactor instead of AdamW for massive memory savings
        )

        loss = losses.MultipleNegativesRankingLoss(train_model)

        trainer = SentenceTransformerTrainer(
            model=train_model,
            args=args,
            train_dataset={"default": train_dataset},
            loss={"default": loss},
            callbacks=[JobProgressCallback(job_id)]
        )

        # 5. Start
        print(f"[{job_id}] Starting Trainer with Batch={PHYSICAL_BATCH_SIZE}, Accumulation={ACCUMULATION_STEPS}...")
        trainer.train()

        # 6. Saving
        save_path = f"./models/{req.model_name}"
        if not os.path.exists("./models"): os.makedirs("./models")
        
        # Save the final model
        train_model.save(save_path)
        print(f"[{job_id}] Saved to {save_path}")

        # Cleanup tmp dir if needed, but SentenceTransformerTrainer usually handles its own checkpoints
        # if save_strategy is "no", it won't clutter much.

        del train_model
        cleanup_memory()
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100

    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    
    finally:
        print(f"[{job_id}] Reloading serving model...")
        try:
            load_serving_model()
        except Exception:
            pass

@app.post("/fine-tune")
async def fine_tune(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Queued"
    }
    # Run in background
    background_tasks.add_task(train_worker, job_id, req)
    return {"job_id": job_id, "status": "pending"}

@app.get("/train-status/{job_id}")
async def get_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]