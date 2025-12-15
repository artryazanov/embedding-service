from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, datasets, losses, InputExample
from torch.utils.data import DataLoader
import torch
import os
import uuid
import nltk
from typing import Dict, Any, List
import os

app = FastAPI()

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('punkt_tab')

# 1. Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

# 2. Load the model ONCE at startup
model_name = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-large')
local_model_path = f"./models/{model_name}"

if os.path.exists(local_model_path):
    print(f"Loading local model from {local_model_path} on {device}...")
    model_name_or_path = local_model_path
else:
    print(f"Loading model {model_name} from HuggingFace on {device}...")
    model_name_or_path = model_name

model = SentenceTransformer(model_name_or_path, device=device)
model.max_seq_length = 512

# --- Vectorization Logic ---

class TextRequest(BaseModel):
    text: str
    is_query: bool = True # Flag: is this a user query or product indexing?

@app.post("/vectorize")
async def vectorize(req: TextRequest):
    # Add correct prefixes for E5
    prefix = "query: " if req.is_query else "passage: "
    full_text = prefix + req.text

    # Generate vector
    embedding = model.encode(full_text, normalize_embeddings=True, convert_to_tensor=False)

    # Convert numpy array to list so FastAPI can serialize it
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    return {"vector": embedding}


class BatchTextRequest(BaseModel):
    items: List[str]
    is_query: bool = False  # Default to False (passage) for bulk operations


@app.post("/vectorize-batch")
async def vectorize_batch(req: BatchTextRequest):
    # Add correct prefixes for E5
    prefix = "query: " if req.is_query else "passage: "
    full_texts = [prefix + text for text in req.items]

    # Generate vectors in batch
    # batch_size can be tuned, but 32 is a reasonable default
    embeddings = model.encode(
        full_texts,
        normalize_embeddings=True,
        convert_to_tensor=False,
        batch_size=32,
        show_progress_bar=False
    )

    # Convert numpy array to list of lists
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()

    return {"vectors": embeddings}

# --- Training / Fine-tuning Logic ---

# Job storage
jobs: Dict[str, Dict[str, Any]] = {}

class TrainRequest(BaseModel):
    text_content: str
    model_name: str

class ProgressCallback:
    def __init__(self, job_id, total_steps):
        self.job_id = job_id
        self.total_steps = total_steps

    def __call__(self, score, epoch, steps):
        # Update job progress
        if self.job_id in jobs:
            # Avoid division by zero
            if self.total_steps > 0:
                jobs[self.job_id]["progress"] = int((steps / self.total_steps) * 100)
            jobs[self.job_id]["steps"] = steps
            jobs[self.job_id]["epoch"] = epoch

def run_training_task(job_id: str, req: TrainRequest):
    try:
        model_name = req.model_name
        jobs[job_id]["status"] = "running"
        jobs[job_id]["message"] = "Preparing data..."
        
        # --- GPU MEMORY MANAGEMENT START ---
        # Move the global serving model to CPU to free up VRAM for training
        global model
        print(f"[{job_id}] Moving global serving model to CPU to free VRAM...")
        model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # -----------------------------------

        try:
            # 1. Parsing and splitting data into pairs
            raw_lines = req.text_content.splitlines()
            train_examples = []
            
            for line in raw_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Try to split Title and Characteristics+Description
                # In your data the separator is " Характеристики: "
                parts = line.split(" Характеристики: ")
                
                if len(parts) == 2:
                    # part[0] = Title + Category
                    # part[1] = Characteristics + Description itself
                    
                    # FORM A CORRECT PAIR FOR E5
                    # The model learns to bring closer the "characteristics" vector and the "description" vector
                    q_text = f"query: {parts[0].strip()}" 
                    p_text = f"passage: {parts[1].strip()}"
                    
                    train_examples.append(InputExample(texts=[q_text, p_text]))

            if not train_examples:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = "No valid data pairs found (check 'Характеристики:' separator within the data)"
                return

            print(f"[{job_id}] Created {len(train_examples)} training pairs.")

            # 2. Load a FRESH model instance for training
            # We start from the base model defined in environment, or default.
            env_model_name = os.getenv('MODEL_NAME', 'intfloat/multilingual-e5-large')
            local_base_path = f"./models/{env_model_name}"
            
            # Determine where to load the BASE model from
            if os.path.exists(local_base_path):
                print(f"[{job_id}] Loading base model from local path: {local_base_path}...")
                train_model_path = local_base_path
            else:
                print(f"[{job_id}] Loading base model from HuggingFace: {env_model_name}...")
                train_model_path = env_model_name

            # Initialize the fresh model for training
            print(f"[{job_id}] Initializing fresh SentenceTransformer for training from {train_model_path}...")
            train_model = SentenceTransformer(train_model_path, device=device)
            train_model.max_seq_length = 512

            # 3. DataLoader
            # IMPORTANT: For MultipleNegativesRankingLoss, batch_size must be > 1
            # The larger the batch, the better the quality (since other batch elements serve as negative examples)
            # For RTX 4080, you can safely set 16 or 32.
            train_dataloader = DataLoader(train_examples, batch_size=4, shuffle=True)

            # 4. Loss Function (The most powerful function for retrieval)
            # It says: "Vector A should be similar to Vector B, and dissimilar to all other vectors in this batch"
            train_loss = losses.MultipleNegativesRankingLoss(train_model)

            # Step count
            total_steps = len(train_dataloader) # 1 epoch is usually enough for fine-tuning
            jobs[job_id]["total_steps"] = total_steps
            jobs[job_id]["message"] = "Training (Contrastive Learning)..."

            # 5. Start training
            print(f"[{job_id}] Starting fit...")
            train_model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1, # One epoch is enough! Otherwise it typically overfits.
                warmup_steps=int(total_steps * 0.1), # 10% warmup
                optimizer_params={'lr': 2e-5},       # Careful Learning Rate
                show_progress_bar=False,
                use_amp=True,
                callback=ProgressCallback(job_id, total_steps)
            )
            print(f"[{job_id}] Fit complete.")

            # 6. Save
            save_dir = "./models"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            save_path = os.path.join(save_dir, model_name)
            train_model.save(save_path)
            print(f"[{job_id}] Model saved to {save_path}")

            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["message"] = f"Model saved to {save_path}"

        finally:
            # --- GPU MEMORY MANAGEMENT END ---
            # Restore the global serving model to GPU
            print(f"[{job_id}] Restoring global serving model to {device}...")
            model.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[{job_id}] Serving model restored.")
            # ---------------------------------

    except Exception as e:
        print(f"[{job_id}] Error: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/train-tsdae")
async def train_tsdae(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "error": None
    }
    background_tasks.add_task(run_training_task, job_id, req)
    return {"job_id": job_id, "status": "pending"}

@app.get("/train-status/{job_id}")
async def get_train_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
