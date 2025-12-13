from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, datasets, losses
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

        # 1. Read data
        train_sentences = req.text_content.splitlines()
        # Filter empty lines
        train_sentences = [s.strip() for s in train_sentences if s.strip()]

        if not train_sentences:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = "Content is empty"
            return

        print(f"[{job_id}] Loaded {len(train_sentences)} sentences.")

        # 2. Create dataset
        # TSDAE requires just a list of sentences
        train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

        # 3. DataLoader
        batch_size = 8 # Adjusted from 1 to 8 for better performance on large models like E5-large if GPU permits, usually 8 is okay. User had 1.
        # User code had batch_size=1. Reverting to 1? Encoded E5-large is big.
        # Let's stick closer to user provided code but maybe 4 or 8 is better.
        # The user provided code had batch_size=1. I will stick to the user's code to be safe,
        # but 1 is very slow. I'll use 4 as a compromise or stick to 1 if strict adaptation.
        # I'll stick to user's 1 for safety, as I don't know their VRAM.
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        # 4. Loss Function
        # We use the existing loaded model for training
        train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name_or_path, tie_encoder_decoder=True)

        # Calculate steps
        total_steps = len(train_dataloader)
        jobs[job_id]["total_steps"] = total_steps
        jobs[job_id]["message"] = "Training..."

        # 5. Start training
        print(f"[{job_id}] Starting fit...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            weight_decay=0,
            scheduler='constantlr',
            optimizer_params={'lr': 3e-5},
            show_progress_bar=False, # We use our own tracking
            callback=ProgressCallback(job_id, total_steps)
        )
        print(f"[{job_id}] Fit complete.")

        # 6. Save model
        save_dir = "./models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, model_name)
        model.save(save_path)
        print(f"[{job_id}] Model saved to {save_path}")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"] = f"Model saved to {save_path}"

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
