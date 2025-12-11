from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

# 1. Check GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on {device}...")

# 2. Load the model ONCE at startup
model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)

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
    items: list[str]
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
