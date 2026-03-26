from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import settings
from engine import engine

security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    if settings.api_token:
        if not credentials or credentials.credentials != settings.api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials


# --- DTOs ---
class TextRequest(BaseModel):
    text: str


class VectorResponse(BaseModel):
    vector: List[float]


class ItemRequest(BaseModel):
    id: Union[str, int]
    text: str


class ItemResponse(BaseModel):
    id: Union[str, int]
    vector: List[float]


class BatchTextRequest(BaseModel):
    items: Union[Dict[str, str], List[ItemRequest], List[str]]


class BatchVectorResponse(BaseModel):
    vectors: Union[Dict[str, List[float]], List[ItemResponse], List[List[float]]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    engine.load()
    yield
    # Shutdown
    engine.unload()


app = FastAPI(lifespan=lifespan, title="BGE-M3 Embedding Service")


@app.get("/health", dependencies=[Depends(verify_token)])
async def health():
    if engine.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )
    return {"status": "ok", "model": settings.model_name, "device": engine.device}


@app.post(
    "/vectorize", response_model=VectorResponse, dependencies=[Depends(verify_token)]
)
async def vectorize(req: TextRequest):
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")
    try:
        vector = await engine.encode_async([req.text])
        return {"vector": vector[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/vectorize-batch",
    response_model=BatchVectorResponse,
    dependencies=[Depends(verify_token)],
)
async def vectorize_batch(req: BatchTextRequest):
    if engine.model is None:
        raise HTTPException(status_code=503, detail="Model is initializing.")
    try:
        if not req.items:
            return {"vectors": []}

        if isinstance(req.items, dict):
            # Dict[str, str] format mapping ID -> text
            keys = list(req.items.keys())
            texts = list(req.items.values())
            vectors = await engine.encode_batch_chunked_async(texts)
            return {"vectors": {k: v for k, v in zip(keys, vectors)}}

        elif isinstance(req.items, list):
            if isinstance(req.items[0], str):
                # Legacy List[str] format
                items = req.items  # type: ignore
                vectors = await engine.encode_batch_chunked_async(items)
                return {"vectors": vectors}
            else:
                # List[ItemRequest] format
                keys = [item.id for item in req.items]  # type: ignore
                texts = [item.text for item in req.items]  # type: ignore
                vectors = await engine.encode_batch_chunked_async(texts)
                return {
                    "vectors": [{"id": k, "vector": v} for k, v in zip(keys, vectors)]
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
