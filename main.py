import asyncio
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from config import settings
from engine import engine
from worker import websocket_worker_task

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


class BatchTextRequest(BaseModel):
    items: List[str]


class VectorResponse(BaseModel):
    vector: List[float]


class BatchVectorResponse(BaseModel):
    vectors: List[List[float]]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    engine.load()
    ws_task = asyncio.create_task(websocket_worker_task())
    yield
    # Shutdown
    ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass
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
        return {"vector": engine.encode([req.text])[0]}
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
        return {"vectors": engine.encode(req.items, batch_size=64)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
