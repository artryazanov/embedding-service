# BGE-M3 Embedding Service

This is a high-performance, FastAPI-based microservice dedicated to generating text embeddings using the state-of-the-art **`BAAI/bge-m3`** model. Designed for international scalability, the architecture features a strictly validated configuration system and seamless CPU/GPU Docker deployments.

[![Tests](https://github.com/artryazanov/embedding-service/actions/workflows/tests.yml/badge.svg)](https://github.com/artryazanov/embedding-service/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/artryazanov/embedding-service/graph/badge.svg)](https://codecov.io/gh/artryazanov/embedding-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Versions](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

## 🔥 Core Features
- **Pydantic Driven**: Centralized and type-safe `.env` parsing via `pydantic-settings`.
- **Dedicated Engine**: Refactored OOP `EmbeddingEngine` tailored specifically for extracting embeddings safely and closing memory leaks reliably. It utilizes `asyncio.to_thread` with granular chunking to ensure massive batched vectorization requests never block the main event loop, allowing single requests to run simultaneously.
- **FastAPI Core**: A high-performance REST API managed by advanced application `lifespan` generators.
- **Smart Hardware Detection**: Automatically targets `cuda` if available and safely falls back to `cpu`. 
- **Modular Dockerfile**: A single Dockerfile handles both CPU and GPU builds natively via `ARG DEVICE`.

---

## 🛠 Configuration (`.env`)

To start, copy the example configuration.
```bash
cp .env.example .env
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `API_TOKEN` | Optional Bearer token for secure REST endpoints. | `None` |
| `MODEL_NAME` | The HuggingFace model path or local repository name. | `BAAI/bge-m3` |
| `MAX_SEQ_LENGTH` | Maximum tokens per sequence. | `8192` |
| `CHUNK_SIZE` | Batch processing chunk size (elements per single GPU array). | `64` |
| `DEVICE` | Target hardware. (`auto`, `cpu`, or `cuda`) | `auto` |

---

## 🚀 Running the Service (Docker)

### 1️⃣ Run on GPU (Recommended)
This requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

**Build the image:**
```bash
# DEVICE=gpu is the default argument
docker build -t embedding-service:gpu .
```

**Launch the container:**
```bash
docker run -d -p 8000:8000 --gpus all \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  --name embedding-service embedding-service:gpu
```

### 2️⃣ Run on CPU (Space & Compute Optimization)
If running on a standard server without GPU access, you can build a severely optimized environment relying on PyTorch's `cpu` wheels to drastically lower image weight.

**Build the optimized image:**
```bash
docker build --build-arg DEVICE=cpu -t embedding-service:cpu .
```

**Launch the container:**
```bash
docker run -d -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  --name embedding-service-cpu embedding-service:cpu
```

---

## 📚 REST API Usage

### Health & Capabilities (`GET /health`)
Check service availability, loaded model identity, and active hardware device.
```bash
curl -X GET "http://localhost:8000/health" \
     -H "Authorization: Bearer <API_TOKEN_IF_CONFIGURED>"
```

### Generate Single Embedding (`POST /vectorize`)
Extract a base embedding array for a single query or document.
```bash
curl -X POST "http://localhost:8000/vectorize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Artificial Intelligence is evolving rapidly."}'
```
**Response:**
```json
{
  "vector": [0.0123, -0.0456, 0.0789, ...]
}
```

### Generate Batch Embeddings (`POST /vectorize-batch`)
Compute multiple vectors highly optimally in a single pass. The engine explicitly breaks down massive payloads into smaller sub-batches (chunks of `CHUNK_SIZE` requests, 64 by default) yielding to the asynchronous event loop (`asyncio.sleep(0.001)`) between them. This architectural feature prevents GPU OOM errors and guarantees that isolated, single priority requests won't queue and timeout behind long 30+ second batches.

The endpoint uniquely supports three flexible payload formats for `items`:
1. **Dictionary Format (Recommended):** Perfect for product indexing. Pass a JSON object mapping unique IDs to texts. The vectors are instantly mapped strictly back to those specific IDs.
2. **Object Array Format:** Pass a standard array of objects: `[{"id": "...", "text": "..."}]`.
3. **Legacy Format:** A standard list of plain strings. 

**Example Request (Dictionary Format):**
```bash
curl -X POST "http://localhost:8000/vectorize-batch" \
     -H "Content-Type: application/json" \
     -d '{"items": {"prod_1": "First document segment.", "prod_2": "Second document segment."}}'
```
**Response:**
```json
{
  "vectors": {
    "prod_1": [0.0123, ...],
    "prod_2": [-0.0789, ...]
  }
}
```

---

## 🧪 Development & Testing

This project adheres explicitly to **Senior Python Developer Guidelines** featuring `pytest`, mock patching, `pytest-cov`, and `pytest-asyncio` strictly executing in a sandboxed `venv`.

1. **Activate Environment and Install dependencies:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Run the complete testing suite (Target: 90%+ Coverage):**
```bash
pytest tests/ -v -p no:warnings --cov=.
```

## License

This project is licensed under the [MIT License](LICENSE).