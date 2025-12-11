# Embedding Service

This is a FastAPI-based service for generating text embeddings using the `intfloat/multilingual-e5-large` model. It supports both single text and batch processing.

## 🔥 Features
- **Multilingual Support**: Uses `intfloat/multilingual-e5-large`.
- **FastAPI**: High performance, easy to use.
- **GPU Support**: Automatically detects CUDA if available (requires proper PyTorch build).
- **Batch Processing**: Efficiently vectorize multiple texts at once.

## 🛠 Prerequisites
- [Docker](https://www.docker.com/) installed.
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

---

## 🚀 Running the Service

### 1️⃣ Run on GPU (Default)

The default configuration includes standard PyTorch with CUDA support. This is the recommended way to run if you have a GPU.

**Prerequisites:**
- Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

**Build the image:**
```bash
docker build -t embedding-service:gpu .
```

**Run with GPU access:**
You MUST pass `--gpus all` to enable GPU access inside the container.

```bash
docker run -d -p 8000:8000 --gpus all --name embedding-service embedding-service:gpu
```

> **Note:** If you run this image without `--gpus all` (or on a machine without a GPU), it will fall back to CPU, but the image size will be larger than the dedicated CPU-optimized version.

---

### 2️⃣ Run on CPU (Optimization)

If you do not have a GPU or want to save disk space, you can build a smaller CPU-only version.

**Step 1: Modify `requirements.txt`**
Change the line:
```
torch
```
to the CPU-specific wheel:
```
torch --index-url https://download.pytorch.org/whl/cpu
```

**Step 2: Build the image**
```bash
docker build -t embedding-service:cpu .
```

**Step 3: Run the container**
```bash
docker run -d -p 8000:8000 --name embedding-service-cpu embedding-service:cpu
```

## 📚 API Usage

### Health/Docs
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Generate Embedding (Single)
**Endpoint:** `POST /vectorize`

**Request:**
```json
{
  "text": "Hello world",
  "is_query": true
}
```

**Response:**
```json
{
  "vector": [0.123, -0.456, ...]
}
```

### Generate Embeddings (Batch)
**Endpoint:** `POST /vectorize-batch`

**Request:**
```json
{
  "items": [
    "Hello world",
    "Machine learning is great"
  ],
  "is_query": false
}
```

**Response:**
```json
{
  "vectors": [
    [0.123, ...],
    [0.789, ...]
  ]
}
```
