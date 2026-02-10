# Embedding Service

This is a FastAPI-based service for generating text embeddings, supporting multiple architectures like `intfloat/multilingual-e5-large` and `BAAI/bge-m3`. It automatically configures prefixes and sequence lengths based on the selected model. It supports both single text and batch processing.

[![Tests](https://github.com/artryazanov/embedding-service/actions/workflows/tests.yml/badge.svg)](https://github.com/artryazanov/embedding-service/actions/workflows/tests.yml)
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)
![Python Versions](https://img.shields.io/badge/python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)

## 🔥 Features
- **Multi-Architecture Support**: Automatically detects and configures for `E5` (prefixes, 512 seq) and `BGE-M3` (no prefixes, 8192 seq) models.
- **Multilingual Support**: Default: `intfloat/multilingual-e5-large`.
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
docker run -d -p 8000:8000 --gpus all \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME=intfloat/multilingual-e5-large \
  --name embedding-service embedding-service:gpu
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
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME=intfloat/multilingual-e5-large \
  --name embedding-service-cpu embedding-service:cpu
```

---

### 3️⃣ Using Custom/Local Models

By default, the service downloads the model specified by `MODEL_NAME` (default: `intfloat/multilingual-e5-large`) from HuggingFace.

To use a **local model** (e.g., for offline usage or a custom fine-tuned model):

1. **Create a `models` directory** on your host machine.
2. **Download or place your model** inside this directory.
   - Example structure:
     ```
     ./models/
     └── my-custom-model/
         ├── config.json
         ├── pytorch_model.bin
         ├── tokenizer.json
         └── ...
     ```
3. **Run the container** with:
   - `-v $(pwd)/models:/app/models`: Mounts your local models directory into the container.
   - `-e MODEL_NAME=my-custom-model`: Tells the app which model folder to look for.

**Example Command:**
```bash
docker run -d -p 8000:8000 --gpus all \
  -v $(pwd)/models:/app/models \
  -e MODEL_NAME=my-custom-model \
  embedding-service:gpu
```

The service will check if `/app/models/my-custom-model` exists.
- If **Yes**: It loads the model locally.
- If **No**: It attempts to download `my-custom-model` from HuggingFace.

### 4️⃣ Supported Models & Auto-Configuration

The service uses a **Smart Strategy Pattern** to automatically configure itself based on the model name in `MODEL_NAME`:

| Model Family | Detected By | Max Sequence Length | Prefixes (query: / passage:) |
| :--- | :--- | :--- | :--- |
| **E5** | "e5" in name | 512 | ✅ Yes |
| **BGE-M3** | "bge-m3" in name | 8192 | ❌ No |
| **Other** | - | - | ❌ Error (Unsupported) |

> **Note:** If you use a custom fine-tuned model, ensure its name contains "e5" or "bge-m3" so the service knows how to handle it.

### 🔗 Model Details

- **[intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)**
  - A specific model from the **E5** family (`E5-Large-V2`).
  - **Languages**: 94+ languages.
  - **Context**: 512 tokens.
  - **Prefixes**: Mandatory (`query: ` for queries, `passage: ` for documents).

- **[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)**
  - A state-of-the-art multilingual model (100+ languages).
  - **Context**: 8192 tokens (supports long documents).
  - **Prefixes**: Not required / automatically handled by the model's tokenizer (though this service treats it as no-prefix by default).

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

### Train/Fine-tune Model
**Endpoint:** `POST /fine-tune`

Starts a background entry to fine-tune the model using Contrastive Learning (MultipleNegativesRankingLoss).

**Input Format:**
Each line in `text_content` must follow the format:
`Query Text<TAB>Passage Text`

The separator is a **tab character** (`\t`).
Lines that do not match this format will be ignored.

**Request:**
```json
{
  "text_content": "Product A\tDescription of Product A\nProduct B\tDescription of Product B",
  "model_name": "my-finetuned-model"
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "pending"
}
```

### Check Training Status
**Endpoint:** `GET /train-status/{job_id}`

Checks the progress of a running training job.

**Response:**
```json
{
  "status": "running",
  "progress": 45,
  "message": "Training...",
  "error": null,
  "steps": 100,
  "epoch": 0
}
```

## 🧪 Development & Testing

### Running Tests Locally
To run the test suite, you need to install the development dependencies:

```bash
pip install -r requirements.txt
pip install pytest httpx
```

Then run the tests:

```bash
pytest tests/
```

## License

This project is released under the [Unlicense](LICENSE).