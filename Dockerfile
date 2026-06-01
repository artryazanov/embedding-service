FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Default build is for GPU (standard PyTorch). 
# For CPU build use: docker build --build-arg DEVICE=cpu -t my-image .
ARG DEVICE=gpu

# Install different PyTorch versions depending on the argument
RUN if [ "$DEVICE" = "cpu" ]; then \
        pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Pre-download the bge-m3 model so it's baked into the image
RUN python -c "import os; os.makedirs('./models', exist_ok=True); from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3').save('./models/BAAI_bge-m3')" \
    && rm -rf /root/.cache/huggingface

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
