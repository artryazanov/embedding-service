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

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
