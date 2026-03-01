# ---- base ----
FROM python:3.12-slim AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
ARG TORCH_CHANNEL=cpu   # "cpu" (default) or "cu124"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ pkg-config \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    libopenblas-dev liblapack-dev \
    ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    if [ "$TORCH_CHANNEL" = "cu124" ]; then \
      PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124" \
      pip install --no-cache-dir -r requirements.txt; \
    else \
      pip install --no-cache-dir -r requirements.txt; \
    fi

COPY . .

# ---- test ----
FROM base AS test
ENV PYTHONPATH=/app
RUN python -m pytest -q tests/pytests

# ---- runtime ----
FROM base AS runtime
CMD ["python", "-c", "import torch; print('Torch:', torch.__version__, '| CUDA:', getattr(torch.version,'cuda',None), '| Available:', torch.cuda.is_available())"]