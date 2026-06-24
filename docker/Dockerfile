# CUDA 12.8 runtime + cuDNN — required for the RTX 5080 (Blackwell, sm_120).
# Ubuntu 22.04 ships Python 3.10, matching the cu128 cp310 wheels we install.
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    NO_ALBUMENTATIONS_UPDATE=1

# Python 3.10 + system libs needed by OpenCV (libGL / glib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Install PyTorch (cu128) FIRST so the GPU build is not overridden by
# requirements.txt (which pulls the rest from PyPI).
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 8000 = inference API, 8501 = training dashboard.
EXPOSE 8000 8501

# Default command (overridden per service in docker-compose.yml).
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
