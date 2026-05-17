# CUDA 12.6 runtime + cuDNN. "runtime" variant ships libcudart / cuDNN libs
# without nvcc. NeuroSIM C++ only needs g++/make, so this is enough.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_PREFERENCE=managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# uv installs itself + manages Python 3.13 inside the image.
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# --- Layer 1: dependency resolution only (best cache reuse) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# --- Layer 2: project source + editable TensorRT fork install ---
COPY pytorch-quantization ./pytorch-quantization
RUN uv pip install -e ./pytorch-quantization

# --- Layer 3: rest of the project + C++ NeuroSIM build ---
COPY . .
RUN cd NeuroSIM && make

EXPOSE 7860

# server_name=0.0.0.0 is already set in gradio_app.py for container reachability.
CMD ["uv", "run", "python", "-u", "gradio_app.py"]
