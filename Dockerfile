# CUDA 12.8 devel + cuDNN. "devel" includes nvcc + headers (needed because
# pytorch-quantization compiles CUDA C++ extensions). 12.8+ is required to
# target sm_100 (Blackwell B200) and sm_120 (Blackwell RTX 50xx).
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

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
        ninja-build \
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
# Pin target CUDA archs here (close to the layer that uses it) so future
# arch-list changes only invalidate the pytorch-quantization build layer
# instead of forcing uv sync to re-download wheels. PyTorch auto-appends +PTX
# to the last entry; 12.0 = Blackwell / RTX 50xx, 10.0 = B200.
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
COPY pytorch-quantization ./pytorch-quantization
# pytorch-quantization's setup.py imports torch, but uv builds in an isolated
# env by default. Reuse the already-synced .venv (which has torch) for the
# build step via --no-build-isolation.
RUN uv pip install --no-build-isolation -e ./pytorch-quantization

# --- Layer 3: C++ NeuroSIM build (invalidated only when NeuroSIM source changes) ---
COPY NeuroSIM ./NeuroSIM
RUN cd NeuroSIM && make

# --- Layer 4: Python application source (changes most often — no compilation) ---
COPY . .

EXPOSE 7860

# server_name=0.0.0.0 is already set in gradio_app.py for container reachability.
CMD ["uv", "run", "python", "-u", "gradio_app.py"]
