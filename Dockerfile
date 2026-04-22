# ── AI4AnimationsPy web demo — local Docker build ────────────
FROM python:3.12-slim

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# System deps (OpenGL stubs for pygltflib, glib for general use)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + web demo dependencies
COPY requirements-hf.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        --prefer-binary --no-compile torch && \
    python -m pip install --prefer-binary --no-compile -r requirements-hf.txt

# Framework source (imported via sys.path in run.py)
COPY ai4animation/ ai4animation/

# Assets
COPY Demos/_ASSETS_/Geno/ Demos/_ASSETS_/Geno/
COPY Demos/_ASSETS_/Quadruped/ Demos/_ASSETS_/Quadruped/

# Locomotion biped demo
COPY Demos/Locomotion/Biped/Network.pt      Demos/Locomotion/Biped/Network.pt
COPY Demos/Locomotion/Biped/Guidances/      Demos/Locomotion/Biped/Guidances/
COPY Demos/Locomotion/Biped/LegIK.py        Demos/Locomotion/Biped/LegIK.py
COPY Demos/Locomotion/Biped/Sequence.py     Demos/Locomotion/Biped/Sequence.py
COPY Demos/Locomotion/Biped/WebProgram.py   Demos/Locomotion/Biped/WebProgram.py

# Locomotion quadruped demo
COPY Demos/Locomotion/Quadruped/Network.pt      Demos/Locomotion/Quadruped/Network.pt
COPY Demos/Locomotion/Quadruped/Guidances/      Demos/Locomotion/Quadruped/Guidances/
COPY Demos/Locomotion/Quadruped/LegIK.py        Demos/Locomotion/Quadruped/LegIK.py
COPY Demos/Locomotion/Quadruped/Sequence.py     Demos/Locomotion/Quadruped/Sequence.py
COPY Demos/Locomotion/Quadruped/WebProgram.py   Demos/Locomotion/Quadruped/WebProgram.py

# Go1 physics puppeteer: MuJoCo model + STL meshes + retarget helper.
# The quadruped WebProgram imports go1_puppeteer.py at startup.
COPY Demos/Go1/                                 Demos/Go1/

# Web demo source + client
COPY Demos/run.py Demos/run.py
COPY Demos/server.py Demos/server.py
COPY Demos/client/ Demos/client/

EXPOSE 7860

# Local-only config: no session queue, no browser open
ENV NO_BROWSER=1
ENV PORT=7860
ENV PRELOAD_DEMOS=all

# CPU thread config for M4 Max
ENV MAX_CPU_THREADS=8
ENV OMP_NUM_THREADS=8
ENV MKL_NUM_THREADS=8
ENV TORCH_NUM_THREADS=8

ENV NETWORK_ITERATIONS=3
ENV PREDICTION_FPS=10

ENV TORCH_COMPILE_DISABLE=1
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/health')"

CMD ["python", "Demos/run.py"]
