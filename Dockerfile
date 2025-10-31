# Start from lightweight official PyTorch image
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# ---- Build and install SAM2 (cached layer) ----
WORKDIR /opt/ml/code/sam2
# Copy only requirements + setup first so Docker caches this unless SAM2 changes
RUN git clone --depth=1 https://github.com/facebookresearch/sam2.git /opt/ml/code/sam2 && \
    pip install --no-cache-dir -r requirements.txt || true && \
    pip install --no-cache-dir . && \
    python setup.py build_ext --inplace && \
    rm -rf ~/.cache/pip

ENV PYTHONPATH=/opt/ml/code/sam2:$PYTHONPATH

# ---- Application Code ----
WORKDIR /opt/ml/code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm -rf ~/.cache/pip

# Copy only the minimal inference-related code (faster Docker context)
COPY inference.py .
COPY config/sam2_config.yaml ./config/sam2_config.yaml
COPY utility ./utility

# Environment variables for cleaner logs and deterministic behavior
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# ---- Entrypoint ----
ENTRYPOINT ["python", "inference.py"]
