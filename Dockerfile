# Base: PyTorch 2.5 + CUDA 12.4 (yakın senin local'e)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /workspace

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY tests/ ./tests/

# Install Python dependencies + the package itself
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir -e ".[dev]"

# Default entrypoint — interactive bash
CMD ["bash"]