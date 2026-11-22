# Dockerfile
#
# Environment of the code that ran on the TU Wien dataLAB cluster: preprocessing
# step 10, training, evaluation and analysis. The packages are those of the
# "training" group in uv.lock. On x86_64 the PyTorch wheels include CUDA, so the
# container uses the GPU of the host with `docker run --gpus all`.
#
# Build and usage: see docs/reproducibility.md#docker.

FROM python:3.11-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

# Shared libraries required by OpenCV (preprocessing step 10)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /workspace
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project --no-default-groups --group training \
    && rm -rf /root/.cache/uv

COPY . .

# The training scripts are run from training/ with this folder on the Python path
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/training
WORKDIR /workspace/training

CMD ["bash"]
